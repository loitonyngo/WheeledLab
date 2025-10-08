"""
Utilities for map creation, waypoint-based pose sampling, and Frenet helpers.

This module provides:
- `create_maps_from_waypoints`: builds a USD stage with drivable maps and caches
  per-map arrays (waypoints, track bounds, curvatures, opponent trajectories, etc.)
- Pose generators used at reset-time:
  * grid-based random poses from traversability bitmaps
  * waypoint-based random poses
  * paired ego/opponent waypoint-based poses (with initial spacing & velocities)
  * deterministic start poses (e.g., at the first waypoint)
- `find_frenet_coord_along_waypoints`: fast nearest waypoint + signed lateral
  distance (left positive) for many positions at once.

Notes
-----
- Functions are vectorized where practical and support batches across multiple
  maps by indexing with `map_level`.
- All return values and external function signatures are preserved to avoid
  regressions elsewhere in the project.
"""

from __future__ import annotations

import os
import random
import time
from typing import List, Sequence, Tuple

import numpy as np
import torch  # required by several functions below

from .traversability_utils import *  # noqa: F401,F403  (project-internal utilities)
from .maps_utils import *            # noqa: F401,F403  (project-internal utilities)

from wheeledlab_tasks.config_loader import load_config

CONFIG = load_config()


# ---------------------------------------------------------------------------
# Map creation / USD stage helpers
# ---------------------------------------------------------------------------

def create_maps_from_waypoints(
    maps_folder_path: str,
    map_name_list: Sequence[str],
    origin_list: Sequence[Sequence[float]],
    stage_path: str,
    resolution: float,
):
    """
    Create drivable maps from stored waypoint jsons and write them into a USD stage.

    For each map in `map_name_list`, this function:
      1) Loads map geometry (centerline, inner/outer bounds) + derived track data.
      2) Computes a common square map size (pixels/meters) so all maps align.
      3) Writes waypoints/track bounds/opponent trajectories into the stage
         (optionally visible, according to config).
      4) Collects and returns per-map arrays needed by the environment.

    Args
    ----
    maps_folder_path : str
        Root folder containing one subfolder per map.
    map_name_list : Sequence[str]
        Names of map subfolders. Each should include 'global_waypoints.json'.
    origin_list : Sequence[Sequence[float]]
        Per-map world origins [x, y, z] used to place geometry in the stage.
    stage_path : str
        Path to the USD stage to create or open.
    resolution : float
        Grid resolution (meters per pixel) for drivable map generation.

    Returns
    -------
    tuple:
        (
          waypoints_list,        # list of [N_i, 3] lists (x,y,z) per map
          outer_list,            # list of [N_i, 3] lists (x,y,z) per map
          inner_list,            # list of [N_i, 3] lists (x,y,z) per map
          d_lat_list,            # list of [N_i, 2] lists (left,right) per map
          psi_rad_list,          # list of [N_i, 1] heading angles per map
          kappa_radpm_list,      # list of [N_i, 1] curvature per map
          vx_mps_list,           # list of [N_i, 1] reference speeds per map
          opp_traj_center_list,  # list of [N_i, 4] (x,y,psi,vel) per map
          opp_traj_iqp_list,     # list of [N_i, 4] (x,y,psi,vel) per map
          opp_traj_sp_list,      # list of [N_i, 4] (x,y,psi,vel) per map
          spacing_meters_list,   # list of [dx, dy] per map (grid spacing)
          map_size_pixels_list   # list of [H, W] per map (grid size in px)
        )

    Notes
    -----
    - Visibility of boundaries and trajectories in USD is controlled by
      CONFIG['env_config']['BOUNDARY_VISIBLE'] and ['TRAJ_VISIBLE'].
    - The drivable map bitmap is currently not returned (kept aligned with
      original code behavior).
    """
    # Open / create the USD stage once — used for all maps
    stage = set_stage_usd(stage_path)

    # -------- First pass: load all map data; find required common square size.
    map_data = []
    min_common_square_size = 0

    for i, map_name in enumerate(map_name_list):
        waypoints_path = os.path.join(maps_folder_path, map_name, "global_waypoints.json")

        (
            waypoints,
            trackbounds,
            d_lat,
            psi_rad,
            kappa_radpm,
            vx_mps,
            opp_traj_center,
            opp_traj_iqp,
            opp_traj_sp,
        ) = load_waypoints(waypoints_path)

        # Split/align bounds and ensure open-loop matching (project-specific utils)
        outer, inner = separate_and_align_bounds(waypoints, trackbounds)
        outer, inner = match_boundaries_open(waypoints, outer, inner)

        # Cache everything needed for the second pass
        map_data.append(
            dict(
                waypoints=waypoints,
                trackbounds=trackbounds,
                d_lat=d_lat,
                psi_rad=psi_rad,
                kappa_radpm=kappa_radpm,
                vx_mps=vx_mps,
                outer=outer,
                inner=inner,
                opp_traj_center=opp_traj_center,
                opp_traj_iqp=opp_traj_iqp,
                opp_traj_sp=opp_traj_sp,
            )
        )

        # Track the largest minimum square size required to fit this track
        min_map_size = get_min_map_size(outer, resolution)
        if min_map_size > min_common_square_size:
            min_common_square_size = min_map_size

    # -------- Second pass: create drivable maps + write USD prims; collect outputs.
    waypoints_list, outer_list, inner_list = [], [], []
    d_lat_list, psi_rad_list, kappa_radpm_list, vx_mps_list = [], [], [], []
    spacing_meters_list, map_size_pixels_list = [], []

    opp_traj_center_list, opp_traj_iqp_list, opp_traj_sp_list = [], [], []

    z_waypoints = 0.0  # small z-offset to avoid z-fighting if maps overlap

    for i, (map_name, data) in enumerate(zip(map_name_list, map_data)):
        # Build square drivable map bitmask (and associated extents)
        # (hashmap object not returned to keep consistency with current code)
        _hashmap, _size_meters, map_size_pixels, _x_lim, _y_lim, res = create_square_drivable_map_v2(
            min_common_square_size, data["outer"], data["inner"], resolution
        )

        # Write boundaries and trajectories as USD points (visible or hidden)
        if CONFIG["env_config"]["BOUNDARY_VISIBLE"]:
            outer_usd = set_points_usd(
                data["outer"], map_name, "outer", origin_list[i], stage,
                color=[(1.0, 1.0, 0.0)], radius=0.2, height=z_waypoints
            )
            inner_usd = set_points_usd(
                data["inner"], map_name, "inner", origin_list[i], stage,
                color=[(1.0, 1.0, 0.0)], radius=0.2, height=z_waypoints
            )
        else:
            # write invisible placeholders (radius=0) to keep consistent paths
            outer_usd = set_points_usd(
                data["outer"], map_name, "outer", origin_list[i], stage,
                color=[(0.0, 1.0, 0.0)], radius=0.0, height=z_waypoints
            )
            inner_usd = set_points_usd(
                data["inner"], map_name, "inner", origin_list[i], stage,
                color=[(0.25, 1.0, 0.0)], radius=0.0, height=z_waypoints
            )

        if CONFIG["env_config"]["TRAJ_VISIBLE"]:
            waypoints_usd = set_points_usd(
                data["waypoints"], map_name, "waypoints", origin_list[i], stage,
                color=[(1.0, 0.0, 0.0)], radius=0.05, height=z_waypoints
            )
            set_points_usd(
                data["opp_traj_iqp"][:, 0:2], map_name, "opp_traj_iqp", origin_list[i], stage,
                color=[(0.25, 1.0, 0.0)], radius=0.05, height=z_waypoints
            )
            set_points_usd(
                data["opp_traj_sp"][:, 0:2], map_name, "opp_traj_sp", origin_list[i], stage,
                color=[(0.0, 1.0, 1.0)], radius=0.05, height=z_waypoints
            )
        else:
            waypoints_usd = set_points_usd(
                data["waypoints"], map_name, "waypoints", origin_list[i], stage,
                color=[(1.0, 0.0, 0.0)], radius=0.0, height=z_waypoints
            )
            set_points_usd(
                data["opp_traj_iqp"][:, 0:2], map_name, "opp_traj_iqp", origin_list[i], stage,
                color=[(0.25, 1.0, 0.0)], radius=0.0, height=z_waypoints
            )
            set_points_usd(
                data["opp_traj_sp"][:, 0:2], map_name, "opp_traj_sp", origin_list[i], stage,
                color=[(0.0, 1.0, 1.0)], radius=0.0, height=z_waypoints
            )

        z_waypoints += 0.3  # bump the z-plane for the next map’s points

        # Cache per-map arrays (convert to [x, y, 0] lists expected elsewhere)
        waypoints_list.append([[p[0], p[1], 0.0] for p in waypoints_usd])
        outer_list.append([[p[0], p[1], 0.0] for p in outer_usd])
        inner_list.append([[p[0], p[1], 0.0] for p in inner_usd])

        d_lat_list.append(data["d_lat"].tolist())
        psi_rad_list.append(data["psi_rad"].tolist())
        kappa_radpm_list.append(data["kappa_radpm"].tolist())
        vx_mps_list.append(data["vx_mps"].tolist())

        opp_traj_center_list.append(data["opp_traj_center"].tolist())
        opp_traj_iqp_list.append(data["opp_traj_iqp"].tolist())
        opp_traj_sp_list.append(data["opp_traj_sp"].tolist())

        spacing_meters_list.append([res, res])
        map_size_pixels_list.append(map_size_pixels)

    # Persist the USD stage on disk
    stage.GetRootLayer().Save()

    return (
        waypoints_list,
        outer_list,
        inner_list,
        d_lat_list,
        psi_rad_list,
        kappa_radpm_list,
        vx_mps_list,
        opp_traj_center_list,
        opp_traj_iqp_list,
        opp_traj_sp_list,
        spacing_meters_list,
        map_size_pixels_list,
    )


# ---------------------------------------------------------------------------
# Pose generators (map-index aware, vectorized where possible)
# ---------------------------------------------------------------------------

def generate_random_poses_from_list(
    env_ids,
    num_poses: int,
    map_levels,
    env_origins,
    map_origin_list,
    row_spacing_list,
    col_spacing_list,
    traversability_hashmap_list,
    waypoints_usd_list,
    outer_usd_list,
    inner_usd_list,
    margin: float = 0.1,
):
    """
    Sample random robot poses from drivable bitmaps for a subset of envs.

    For each env in `env_ids`, we:
      - choose a random drivable pixel (1) in that env's `map_level` bitmap
      - project to world XY using the (row/col) spacing and per-env origin
      - set yaw so the robot roughly faces along the inner boundary direction

    Returns
    -------
    poses : list[tuple(float, float, float)]
        (x, y, yaw_deg) for each env in `env_ids` (kept in order).
    current_wps_idx : np.ndarray
        Index of the nearest inner boundary point per env (for book-keeping).
    """
    env_ids_np = env_ids.cpu().numpy() if torch.is_tensor(env_ids) else np.array(env_ids)
    map_levels_np = map_levels.cpu().numpy() if torch.is_tensor(map_levels) else np.array(map_levels)

    requested_map_levels = map_levels_np[env_ids_np]

    all_xs_shifted = np.zeros(len(env_ids_np))
    all_ys_shifted = np.zeros(len(env_ids_np))
    current_wps_idx = np.zeros(len(env_ids_np))
    all_angles = np.zeros(len(env_ids_np))

    for map_level in np.unique(requested_map_levels):
        env_mask = requested_map_levels == map_level
        current_env_ids = env_ids_np[env_mask]
        if current_env_ids.size == 0:
            continue

        trav = np.array(traversability_hashmap_list[map_level])
        H, W = trav.shape
        valid_y, valid_x = trav.nonzero()

        # random choice (with replacement if necessary)
        idxs = np.random.choice(
            len(valid_x), size=len(current_env_ids), replace=len(current_env_ids) > len(valid_x)
        )

        xs = (valid_x[idxs] - W // 2) * row_spacing_list[map_level]
        ys = (valid_y[idxs] - H // 2) * col_spacing_list[map_level]

        inner_xy = torch.tensor(inner_usd_list[map_level])[:, :2].to(torch.float32)

        # Heading: tangent of inner boundary a few points ahead
        positions = torch.stack([torch.tensor(xs), torch.tensor(ys)], dim=1) + torch.tensor(
            map_origin_list[map_level][:2]
        )
        indices, _ = find_frenet_coord_along_waypoints(inner_xy, positions)
        current_wps_idx[env_mask] = indices.numpy()

        lookahead = 5
        next_indices = (indices + lookahead) % len(inner_xy)
        current_wps = inner_xy[indices]
        next_wps = inner_xy[next_indices]
        deltas = next_wps - current_wps
        angles_deg = torch.rad2deg(torch.atan2(deltas[:, 1], deltas[:, 0])) + np.random.uniform(
            -15, 15, size=len(current_env_ids)
        )

        xs_shifted = xs + env_origins[current_env_ids, 0].cpu().numpy() + np.array(map_origin_list)[map_level, 0]
        ys_shifted = ys + env_origins[current_env_ids, 1].cpu().numpy() + np.array(map_origin_list)[map_level, 1]

        all_xs_shifted[env_mask] = xs_shifted
        all_ys_shifted[env_mask] = ys_shifted
        all_angles[env_mask] = angles_deg.numpy()

    poses = list(zip(all_xs_shifted.tolist(), all_ys_shifted.tolist(), all_angles.tolist()))
    return poses, current_wps_idx


def generate_start_idx_poses_from_list(
    env_ids,
    num_poses: int,
    map_levels,
    env_origins,
    row_spacing_list,
    col_spacing_list,
    traversability_hashmap_list,
    waypoints_usd_list,
    outer_usd_list,
    inner_usd_list,
    margin: float = 0.1,
):
    """
    Deterministic start poses at the first inner-boundary waypoint for each env.

    Useful for reproducible evaluation or curriculum starts.

    Returns
    -------
    poses : list[(x, y, yaw_deg)]
    current_wps_idx : np.ndarray of indices (all zeros here)
    """
    env_ids_np = env_ids.cpu().numpy() if torch.is_tensor(env_ids) else np.array(env_ids)
    map_levels_np = map_levels.cpu().numpy() if torch.is_tensor(map_levels) else np.array(map_levels)

    requested_map_levels = map_levels_np[env_ids_np]

    all_xs_shifted = np.zeros(len(env_ids_np))
    all_ys_shifted = np.zeros(len(env_ids_np))
    current_wps_idx = np.zeros(len(env_ids_np))
    all_angles = np.zeros(len(env_ids_np))

    for map_level in np.unique(requested_map_levels):
        env_mask = requested_map_levels == map_level
        current_env_ids = env_ids_np[env_mask]
        if current_env_ids.size == 0:
            continue

        inner_xy = torch.tensor(inner_usd_list[map_level])[:, :2].to(torch.float32)
        # place at inner[0]
        x0 = float(inner_xy[0, 0].cpu().numpy())
        y0 = float(inner_xy[0, 1].cpu().numpy())

        # heading toward a lookahead waypoint
        lookahead = 5
        next_idx = lookahead % len(inner_xy)
        delta = inner_xy[next_idx] - inner_xy[0]
        angle_deg = float(torch.rad2deg(torch.atan2(delta[1], delta[0])))

        xs_shifted = x0 + env_origins[current_env_ids, 0].cpu().numpy()
        ys_shifted = y0 + env_origins[current_env_ids, 1].cpu().numpy()

        all_xs_shifted[env_mask] = xs_shifted
        all_ys_shifted[env_mask] = ys_shifted
        all_angles[env_mask] = angle_deg  # same for all envs on this map_level
        current_wps_idx[env_mask] = 0

    poses = list(zip(all_xs_shifted.tolist(), all_ys_shifted.tolist(), all_angles.tolist()))
    return poses, current_wps_idx


def generate_random_poses(
    env_origins,
    env_ids,
    num_poses: int,
    row_spacing: float,
    col_spacing: float,
    traversability_hashmap,
    waypoints_usd,
    outer_usd,
    inner_usd,
    margin: float = 0.1,
):
    """
    Sample random robot poses from a single traversability bitmap (no map levels).

    Returns
    -------
    poses : list[(x, y, yaw_deg)] length = num_poses
    """
    trav = np.array(traversability_hashmap)
    H, W = trav.shape
    valid_y, valid_x = trav.nonzero()

    idxs = np.random.choice(len(valid_x), size=num_poses, replace=False)

    xs = (valid_x[idxs] - W // 2) * row_spacing
    ys = (valid_y[idxs] - H // 2) * col_spacing

    inner_xy = torch.tensor(inner_usd)[:, :2].to(torch.float32)

    # heading along inner boundary (avoid centerline wrap from other side)
    positions = torch.stack([torch.tensor(xs), torch.tensor(ys)], dim=1)
    indices, _ = find_frenet_coord_along_waypoints(inner_xy, positions)

    lookahead = 5
    next_indices = (indices + lookahead) % len(inner_xy)
    current_wps = inner_xy[indices]
    next_wps = inner_xy[next_indices]
    deltas = next_wps - current_wps
    angles_deg = torch.rad2deg(torch.atan2(deltas[:, 1], deltas[:, 0])) + np.random.uniform(-15, 15)

    xs_shifted = (valid_x[idxs] - W // 2) * row_spacing + env_origins[env_ids, 0].cpu().numpy()
    ys_shifted = (valid_y[idxs] - H // 2) * col_spacing + env_origins[env_ids, 1].cpu().numpy()

    poses = list(zip(xs_shifted.tolist(), ys_shifted.tolist(), angles_deg.tolist()))
    return poses


def generate_random_poses_from_waypoints(
    env_ids,
    num_poses: int,
    map_levels,
    env_origins,
    waypoints_usd_list,
    inner_usd_list,
    max_radius_offset: float = 1.0,
):
    """
    Sample random poses by picking random centerline waypoints per env.

    Heading is set to point toward a fixed lookahead waypoint.

    Returns
    -------
    poses : list[(x, y, yaw_deg)]
    current_wps_idx : np.ndarray of chosen waypoint indices
    """
    env_ids_np = env_ids.cpu().numpy() if torch.is_tensor(env_ids) else np.array(env_ids)
    map_levels_np = map_levels.cpu().numpy() if torch.is_tensor(map_levels) else np.array(map_levels)

    requested_map_levels = map_levels_np[env_ids_np]

    all_xs_shifted = np.zeros(len(env_ids_np))
    all_ys_shifted = np.zeros(len(env_ids_np))
    current_wps_idx = np.zeros(len(env_ids_np))
    all_angles = np.zeros(len(env_ids_np))

    for map_level in np.unique(requested_map_levels):
        env_mask = requested_map_levels == map_level
        current_env_ids = env_ids_np[env_mask]
        if current_env_ids.size == 0:
            continue

        waypoints_xy = torch.tensor(waypoints_usd_list[map_level])[:, :2].to(torch.float32)
        num_wps = len(waypoints_xy)

        selected_indices = np.random.choice(num_wps, size=len(current_env_ids), replace=True)
        selected = waypoints_xy[selected_indices]
        xs = selected[:, 0].cpu().numpy()
        ys = selected[:, 1].cpu().numpy()

        lookahead = 5
        next_indices = (selected_indices + lookahead) % num_wps
        next_pts = waypoints_xy[next_indices]
        deltas = next_pts - selected
        angles_deg = torch.rad2deg(torch.atan2(deltas[:, 1], deltas[:, 0])) + np.random.uniform(
            -15, 15, size=len(current_env_ids)
        )

        xs_shifted = xs + env_origins[current_env_ids, 0].cpu().numpy()
        ys_shifted = ys + env_origins[current_env_ids, 1].cpu().numpy()

        all_xs_shifted[env_mask] = xs_shifted
        all_ys_shifted[env_mask] = ys_shifted
        all_angles[env_mask] = angles_deg.numpy()
        current_wps_idx[env_mask] = selected_indices

    poses = list(zip(all_xs_shifted.tolist(), all_ys_shifted.tolist(), all_angles.tolist()))
    return poses, current_wps_idx


def generate_random_poses_from_waypoints_with_opponent(
    env_ids,
    num_poses: int,
    map_levels,
    env_origins,
    waypoints_usd_list,
    inner_usd_list,
    vx_mps_list,
    max_radius_offset: float = 1.0,
):
    """
    Sample ego & opponent poses and initial velocities from centerline waypoints.

    For each env:
      - pick a random waypoint for ego; compute heading from a fixed lookahead
      - place opponent at a random offset ahead (in index space)
      - set ego/opponent velocities in world frame using reference speeds

    Returns
    -------
    ego_poses : list[(x, y, yaw_deg)]
    current_wps_idx : np.ndarray
    ego_velocities : list[(vx, vy)]
    opp_poses : list[(x, y, yaw_deg)]
    opp_current_wps_idx : np.ndarray
    opp_velocities : list[(vx, vy)]
    """
    env_ids_np = env_ids.cpu().numpy() if torch.is_tensor(env_ids) else np.array(env_ids)
    map_levels_np = map_levels.cpu().numpy() if torch.is_tensor(map_levels) else np.array(map_levels)

    requested_map_levels = map_levels_np[env_ids_np]

    all_xs_shifted = np.zeros(len(env_ids_np))
    all_ys_shifted = np.zeros(len(env_ids_np))
    current_wps_idx = np.zeros(len(env_ids_np))
    all_angles = np.zeros(len(env_ids_np))
    all_vx = np.zeros(len(env_ids_np))
    all_vy = np.zeros(len(env_ids_np))

    all_opp_xs_shifted = np.zeros(len(env_ids_np))
    all_opp_ys_shifted = np.zeros(len(env_ids_np))
    opp_current_wps_idx = np.zeros(len(env_ids_np))
    all_opp_angles = np.zeros(len(env_ids_np))
    all_opp_vx = np.zeros(len(env_ids_np))
    all_opp_vy = np.zeros(len(env_ids_np))

    for map_level in np.unique(requested_map_levels):
        env_mask = requested_map_levels == map_level
        current_env_ids = env_ids_np[env_mask]
        if current_env_ids.size == 0:
            continue

        waypoints_xy = torch.tensor(waypoints_usd_list[map_level])[:, :2].to(torch.float32)
        speed = torch.tensor(vx_mps_list[map_level]).to(torch.float32).flatten()
        num_wps = len(waypoints_xy)

        # Ego selection
        selected_indices = np.random.choice(num_wps, size=len(current_env_ids), replace=True)
        ego_pts = waypoints_xy[selected_indices]

        lookahead = 5
        next_indices = (selected_indices + lookahead) % num_wps
        next_pts = waypoints_xy[next_indices]
        deltas = next_pts - ego_pts

        # Heading radians (for velocity projection) + degrees (for pose)
        angles_rad = torch.atan2(deltas[:, 1], deltas[:, 0])
        angles_deg = torch.rad2deg(angles_rad) + np.random.uniform(-15, 15, size=len(current_env_ids))

        ego_speeds = speed[selected_indices].cpu().numpy()
        vx = ego_speeds * np.cos(angles_rad.cpu().numpy())
        vy = ego_speeds * np.sin(angles_rad.cpu().numpy())

        xs_shifted = ego_pts[:, 0].cpu().numpy() + env_origins[current_env_ids, 0].cpu().numpy()
        ys_shifted = ego_pts[:, 1].cpu().numpy() + env_origins[current_env_ids, 1].cpu().numpy()

        all_xs_shifted[env_mask] = xs_shifted
        all_ys_shifted[env_mask] = ys_shifted
        all_angles[env_mask] = angles_deg if isinstance(angles_deg, np.ndarray) else angles_deg.numpy()
        current_wps_idx[env_mask] = selected_indices
        all_vx[env_mask] = vx
        all_vy[env_mask] = vy

        # Opponent placement: ahead by random index offset within configured bounds
        offset = np.random.randint(
            low=CONFIG["env_config"]["OPPONENT_INIT_DISTANCE_IDX_MIN"],
            high=CONFIG["env_config"]["OPPONENT_INIT_DISTANCE_IDX_MAX"],
            size=len(selected_indices),
        )
        opp_indices = (selected_indices + offset) % num_wps
        opp_pts = waypoints_xy[opp_indices]

        opp_next_indices = (opp_indices + lookahead) % num_wps
        opp_next_pts = waypoints_xy[opp_next_indices]
        opp_deltas = opp_next_pts - opp_pts
        opp_angles_rad = torch.atan2(opp_deltas[:, 1], opp_deltas[:, 0])
        opp_angles_deg = torch.rad2deg(opp_angles_rad)

        opp_speeds = speed[opp_indices].cpu().numpy()
        opp_vx = opp_speeds * np.cos(opp_angles_rad.cpu().numpy())
        opp_vy = opp_speeds * np.sin(opp_angles_rad.cpu().numpy())

        opp_xs_shifted = opp_pts[:, 0].cpu().numpy() + env_origins[current_env_ids, 0].cpu().numpy()
        opp_ys_shifted = opp_pts[:, 1].cpu().numpy() + env_origins[current_env_ids, 1].cpu().numpy()

        all_opp_xs_shifted[env_mask] = opp_xs_shifted
        all_opp_ys_shifted[env_mask] = opp_ys_shifted
        all_opp_angles[env_mask] = (
            opp_angles_deg.cpu().numpy() if torch.is_tensor(opp_angles_deg) else opp_angles_deg
        )
        opp_current_wps_idx[env_mask] = opp_indices
        all_opp_vx[env_mask] = opp_vx
        all_opp_vy[env_mask] = opp_vy

    ego_poses = list(zip(all_xs_shifted.tolist(), all_ys_shifted.tolist(), all_angles.tolist()))
    ego_velocities = list(zip(all_vx.tolist(), all_vy.tolist()))
    opp_poses = list(zip(all_opp_xs_shifted.tolist(), all_opp_ys_shifted.tolist(), all_opp_angles.tolist()))
    opp_velocities = list(zip(all_opp_vx.tolist(), all_opp_vy.tolist()))

    return ego_poses, current_wps_idx, ego_velocities, opp_poses, opp_current_wps_idx, opp_velocities


# ---------------------------------------------------------------------------
# Frenet / nearest waypoint helper
# ---------------------------------------------------------------------------

def find_frenet_coord_along_waypoints(
    waypoints: torch.Tensor,     # [M, 2]
    positions: torch.Tensor,     # [N, 2]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Nearest waypoint index and signed lateral distance for many positions.

    The sign convention is: left of the path tangent is positive, right is negative.
    Tangent is estimated from the next waypoint (with wrap-around).

    Args
    ----
    waypoints : torch.Tensor [M, 2]
        2D coordinates of waypoints along the path (closed or open).
    positions : torch.Tensor [N, 2]
        2D world coordinates of positions to project.

    Returns
    -------
    closest_idx : torch.Tensor [N]
        Index of the nearest waypoint (per position).
    signed_dists : torch.Tensor [N]
        Signed perpendicular distance to the path at `closest_idx`.
    """
    M = waypoints.shape[0]
    N = positions.shape[0]

    # Vector from each waypoint to each query position: [N, M, 2]
    diffs = positions.unsqueeze(1) - waypoints.unsqueeze(0)

    # Euclidean distance to each waypoint: [N, M]
    dists = torch.norm(diffs, p=2, dim=2)

    # Closest waypoint index: [N]
    closest_idx = torch.argmin(dists, dim=1)

    # Tangent = vector to next waypoint (wrap-around), normalized: [N, 2]
    next_idx = (closest_idx + 1) % M
    tangents = waypoints[next_idx] - waypoints[closest_idx]
    tangents = tangents / (torch.norm(tangents, p=2, dim=1, keepdim=True) + 1e-9)

    # Vector from closest waypoint to position: [N, 2]
    closest_diffs = diffs[torch.arange(N), closest_idx]

    # z-component of 2D cross product (tangent × diff)
    cross_z = closest_diffs[:, 0] * tangents[:, 1] - closest_diffs[:, 1] * tangents[:, 0]

    # Signed distance = unsigned distance with sign from cross product
    signed_dists = dists[torch.arange(N), closest_idx] * torch.sign(cross_z)

    return closest_idx, signed_dists
