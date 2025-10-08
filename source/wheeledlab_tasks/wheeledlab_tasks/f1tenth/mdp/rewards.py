"""
Reward terms for F1Tenth overtaking task.

This file provides small, composable reward/penalty functions that:
- read state from the IsaacLab `ManagerBasedEnv`,
- optionally maintain short histories on `env` (when needed by the term),
- compute geometry/kinematics features and convert them into rewards.

Design
------
- Every function is idempotent: safe to call at every step.
- Per-env buffers are allocated lazily on first use, then reused.
- For multi-map vectorized envs, we group envs by map level so we never mix waypoint arrays.
- Shapes are stable; keep them if you consume these rewards elsewhere.

Conventions
-----------
- "s" = longitudinal progress (index along waypoints). "d" = lateral offset (meters).
- World frame positions are used for geometry to the track boundaries.
- Rewards return a tensor of shape [num_envs], unless otherwise noted.
"""

import torch
import numpy as np
import isaaclab.utils.math as math_utils

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.terrains import TerrainImporter
from ..utils import find_frenet_coord_along_waypoints
import isaaclab.envs.mdp as mdp

from wheeledlab_tasks.config_loader import load_config
CONFIG = load_config()


# ---------------------------------------------------------------------------
# Collision / safety penalties
# ---------------------------------------------------------------------------

def wall_collision_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Hard penalty when the car intersects the inner or outer wall.

    How it works
    ------------
    For each env:
      1) Find the nearest points on the inner and outer boundaries.
      2) If the absolute lateral distance to either boundary is below the
         HARD_WALL_COLLISION_RADIUS, mark a collision.
    Returns -1 for collision, 0 otherwise.
    """
    pos_xy_world = mdp.root_pos_w(env)[..., :2]

    # Ensure required per-map tensors exist
    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_outer_list"):
        env._outer_list = [torch.tensor(outer, device=env.device, dtype=torch.float32)
                           for outer in env.scene.terrain.cfg.outer_list]
    if not hasattr(env, "_inner_list"):
        env._inner_list = [torch.tensor(inner, device=env.device, dtype=torch.float32)
                           for inner in env.scene.terrain.cfg.inner_list]

    map_levels = env._map_levels
    unique_map_levels = torch.unique(map_levels)

    collision_bool = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    for map_level in unique_map_levels:
        env_mask = (map_levels == map_level)
        if not env_mask.any():
            continue

        map_positions = pos_xy_world[env_mask]
        inner_xy_world = env._inner_list[map_level][:, :2]
        outer_xy_world = env._outer_list[map_level][:, :2]

        _, dist_from_inner = find_frenet_coord_along_waypoints(inner_xy_world, map_positions)
        _, dist_from_outer = find_frenet_coord_along_waypoints(outer_xy_world, map_positions)

        collision_with_inner = torch.abs(dist_from_inner) < CONFIG["env_config"]["HARD_WALL_COLLISION_RADIUS"]
        collision_with_outer = torch.abs(dist_from_outer) < CONFIG["env_config"]["HARD_WALL_COLLISION_RADIUS"]
        collision_bool[env_mask] = (collision_with_inner | collision_with_outer).long()

    return torch.where(collision_bool.bool(), -1.0, 0.0)


def soft_wall_collision_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Softer collision penalty using a larger (SOFT) radius.

    Same logic as `wall_collision_penalty`, but with `SOFT_WALL_COLLISION_RADIUS`.
    Useful as an earlier warning signal to discourage scraping along track bounds.
    Returns -1 for "soft collision", 0 otherwise.
    """
    pos_xy_world = mdp.root_pos_w(env)[..., :2]

    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_outer_list"):
        env._outer_list = [torch.tensor(outer, device=env.device, dtype=torch.float32)
                           for outer in env.scene.terrain.cfg.outer_list]
    if not hasattr(env, "_inner_list"):
        env._inner_list = [torch.tensor(inner, device=env.device, dtype=torch.float32)
                           for inner in env.scene.terrain.cfg.inner_list]

    map_levels = env._map_levels
    unique_map_levels = torch.unique(map_levels)

    collision_bool = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    for map_level in unique_map_levels:
        env_mask = (map_levels == map_level)
        if not env_mask.any():
            continue

        map_positions = pos_xy_world[env_mask]
        inner_xy_world = env._inner_list[map_level][:, :2]
        outer_xy_world = env._outer_list[map_level][:, :2]

        _, dist_from_inner = find_frenet_coord_along_waypoints(inner_xy_world, map_positions)
        _, dist_from_outer = find_frenet_coord_along_waypoints(outer_xy_world, map_positions)

        collision_with_inner = torch.abs(dist_from_inner) < CONFIG["env_config"]["SOFT_WALL_COLLISION_RADIUS"]
        collision_with_outer = torch.abs(dist_from_outer) < CONFIG["env_config"]["SOFT_WALL_COLLISION_RADIUS"]
        collision_bool[env_mask] = (collision_with_inner | collision_with_outer).long()

    return torch.where(collision_bool.bool(), -1.0, 0.0)


def side_slip_penalty(
    env: ManagerBasedEnv,
    min_vel_x: float = 2.5,
    slip_thresh: float = 0.12,
    max_slip_angle: float = 0.5,
) -> torch.Tensor:
    """Penalty proportional to side-slip angle when moving fast enough.

    Why
    ---
    Large slip angles (atan2(vy, vx)) imply instability / poor control.
    We penalize once |slip| exceeds `slip_thresh`, saturating at `max_slip_angle`.

    Returns:
        Negative value in [- (max_slip_angle - slip_thresh), 0], per env.
    """
    vel = mdp.base_lin_vel(env)  # [num_envs, 2]
    slip_angle = torch.abs(torch.atan2(vel[..., 1], vel[..., 0]))
    moving_mask = torch.abs(vel[..., 0]) >= min_vel_x

    penalty = torch.clamp(slip_angle - slip_thresh, min=0.0)
    penalty = torch.clamp(penalty, max=max_slip_angle - slip_thresh)
    penalty = penalty * moving_mask.float()
    return -penalty


def opponent_collision_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Hard penalty when ego is too close to opponent in s and d.

    Logic
    -----
    If both:
      |Δs| < OPP_FRONT_COLLISION_RADIUS  AND
      |Δd| < OPP_LAT_COLLISION_RADIUS
    -> collision considered; returns -1, else 0.

    Notes
    -----
    Uses the most recent entries from histories:
      env._s_idx_diff_history[:, 0], env._d_diff_history[:, 0]
    These are populated by your observation code.
    """
    # Ensure histories exist (shape: [num_envs, history_len])
    if not hasattr(env, "_s_idx_diff_history"):
        env._s_idx_diff_history = torch.ones((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
    if not hasattr(env, "_d_diff_history"):
        env._d_diff_history = torch.ones((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)

    opp_collision = (
        (torch.abs(env._s_idx_diff_history[:, 0]) < CONFIG["env_config"]["OPP_FRONT_COLLISION_RADIUS"])
        & (torch.abs(env._d_diff_history[:, 0]) < CONFIG["env_config"]["OPP_LAT_COLLISION_RADIUS"])
    )
    return torch.where(opp_collision, -1.0, 0.0)


# ---------------------------------------------------------------------------
# Overtake-related rewards
# ---------------------------------------------------------------------------

def opponent_mean_delta_speed(env: ManagerBasedEnv) -> torch.Tensor:
    """Encourage faster-than-opponent behavior (on average).

    Returns the negative mean of (v_opp_x - v_ego_x) over a short history.
    If ego is faster (negative diff), the mean is negative, and -mean is positive (reward).
    """
    if not hasattr(env, "_vx_diff_history"):
        env._vx_diff_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
    mean_vx_diff = torch.mean(env._vx_diff_history, dim=1)
    return -mean_vx_diff


def opponent_overtake_delta_distance_reward(env: ManagerBasedEnv) -> torch.Tensor:
    """Reward both closing the gap and extending the lead w.r.t. opponent.
    See Song GT7 paper
    Idea
    ----
    Δs_opp_ego(t) = s_opp - s_ego (signed, shortest on the track)
    We compare current Δs with the previous Δs (stored in env._prev_delta_s_opp_ego).
      - If ego is behind (Δs>0), reward reduction of Δs (closing the gap).
      - If ego is ahead  (Δs<0), reward *increase* in |Δs| (extending the lead).

    Safety gates
    ------------
    Reward is zeroed if any off-track (wall) or opponent-collision has been detected
    in the recent history buffer.
    """
    if not hasattr(env, "_prev_delta_s_opp_ego"):
        env._prev_delta_s_opp_ego = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device) \
            + CONFIG["env_config"]["OPPONENT_INIT_DISTANCE_IDX_MIN"] * CONFIG["env_config"]["LEN_S_IDX"]

    ego_position_xy = mdp.root_pos_w(env, SceneEntityCfg("robot"))[..., :2]
    opp_position_xy = mdp.root_pos_w(env, SceneEntityCfg("opponent"))[..., :2]

    # Ensure per-map waypoint lists exist
    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    map_levels = env._map_levels
    unique_map_levels = torch.unique(map_levels)

    delta_s_opp_ego = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    delta_delta_s_opp_ego = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    for map_level in unique_map_levels:
        env_mask = (map_levels == map_level)
        if not env_mask.any():
            continue

        waypoints_world = env._waypoints_list[map_level][:, :2]
        num_waypoints = len(waypoints_world)

        ego_idx, _ = find_frenet_coord_along_waypoints(waypoints_world, ego_position_xy[env_mask])
        opp_idx, _ = find_frenet_coord_along_waypoints(waypoints_world, opp_position_xy[env_mask])

        # Signed shortest difference on circular track
        delta_s_opp_ego[env_mask] = ((opp_idx - ego_idx + num_waypoints // 2) % num_waypoints - num_waypoints // 2).float()
        # Reward uses meters (multiply by LEN_S_IDX)
        delta_delta_s_opp_ego[env_mask] = env._prev_delta_s_opp_ego[env_mask] - delta_s_opp_ego[env_mask] * CONFIG["env_config"]["LEN_S_IDX"]
        env._prev_delta_s_opp_ego[env_mask] = delta_s_opp_ego[env_mask] * CONFIG["env_config"]["LEN_S_IDX"]

    # Safety gates: zero reward if any wall/opp collision in recent buffer
    if not hasattr(env, "_wall_collision_history"):
        env._wall_collision_history = torch.zeros((env.num_envs, CONFIG["env_config"]["REW_HISTORY_LENGTH"]), dtype=torch.long, device=env.device)
    if not hasattr(env, "_opponent_collision_history"):
        env._opponent_collision_history = torch.zeros((env.num_envs, CONFIG["env_config"]["REW_HISTORY_LENGTH"]), dtype=torch.long, device=env.device)

    no_off_track = env._wall_collision_history.max(dim=1).values == 0
    no_opp_collision = env._opponent_collision_history.max(dim=1).values == 0

    # If ego ahead (Δs<0): reward is -ΔΔs (increase negative gap).
    # If ego behind (Δs>=0): reward is  ΔΔs (close positive gap).
    reward = torch.where(delta_s_opp_ego < 0, -delta_delta_s_opp_ego, delta_delta_s_opp_ego)
    return torch.where(no_off_track & no_opp_collision, reward * CONFIG["env_config"]["LEN_S_IDX"], 0.0)


def opponent_overtake_distance_reward(env: ManagerBasedEnv) -> torch.Tensor:
    """Dense shaping to incentivize passing the opponent.

    Computes Δs = (s_opp - s_ego) after shifting the opponent index by
    `OPPONENT_OVERTAKEN_IDX` (i.e., how far until ego surpasses that threshold).
    Returns negative Δs (in meters): more negative means closer to/after overtake.
    """
    ego_position_xy = mdp.root_pos_w(env, SceneEntityCfg("robot"))[..., :2]
    opp_position_xy = mdp.root_pos_w(env, SceneEntityCfg("opponent"))[..., :2]

    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_waypoints_list"):
        env._waypoints_list = [torch.tensor(wps, device=env.device, dtype=torch.float32)
                               for wps in env.scene.terrain.cfg.waypoints_list]
    if not hasattr(env, "_inner_list"):
        env._inner_list = [torch.tensor(inner, device=env.device, dtype=torch.float32)
                           for inner in env.scene.terrain.cfg.inner_list]

    map_levels = env._map_levels
    unique_map_levels = torch.unique(map_levels)

    delta_s_opp_ego = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    for map_level in unique_map_levels:
        env_mask = (map_levels == map_level)
        if not env_mask.any():
            continue

        waypoints_world = env._waypoints_list[map_level][:, :2]
        num_waypoints = len(waypoints_world)

        ego_idx, _ = find_frenet_coord_along_waypoints(waypoints_world, ego_position_xy[env_mask])
        opp_idx, _ = find_frenet_coord_along_waypoints(waypoints_world, opp_position_xy[env_mask])

        opp_overtaken_idx = opp_idx + CONFIG["env_config"]["OPPONENT_OVERTAKEN_IDX"]
        delta_s_opp_ego[env_mask] = ((opp_overtaken_idx - ego_idx + num_waypoints // 2) % num_waypoints - num_waypoints // 2).float()

    # Safety gates
    if not hasattr(env, "_wall_collision_history"):
        env._wall_collision_history = torch.zeros((env.num_envs, CONFIG["env_config"]["REW_HISTORY_LENGTH"]), dtype=torch.long, device=env.device)
    if not hasattr(env, "_opponent_collision_history"):
        env._opponent_collision_history = torch.zeros((env.num_envs, CONFIG["env_config"]["REW_HISTORY_LENGTH"]), dtype=torch.long, device=env.device)
    no_off_track = env._wall_collision_history.max(dim=1).values == 0
    no_opp_collision = env._opponent_collision_history.max(dim=1).values == 0

    return torch.where(no_off_track & no_opp_collision, -delta_s_opp_ego * CONFIG["env_config"]["LEN_S_IDX"], 0.0)


def opponent_overtake_completed_reward(env: ManagerBasedEnv) -> torch.Tensor:
    """Sparse success reward when a full overtake is completed cleanly.

    Conditions (all must hold):
      - ego is ahead by more than OPPONENT_OVERTAKEN_IDX (Δs < -threshold),
      - no off-track (walls) and no opponent collision (recent history),
      - ego has not progressed "too far" since a progress checkpoint
        (prevents giving reward repeatedly or in degenerate loops).

    Returns:
        reward = (overtake_completed.float() * env._opponent_vel_scaling)
        i.e., scales with current opponent speed scaling to keep signal meaningful.
    """
    ego_position_xy = mdp.root_pos_w(env, SceneEntityCfg("robot"))[..., :2]
    opp_position_xy = mdp.root_pos_w(env, SceneEntityCfg("opponent"))[..., :2]

    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_progress_history_indices"):
        env._progress_history_length = CONFIG["env_config"]["REW_HISTORY_LENGTH"]
        env._progress_history_indices = torch.zeros((env.num_envs, env._progress_history_length), dtype=torch.long, device=env.device)
        env._reset_env_bool = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    map_levels = env._map_levels
    unique_map_levels = torch.unique(map_levels)

    delta_s_opp_ego = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    ego_progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    for map_level in unique_map_levels:
        env_mask = (map_levels == map_level)
        if not env_mask.any():
            continue

        waypoints_world = env._waypoints_list[map_level][:, :2]
        num_waypoints = len(waypoints_world)

        ego_idx, _ = find_frenet_coord_along_waypoints(waypoints_world, ego_position_xy[env_mask])
        opp_idx, _ = find_frenet_coord_along_waypoints(waypoints_world, opp_position_xy[env_mask])

        delta_s_opp_ego[env_mask] = ((opp_idx - ego_idx + num_waypoints // 2) % num_waypoints - num_waypoints // 2).float()
        ego_progress[env_mask] = (ego_idx - env._progress_history_indices[env_mask, env._progress_history_checkpoint_idx]) % num_waypoints

    if not hasattr(env, "_wall_collision_history"):
        env._wall_collision_history = torch.zeros((env.num_envs, CONFIG["env_config"]["REW_HISTORY_LENGTH"]), dtype=torch.long, device=env.device)
    if not hasattr(env, "_opponent_collision_history"):
        env._opponent_collision_history = torch.zeros((env.num_envs, CONFIG["env_config"]["REW_HISTORY_LENGTH"]), dtype=torch.long, device=env.device)

    no_off_track = env._wall_collision_history.max(dim=1).values == 0
    no_opp_collision = env._opponent_collision_history.max(dim=1).values == 0

    env._opponent_overtaken_bool = torch.where(
        (delta_s_opp_ego < -CONFIG["env_config"]["OPPONENT_OVERTAKEN_IDX"])
        & no_off_track
        & no_opp_collision
        & (ego_progress <= CONFIG["env_config"]["MAX_PROGRESS_IDX"]),
        torch.ones(env.num_envs, dtype=torch.bool, device=env.device),
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )

    if not hasattr(env, "_opponent_overtaken_history"):
        env._opponent_overtaken_history = torch.zeros((env.num_envs, CONFIG["env_config"]["REW_HISTORY_LENGTH"]), dtype=torch.long, device=env.device)

    env._opponent_overtaken_history[:, 1:] = env._opponent_overtaken_history[:, :-1].clone()
    env._opponent_overtaken_history[:, 0] = env._opponent_overtaken_bool

    overtake_completed = env._opponent_overtaken_history.min(dim=1).values == 1

    # ---- Logging (side effects kept intentionally) ----
    env.extras["log"]["Info/opponent_vel_scaling"] = env._opponent_vel_scaling_lvl
    if bool(overtake_completed.any()):
        env.extras["log"]["Info/opponent_vel_scaling_max_overtaken"] = torch.max(env._opponent_vel_scaling[overtake_completed])
    else:
        env.extras["log"]["Info/opponent_vel_scaling_max_overtaken"] = 0.0
    env.extras["log"]["Info/opponent_overtaken_step"] = torch.sum(overtake_completed.float())
    env.extras["log"]["Info/opponent_collision_counter"] = env._opponent_collision_counter
    env.extras["log"]["Info/opponent_collision_step"] = torch.sum(
        env._opponent_collision_history[:, CONFIG["env_config"]["OPPONENT_COLLISION_CHECK_IDX"]]
    )
    env.extras["log"]["Info/opponent_overtaken_collision_ratio"] = (
        env._opponent_overtaken_counter
        / (env._opponent_collision_counter + env._opponent_overtaken_counter + env._wall_collision_counter + 1)
    )
    env.extras["log"]["Info/opponent_overtaken_collision_ratio_step"] = (
        torch.sum(overtake_completed.float())
        / (
            torch.sum(env._opponent_collision_history[:, CONFIG["env_config"]["OPPONENT_COLLISION_CHECK_IDX"]])
            + torch.sum(overtake_completed.float())
            + torch.sum(env._wall_collision_history[:, CONFIG["env_config"]["WALL_COLLISION_CHECK_IDX"]].float())
            + 1
        )
    )
    env.extras["delta_s_opp_ego"] = delta_s_opp_ego[0]
    env.extras["opp_pos_xy"] = opp_position_xy
    env.extras["opp_speed"] = env._opponent_speed[:]
    # ---------------------------------------------------

    return overtake_completed.float() * env._opponent_vel_scaling


def opponent_overtake_positioning_reward(env: ManagerBasedEnv) -> torch.Tensor:
    """Reward spatial positioning that supports a clean overtake.
    DEPRECATED: the agent abuses of this reward by just trailing laterally and not attempting to overtake.

    Intuition
    ---------
    - When ego is behind, it should establish lateral offset (prepare to pass).
    - Reward grows with lateral separation to opponent, but decays with longitudinal distance
      (i.e., it matters most when we are near the opponent along s).

    Implementation (kept consistent with original)
    ----------------------------------------------
    positioning_reward = |Δd| * exp(- (Δs^2) * LEN_S_IDX * 2)
    gated by: no off-track, no opponent collision, limited ego progress since checkpoint.
    """
    num_episodes = env.common_step_counter // env.max_episode_length
    if num_episodes < CONFIG["env_config"]["IGNORE_OPPONENT_UNTIL_EP"]:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ego_position_xy = mdp.root_pos_w(env, SceneEntityCfg("robot"))[..., :2]
    opp_position_xy = mdp.root_pos_w(env, SceneEntityCfg("opponent"))[..., :2]

    ego_heading_w = env.scene["robot"].data.heading_w
    ego_heading_vec = torch.stack([torch.cos(ego_heading_w), torch.sin(ego_heading_w)], dim=1)

    # L/R relative positioning (kept for clarity and potential logging)
    dist = torch.norm(ego_position_xy - opp_position_xy, dim=1)
    ego_opp_vec = (opp_position_xy - ego_position_xy)
    ego_opp_positioning = (ego_heading_vec[:, 0] * ego_opp_vec[:, 1] - ego_heading_vec[:, 1] * ego_opp_vec[:, 0]) / (dist + 1.0)
    ego_opp_positioning = torch.clamp(ego_opp_positioning, -0.5, 0.5)

    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_progress_history_indices"):
        env._progress_history_length = CONFIG["env_config"]["REW_HISTORY_LENGTH"]
        env._progress_history_indices = torch.zeros((env.num_envs, env._progress_history_length), dtype=torch.long, device=env.device)
        env._reset_env_bool = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_progress_history_checkpoint_idx"):
        env._progress_history_checkpoint_idx = CONFIG["env_config"]["PROGRESS_HISTORY_CHECK_IDX"]

    map_levels = env._map_levels
    unique_map_levels = torch.unique(map_levels)

    delta_s_opp_ego = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    delta_d_opp_ego = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    ego_progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    for map_level in unique_map_levels:
        env_mask = (map_levels == map_level)
        if not env_mask.any():
            continue

        waypoints_world = env._waypoints_list[map_level][:, :2]
        num_waypoints = len(waypoints_world)

        ego_idx, ego_d = find_frenet_coord_along_waypoints(waypoints_world, ego_position_xy[env_mask])
        opp_idx, opp_d = find_frenet_coord_along_waypoints(waypoints_world, opp_position_xy[env_mask])

        delta_s_opp_ego[env_mask] = ((opp_idx - ego_idx + num_waypoints // 2) % num_waypoints - num_waypoints // 2).float()
        delta_d_opp_ego[env_mask] = torch.abs(ego_d - opp_d).float()

        ego_progress[env_mask] = (ego_idx - env._progress_history_indices[env_mask, env._progress_history_checkpoint_idx]) % num_waypoints

    # Safety gates
    if not hasattr(env, "_wall_collision_history"):
        env._wall_collision_history = torch.zeros((env.num_envs, CONFIG["env_config"]["REW_HISTORY_LENGTH"]), dtype=torch.long, device=env.device)
    if not hasattr(env, "_opponent_collision_history"):
        env._opponent_collision_history = torch.zeros((env.num_envs, CONFIG["env_config"]["REW_HISTORY_LENGTH"]), dtype=torch.long, device=env.device)

    no_off_track = env._wall_collision_history.max(dim=1).values == 0
    no_opp_collision = env._opponent_collision_history.max(dim=1).values == 0

    positioning_reward = torch.abs(delta_d_opp_ego) * torch.exp(-((delta_s_opp_ego) ** 2) * CONFIG["env_config"]["LEN_S_IDX"] * 2)
    return torch.where(no_off_track & no_opp_collision & (ego_progress <= CONFIG["env_config"]["MAX_PROGRESS_IDX"]), positioning_reward, 0.0)


# ---------------------------------------------------------------------------
# Driving posture / progress rewards
# ---------------------------------------------------------------------------

def upright_penalty(env: ManagerBasedEnv, thresh_deg: float) -> torch.Tensor:
    """Penalty if the vehicle tilts beyond a threshold (in degrees).

    Uses the world-frame up vector z•z (from rotation matrix) and converts the
    tilt to degrees. Penalty grows linearly above `thresh_deg`.
    """
    rot_mat = math_utils.matrix_from_quat(mdp.root_quat_w(env))
    up_dot = rot_mat[:, 2, 2]
    tilt_deg = torch.rad2deg(torch.arccos(up_dot))
    return torch.where(tilt_deg > thresh_deg, tilt_deg - thresh_deg, 0.0)


def forward_vel(env: ManagerBasedEnv) -> torch.Tensor:
    """Convenience reward: forward x velocity in world frame."""
    return mdp.base_lin_vel(env)[:, 0]


def progress_rew(env: ManagerBasedEnv) -> torch.Tensor:
    """Reward for passing new waypoints (handles lap transitions).

    Implementation
    --------------
    - Calls `progress_waypoint_bool` to get a (bool, amount) pair for each env.
    - Gates by: no recent off-track, no recent opponent collision.
    - Scales progress by `LEN_S_IDX` to convert index delta into meters.
    """
    progress_bool, progress = progress_waypoint_bool(env)

    if not hasattr(env, "_wall_collision_history"):
        env._wall_collision_history = torch.zeros((env.num_envs, CONFIG["env_config"]["REW_HISTORY_LENGTH"]), dtype=torch.long, device=env.device)
    if not hasattr(env, "_opponent_collision_history"):
        env._opponent_collision_history = torch.zeros((env.num_envs, CONFIG["env_config"]["REW_HISTORY_LENGTH"]), dtype=torch.long, device=env.device)

    # Only current step off-track matters here (kept as in original)
    no_off_track = env._wall_collision_history[:, 0] == 0
    no_opp_collision = env._opponent_collision_history.max(dim=1).values == 0

    return torch.where(progress_bool & no_off_track & no_opp_collision, progress * CONFIG["env_config"]["LEN_S_IDX"], 0.0)


def progress_waypoint_bool(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Track per-step waypoint progress, with wrap-around on circular tracks.

    Returns:
        progress_bool: [num_envs] True if new progress within allowed window.
        progress:      [num_envs] Δs_index (modulo N), 0..N-1

    Side effects:
        - Updates `env._progress_history_indices` (rolling indices).
        - Stores rich `env.extras` for logging/visualization tooling.
    """
    # Histories / flags
    if not hasattr(env, "_progress_history_indices"):
        env._rew_history_length = CONFIG["env_config"]["REW_HISTORY_LENGTH"]
        env._progress_history_indices = torch.zeros((env.num_envs, env._rew_history_length), dtype=torch.long, device=env.device)
        env._reset_env_bool = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_progress_history_checkpoint_idx"):
        env._progress_history_checkpoint_idx = CONFIG["env_config"]["PROGRESS_HISTORY_CHECK_IDX"]
    if not hasattr(env, "_total_progress_indices"):
        env._total_progress_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    position_xy_world = mdp.root_pos_w(env)[..., :2]

    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_waypoints_list"):
        env._waypoints_list = [torch.tensor(wps, device=env.device, dtype=torch.float32)
                               for wps in env.scene.terrain.cfg.waypoints_list]
    if not hasattr(env, "_initial_waypoint_indices"):
        env._initial_waypoint_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_wall_collision_history"):
        env._wall_collision_history = torch.zeros((env.num_envs, CONFIG["env_config"]["REW_HISTORY_LENGTH"]), dtype=torch.long, device=env.device)

    no_off_track = env._wall_collision_history.max(dim=1).values == 0

    map_levels = env._map_levels
    unique_map_levels = torch.unique(map_levels)

    progress_bool = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    for map_level in unique_map_levels:
        env_mask = (map_levels == map_level)
        if not env_mask.any():
            continue

        waypoints_world = env._waypoints_list[map_level][:, :2]
        current_idx, _ = find_frenet_coord_along_waypoints(waypoints_world, position_xy_world[env_mask])
        num_waypoints = len(waypoints_world)

        # Update rolling index history
        env._progress_history_indices[env_mask, 1:] = env._progress_history_indices[env_mask, :-1].clone()
        env._progress_history_indices[env_mask, 0] = current_idx

        # Δindex modulo track length
        current_progress = (current_idx - env._progress_history_indices[env_mask, env._progress_history_checkpoint_idx]) % num_waypoints
        progress[env_mask] = current_progress

        progress_bool[env_mask] = (
            (current_progress > 0)
            & (current_progress <= CONFIG["env_config"]["MAX_PROGRESS_IDX"])
            & (env._reset_env_bool[env_mask] == False)
        )

    # Reset the "just reset" flag after first step post-reset
    env._reset_env_bool = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Store extras for visualization / logging (kept as in original)
    asset = env.scene["robot"]

    if not hasattr(env, "_vel_y_calc"):
        env._vel_y_calc = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "_target_velocity_history"):
        env._obs_history_length = CONFIG["env_config"]["OBS_HISTORY_LENGTH"]
        env._target_velocity_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
        env._target_steering_angle_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)

    env._vel_y_calc = mdp.base_lin_vel(env)[:, 1] * mdp.base_lin_vel(env)[:, 0] * 1.2

    if not hasattr(env, "_opponent_xy_position"):
        env._opponent_xy_position = torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device)
    if not hasattr(env, "_opponent_speed"):
        env._opponent_speed = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # ---- extras for play_policy.py / WandB (unchanged behavior) ----
    env.extras["theta"] = asset.data.heading_w
    env.extras["pos_xy"] = position_xy_world
    env.extras["vel_x"] = asset.data.root_lin_vel_b[:, 0]
    env.extras["vel_y"] = -asset.data.root_lin_vel_b[:, 1]
    env.extras["target_velocity"] = env._target_velocity_history[:, 0].clone()
    env.extras["target_steering"] = env._target_steering_angle_history[:, 0].clone()
    env.extras["yaw_rate"] = asset.data.root_ang_vel_b[:, 2]
    env.extras["s_idx"] = current_idx.clone()
    env.extras["time"] = torch.tensor(env.sim.current_time, device=env.device)
    env.extras["s_idx_max"] = torch.tensor(num_waypoints, device=env.device)

    env.extras["vel_y_calc"] = env._vel_y_calc

    vel = mdp.base_lin_vel(env)
    slip_angle = torch.abs(torch.atan2(vel[..., 1], vel[..., 0]))
    moving_mask = torch.abs(vel[..., 0]) >= 2

    env.extras["log"]["Info/max_slip_angle"] = torch.max(slip_angle * moving_mask)
    env.extras["log"]["Info/mean_slip_angle"] = torch.mean(slip_angle * moving_mask)
    env.extras["log"]["Info/mean_speed"] = torch.mean(env._base_lin_vel_x_history)
    env.extras["log"]["Info/max_speed"] = torch.max(env._base_lin_vel_x_history)
    env.extras["log"]["Info/mean_delta_target_speed"] = torch.mean(
        torch.abs(env._target_steering_angle_history - env._base_lin_vel_x_history)
    )
    # ---------------------------------------------------------------

    return progress_bool, progress


# ---------------------------------------------------------------------------
# Target tracking penalties (match commands to motion)
# ---------------------------------------------------------------------------

def delta_target_velocity_penalty(env: ManagerBasedEnv, threshold: float = 0.5) -> torch.Tensor:
    """Penalty if target speed exceeds actual speed by more than a threshold.

    Rationale
    ---------
    Asking for much more speed than the vehicle currently has can cause tail instability
    and wheel spin. We penalize positive (target - actual) beyond `threshold`.
    """
    if not hasattr(env, "_base_lin_vel_x_history"):
        env._base_lin_vel_x_history = torch.zeros((env.num_envs, CONFIG["env_config"]["OBS_HISTORY_LENGTH"]), dtype=torch.float32, device=env.device)
    if not hasattr(env, "_target_velocity_history"):
        env._target_velocity_history = torch.zeros((env.num_envs, CONFIG["env_config"]["OBS_HISTORY_LENGTH"]), dtype=torch.float32, device=env.device)

    actual_vel = env._base_lin_vel_x_history[:, 0]
    target_vel = env._target_velocity_history[:, 0]
    delta_v = target_vel - actual_vel
    return -torch.clamp(delta_v - threshold, min=0.0)


def delta_speed_cmd_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Penalty for mismatch between last speed command and actual speed.

    - Converts the last throttle action to a speed command (scaled & offset).
    - Penalizes squared error vs. current forward speed.
    """
    last_throttle_action = mdp.last_action(env)[..., 0]
    last_speed_cmd = torch.clamp(
        last_throttle_action * CONFIG["env_config"]["MAX_SPEED_SCALING"] + CONFIG["env_config"]["SPEED_OFFSET"],
        min=0,
        max=CONFIG["env_config"]["MAX_SPEED_SCALING"],
    )
    speed = mdp.base_lin_vel(env)[..., 0]
    return -(last_speed_cmd - speed) ** 2


# ---------------------------------------------------------------------------
# Action smoothness & effort penalties
# ---------------------------------------------------------------------------

def negative_throttle_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Penalty for commanding negative throttle (i.e., braking request).

    If your task uses a pure speed target (non-negative), discouraging negative throttle
    keeps the controller from oscillating around zero.
    """
    last_throttle_action = mdp.last_action(env)[..., 0]
    return torch.where(last_throttle_action < 0.0, -1.0, 0.0)


def var_throttle_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Penalty for high variance in recent throttle commands."""
    if not hasattr(env, "_action_history"):
        env._action_history_length = CONFIG["env_config"]["ACTION_HISTORY_LENGTH"]
        env._action_history = torch.zeros((env.num_envs, env._action_history_length, 2), dtype=torch.float32, device=env.device)
    var_throttle = torch.var(env._action_history[:, :, 0], dim=1)
    return -var_throttle


def var_steering_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Penalty for high variance in recent steering commands."""
    if not hasattr(env, "_action_history"):
        env._action_history_length = CONFIG["env_config"]["ACTION_HISTORY_LENGTH"]
        env._action_history = torch.zeros((env.num_envs, env._action_history_length, 2), dtype=torch.float32, device=env.device)
    var_steering = torch.var(env._action_history[:, :, 1], dim=1)
    return -var_steering


def var_throttle_rate_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Penalty on throttle jerk (variance of command time-derivative)."""
    if not hasattr(env, "_action_history"):
        env._action_history_length = CONFIG["env_config"]["ACTION_HISTORY_LENGTH"]
        env._action_history = torch.zeros((env.num_envs, env._action_history_length, 2), dtype=torch.float32, device=env.device)
    throttle_rate = torch.diff(env._action_history[:, :, 0], dim=1)
    var_throttle_rate = torch.var(throttle_rate, dim=1)
    return -var_throttle_rate


def delta_throttle_l2_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Penalty on step-to-step change in throttle (squared L2)."""
    if not hasattr(env, "_action_history"):
        env._action_history_length = CONFIG["env_config"]["ACTION_HISTORY_LENGTH"]
        env._action_history = torch.zeros((env.num_envs, env._action_history_length, 2), dtype=torch.float32, device=env.device)
    return -(env._action_history[:, 0, 0] - env._action_history[:, 1, 0]) ** 2


def delta_steering_l2_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Penalty on step-to-step change in steering (squared L2)."""
    if not hasattr(env, "_action_history"):
        env._action_history_length = CONFIG["env_config"]["ACTION_HISTORY_LENGTH"]
        env._action_history = torch.zeros((env.num_envs, env._action_history_length, 2), dtype=torch.float32, device=env.device)
    return -(env._action_history[:, 0, 1] - env._action_history[:, 1, 1]) ** 2


def effort_throttle_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Quadratic penalty on current throttle effort (discourages saturation)."""
    return -(env._action_history[:, 0, 0]) ** 2


def effort_steering_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Quadratic penalty on current steering effort (discourages saturation)."""
    return -(env._action_history[:, 0, 1]) ** 2


def effort_target_steering_angle_penalty(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mean_noise: float = 0.0,
    std_noise: float = 0.0,
) -> torch.Tensor:
    """Quadratic penalty on the (integrated) target steering angle.

    This integrates the steering command (incremental mode) into a running
    target angle (bounded by ±MAX_STEERING_ANGLE), then penalizes its square.

    Note: this duplicates logic used in observations for target steering history,
    kept here to preserve original behavior.
    """
    asset: RigidObject = env.scene[asset_cfg.name]  # not used directly; kept for parity
    if not hasattr(env, "_target_steering_angle_history"):
        env._obs_history_length = CONFIG["env_config"]["OBS_HISTORY_LENGTH"]
        env._target_steering_angle_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)

    last_action = mdp.last_action(env)[..., 1] * CONFIG["env_config"]["MAX_STEERING_ANGLE_INCREMENT"]
    env._target_steering_angle_history[:, 1:] = env._target_steering_angle_history[:, :-1].clone()
    env._target_steering_angle_history[:, 0] = torch.clamp(
        env._target_steering_angle_history[:, 0] + last_action, min=-0.45, max=0.45
    )
    return -(env._target_steering_angle_history[:, 0]) ** 2
