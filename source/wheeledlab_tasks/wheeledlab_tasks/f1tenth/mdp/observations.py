"""
Observations for the F1Tenth overtake task.

This module exposes small, composable observation-term functions that:
- read state from the IsaacLab `ManagerBasedEnv`,
- maintain short rolling histories in `env` buffers,
- compute geometric features from racetrack waypoints (deviation, heading error, d_lat, curvature),
- compute ego/opponent relative features for overtaking.

Design notes
------------
- Every function is written to be idempotent and safe to call at every sim step.
- Rolling histories are stored on the `env` instance (e.g., `env._base_lin_vel_x_history`).
- When multiple maps are used concurrently (vectorized envs), we group envs by map level to
  avoid mixing waypoint arrays.
- Shapes are consistent across calls; do not change them without adapting consumers.
- We avoid heavy allocations inside the step loop; histories are allocated once per env.

Conventions
-----------
- Frames: world (w) and robot body/root (b). Velocities from IsaacLab helpers use these.
- Waypoint geometry uses world frame coordinates.
- "s" denotes longitudinal progress (index along waypoints). "d" denotes lateral offset.
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


# --------------------------------------------------------------------------
# Simple instantaneous terms (no internal history)
# --------------------------------------------------------------------------

def base_lin_vel_x(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mean_noise: float = 0.0,
    std_noise: float = 0.0,
) -> torch.Tensor:
    """Return ego linear velocity x (body frame) with optional Gaussian noise.

    Shape: [num_envs, 1]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    v = asset.data.root_lin_vel_b[:, 0].unsqueeze(-1)
    if std_noise > 0:
        noise = torch.empty_like(v).normal_(mean=mean_noise, std=std_noise)
        v = v + noise
    return v


def base_lin_vel_y(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mean_noise: float = 0.0,
    std_noise: float = 0.0,
) -> torch.Tensor:
    """Return ego linear velocity y (body frame) with optional Gaussian noise.

    Shape: [num_envs, 1]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    v = asset.data.root_lin_vel_b[:, 1].unsqueeze(-1)
    if std_noise > 0:
        noise = torch.empty_like(v).normal_(mean=mean_noise, std=std_noise)
        v = v + noise
    return v


def base_ang_vel_z(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mean_noise: float = 0.0,
    std_noise: float = 0.0,
) -> torch.Tensor:
    """Return ego angular velocity z (yaw rate, body frame) with optional noise.

    Shape: [num_envs, 1]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    w = asset.data.root_ang_vel_b[:, 2].unsqueeze(-1)
    # Low-risk fix: ensure `noise` is defined if std_noise==0.
    if std_noise > 0:
        noise = torch.empty_like(w).normal_(mean=mean_noise, std=std_noise)
        w = w + noise
    return w


# --------------------------------------------------------------------------
# Rolling histories (allocate once, shift each step)
# --------------------------------------------------------------------------

def base_lin_vel_x_history(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mean_noise: float = 0.0,
    std_noise: float = 0.0,
) -> torch.Tensor:
    """Return rolling history of ego v_x (body frame).

    - Allocates `env._base_lin_vel_x_history` if missing.
    - Shifts right, inserts newest value at index 0.
    Shape: [num_envs, history_len]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    if not hasattr(env, "_base_lin_vel_x_history"):
        env._obs_history_length = CONFIG["env_config"]["OBS_HISTORY_LENGTH"]
        env._base_lin_vel_x_history = torch.zeros(
            (env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device
        )

    # Latest sample (optionally noisy)
    v = asset.data.root_lin_vel_b[:, 0]
    if std_noise > 0:
        v = v + torch.empty_like(v).normal_(mean=mean_noise, std=std_noise)

    env._base_lin_vel_x_history[:, 1:] = env._base_lin_vel_x_history[:, :-1].clone()
    env._base_lin_vel_x_history[:, 0] = v
    return env._base_lin_vel_x_history


def base_lin_vel_y_history(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mean_noise: float = 0.0,
    std_noise: float = 0.15,
) -> torch.Tensor:
    """Return rolling history of ego v_y (body frame), noise sign-preserving.

    - Optional noise added; sign preserved to avoid flipping lateral direction.
    Shape: [num_envs, history_len]
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    if not hasattr(env, "_base_lin_vel_y_history"):
        env._obs_history_length = CONFIG["env_config"]["OBS_HISTORY_LENGTH"]
        env._base_lin_vel_y_history = torch.zeros(
            (env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device
        )

    vy = -asset.data.root_lin_vel_b[:, 1]  # note the project’s convention is `-vy`
    if std_noise > 0:
        noise = torch.randn_like(vy) * std_noise + mean_noise
        vy_noisy = vy + noise
        # Preserve sign (avoid crossing zero just due to noise)
        vy_noisy = torch.where(vy > 0, torch.clamp(vy_noisy, min=0.0), vy_noisy)
        vy_noisy = torch.where(vy < 0, torch.clamp(vy_noisy, max=0.0), vy_noisy)
        vy = vy_noisy

    env._base_lin_vel_y_history[:, 1:] = env._base_lin_vel_y_history[:, :-1].clone()
    env._base_lin_vel_y_history[:, 0] = vy
    return env._base_lin_vel_y_history


def base_ang_vel_z_history(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mean_noise: float = 0.0,
    std_noise: float = 0.0,
) -> torch.Tensor:
    """Return rolling history of ego yaw rate (body frame).

    Shape: [num_envs, history_len]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    if not hasattr(env, "_base_ang_vel_z_history"):
        env._obs_history_length = CONFIG["env_config"]["OBS_HISTORY_LENGTH"]
        env._base_ang_vel_z_history = torch.zeros(
            (env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device
        )

    w = asset.data.root_ang_vel_b[:, 2]
    if std_noise > 0:
        w = w + torch.empty_like(w).normal_(mean=mean_noise, std=std_noise)

    env._base_ang_vel_z_history[:, 1:] = env._base_ang_vel_z_history[:, :-1].clone()
    env._base_ang_vel_z_history[:, 0] = w
    return env._base_ang_vel_z_history


def base_lin_acc_x_history(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mean_noise: float = 0.0,
    std_noise: float = 0.0,
) -> torch.Tensor:
    """Approximate x-acceleration from v_x history (finite difference + smoothing).

    Notes:
    - Requires `base_lin_vel_x_history` to have been updated before this call.
    - Exponential smoothing uses 0.5 / 0.5 current/previous weights.
    Shape: [num_envs, history_len]
    """
    if not hasattr(env, "_base_lin_acc_x_history"):
        env._obs_history_length = CONFIG["env_config"]["OBS_HISTORY_LENGTH"]
        env._base_lin_acc_x_history = torch.zeros(
            (env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device
        )

    # Finite diff on the last two velocity samples
    raw_acc = (env._base_lin_vel_x_history[:, 0] - env._base_lin_vel_x_history[:, 1]) / env.step_dt
    if std_noise > 0:
        raw_acc = raw_acc + torch.empty_like(raw_acc).normal_(mean=mean_noise, std=std_noise)

    env._base_lin_acc_x_history[:, 1:] = env._base_lin_acc_x_history[:, :-1].clone()
    env._base_lin_acc_x_history[:, 0] = 0.5 * raw_acc + 0.5 * env._base_lin_acc_x_history[:, 1]
    return env._base_lin_acc_x_history


def target_velocity_history(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mean_noise: float = 0.0,
    std_noise: float = 0.0,
) -> torch.Tensor:
    """Integrate target speed command (incremental mode) into a rolling history.

    - Uses last action channel 0 as delta speed scaled by `MAX_SPEED_INCREMENT`.
    - Clamps to non-negative speeds.
    Shape: [num_envs, history_len]
    """
    if not hasattr(env, "_target_velocity_history"):
        env._obs_history_length = CONFIG["env_config"]["OBS_HISTORY_LENGTH"]
        env._target_velocity_history = torch.zeros(
            (env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device
        )
        # Ensure steering history also exists if caller expects both
        env._target_steering_angle_history = torch.zeros(
            (env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device
        )

    delta_v = mdp.last_action(env)[..., 0] * CONFIG["env_config"]["MAX_SPEED_INCREMENT"]
    env._target_velocity_history[:, 1:] = env._target_velocity_history[:, :-1].clone()
    env._target_velocity_history[:, 0] = torch.clamp(env._target_velocity_history[:, 0] + delta_v, min=0.0)
    return env._target_velocity_history


def target_steering_angle_history(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mean_noise: float = 0.0,
    std_noise: float = 0.0,
) -> torch.Tensor:
    """Integrate target steering angle in incremental mode into a rolling history.

    - Uses last action channel 1 as delta steering scaled by `MAX_STEERING_ANGLE_INCREMENT`.
    - Clamps by ±`MAX_STEERING_ANGLE`.
    Shape: [num_envs, history_len]
    """
    if not hasattr(env, "_target_steering_angle_history"):
        env._obs_history_length = CONFIG["env_config"]["OBS_HISTORY_LENGTH"]
        env._target_steering_angle_history = torch.zeros(
            (env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device
        )

    delta = mdp.last_action(env)[..., 1] * CONFIG["env_config"]["MAX_STEERING_ANGLE_INCREMENT"]
    env._target_steering_angle_history[:, 1:] = env._target_steering_angle_history[:, :-1].clone()
    env._target_steering_angle_history[:, 0] = torch.clamp(
        env._target_steering_angle_history[:, 0] + delta,
        min=-CONFIG["env_config"]["MAX_STEERING_ANGLE"],
        max=CONFIG["env_config"]["MAX_STEERING_ANGLE"],
    )
    return env._target_steering_angle_history


def action_history(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mean_noise: float = 0.0,
    std_noise: float = 0.0,
) -> torch.Tensor:
    """Maintain a rolling history of the last raw actions (throttle, steering).

    - No clipping inside: actions are assumed in [-1, 1] upstream.
    - Returns a flattened view for the observation space.
    Shape: [num_envs, history_len * 2]
    """
    if not hasattr(env, "_action_history"):
        env._action_history_length = CONFIG["env_config"]["ACTION_HISTORY_LENGTH"]
        env._action_history = torch.zeros(
            (env.num_envs, env._action_history_length, 2), dtype=torch.float32, device=env.device
        )

    a = mdp.last_action(env)[..., :]
    env._action_history[:, 1:, :] = env._action_history[:, :-1, :].clone()
    env._action_history[:, 0, :] = a
    return env._action_history.reshape(-1, env._action_history_length * 2)


# --------------------------------------------------------------------------
# Waypoint-based horizon helpers
# --------------------------------------------------------------------------

def _init_env_waypoints(env: ManagerBasedEnv) -> None:
    """Ensure all per-map waypoint tensors exist on the env (lazy alloc)."""
    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_waypoints_list"):
        env._waypoints_list = [
            torch.tensor(wps, device=env.device, dtype=torch.float32) for wps in env.scene.terrain.cfg.waypoints_list
        ]
    if not hasattr(env, "_inner_list"):
        env._inner_list = [
            torch.tensor(inner, device=env.device, dtype=torch.float32) for inner in env.scene.terrain.cfg.inner_list
        ]
    if not hasattr(env, "_d_lat_list"):
        env._d_lat_list = [
            torch.tensor(d_lat, device=env.device, dtype=torch.float32) for d_lat in env.scene.terrain.cfg.d_lat_list
        ]
    if not hasattr(env, "_kappa_radpm_list"):
        env._kappa_radpm_list = [
            torch.tensor(kappa, device=env.device, dtype=torch.float32)
            for kappa in env.scene.terrain.cfg.kappa_radpm_list
        ]


def _calculate_horizon_indices(
    current_idx: torch.Tensor,
    idx_horizon: torch.Tensor,
    waypoints_length: int,
    n_horizon: int,
    device: torch.device,
) -> torch.Tensor:
    """Vectorized indices for n_horizon lookahead points starting from current_idx.

    Args:
        current_idx: [N] current waypoint indices
        idx_horizon: [N] step size in waypoint indices (can be fractional)
        waypoints_length: scalar, total number of waypoints
        n_horizon: number of lookahead points
        device: torch device for allocations

    Returns:
        [N, n_horizon + 1] integer indices, including the current point (step 0).
    """
    steps_norm = torch.linspace(0, 1, n_horizon + 1, device=device)  # [n_horizon+1]
    steps = (idx_horizon.unsqueeze(-1) * steps_norm.unsqueeze(0)).long()
    return (current_idx.unsqueeze(-1) + steps) % waypoints_length


def get_horizon_indices(
    env: ManagerBasedEnv,
    map_level: int,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    delta_s_idx: int,
    n_horizon: int,
    t_horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute waypoint indices along the reference path for a batch of poses.

    Returns:
        horizon_indices: [batch, n_horizon+1]
        waypoints_xy_world: [n_waypoints, 2] (for convenience to index later)
    """
    waypoints_xy_world = env._waypoints_list[map_level][:, :2]
    inner_xy_world = env._inner_list[map_level][:, :2]

    # nearest waypoint to each position
    current_idx, _ = find_frenet_coord_along_waypoints(inner_xy_world, positions)

    if CONFIG["env_config"]["DYNAMIC_LOOKAHEAD"]:
        s_length = CONFIG["env_config"]["LEN_S_IDX"]           # meters per waypoint index
        s_horizon = velocities * t_horizon                     # meters to look ahead
        idx_horizon = s_horizon / s_length                     # convert to indices
        horizon_indices = _calculate_horizon_indices(
            current_idx=current_idx,
            idx_horizon=idx_horizon,
            waypoints_length=len(waypoints_xy_world),
            n_horizon=n_horizon,
            device=env.device,
        )
    else:
        # Fixed-step lookahead in index space
        lookahead_steps = torch.linspace(0, delta_s_idx * n_horizon, n_horizon + 1, device=env.device)
        horizon_indices = (current_idx.unsqueeze(-1) + lookahead_steps.unsqueeze(0)) % len(waypoints_xy_world)
        horizon_indices = horizon_indices.long()

    return horizon_indices, waypoints_xy_world


# --------------------------------------------------------------------------
# Composite waypoint-based observations (multi-feature per call)
# --------------------------------------------------------------------------

def track_info_horizon(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    delta_s_idx: int = 10,
    n_horizon: int = 5,
    t_horizon: int = 5,
    position_std_noise: float = 0.0,
) -> torch.Tensor:
    """Compute deviation, heading error, left/right d_lat, and curvature over a horizon.

    Returns:
        [num_envs, n_horizon (dev) + n_horizon (heading) + 2*n_horizon (d_lat L/R) + n_horizon (kappa)]
    """
    # Ego state in world frame
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
    if position_std_noise > 0.0:
        pos_xy_world = pos_xy_world + torch.normal(
            mean=0.0, std=position_std_noise, size=pos_xy_world.shape, device=pos_xy_world.device
        )
    vel_x = mdp.base_lin_vel(env)[..., 0]
    heading_w = env.scene[asset_cfg.name].data.heading_w
    num_envs = pos_xy_world.shape[0]

    _init_env_waypoints(env)

    deviation_obs = torch.zeros(num_envs, n_horizon, device=env.device)
    heading_obs = torch.zeros(num_envs, n_horizon, device=env.device)
    dlat_obs = torch.zeros(num_envs, n_horizon * 2, device=env.device)
    kappa_obs = torch.zeros(num_envs, n_horizon, device=env.device)

    map_levels = env._map_levels
    for map_level in torch.unique(map_levels):
        mask = map_levels == map_level
        if not mask.any():
            continue

        map_pos = pos_xy_world[mask]
        map_vel_x = vel_x[mask]
        map_head = heading_w[mask]

        horizon_indices, waypoints_xy_world = get_horizon_indices(
            env, map_level, map_pos, map_vel_x, delta_s_idx, n_horizon, t_horizon
        )

        # 1) Signed deviation to segment between consecutive waypoints
        seg_starts = waypoints_xy_world[horizon_indices[:, :-1]]
        seg_ends = waypoints_xy_world[horizon_indices[:, 1:]]
        track_dirs = seg_ends - seg_starts
        car_offsets = map_pos.unsqueeze(1) - seg_starts
        cross = track_dirs[:, :, 0] * car_offsets[:, :, 1] - track_dirs[:, :, 1] * car_offsets[:, :, 0]
        sign = torch.sign(cross)
        distances = torch.abs(cross) / (torch.norm(track_dirs, dim=2) + 1e-6)
        deviation_obs[mask] = sign * distances

        # 2) Heading error (shortest signed angle difference)
        lookahead_pts = waypoints_xy_world[horizon_indices[:, 1:]]
        desired_headings = torch.atan2(
            lookahead_pts[:, :, 1] - map_pos[:, 1].unsqueeze(-1),
            lookahead_pts[:, :, 0] - map_pos[:, 0].unsqueeze(-1),
        )
        heading_err = torch.atan2(
            torch.sin(desired_headings - map_head.unsqueeze(-1)),
            torch.cos(desired_headings - map_head.unsqueeze(-1)),
        )
        heading_obs[mask] = heading_err

        # 3) Lateral available space (left/right) at future indices
        d_lat = env._d_lat_list[map_level][:, :2]
        next_dlat = d_lat[horizon_indices[:, 1:], :]
        dlat_obs[mask] = next_dlat.reshape(-1, n_horizon * 2)

        # 4) Curvature (kappa) at future indices
        kappa_radpm = env._kappa_radpm_list[map_level][:]
        next_kappa = kappa_radpm[horizon_indices[:, 1:]].squeeze(-1)
        kappa_obs[mask] = next_kappa

    return torch.cat([deviation_obs, heading_obs, dlat_obs, kappa_obs], dim=-1)


# --------------------------------------------------------------------------
# Individual horizon features (exposed separately for modularity)
# --------------------------------------------------------------------------

def deviation_centerline_horizon(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    delta_s_idx: int = 10,
    n_horizon: int = 5,
    t_horizon: int = 5,
) -> torch.Tensor:
    """Signed lateral deviation from multiple future track segments.

    Returns:
        [num_envs, n_horizon]
    """
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
    vel_x = mdp.base_lin_vel(env)[..., 0]
    num_envs = pos_xy_world.shape[0]

    # Ensure waypoint tensors exist
    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_waypoints_list"):
        env._waypoints_list = [torch.tensor(wps, device=env.device, dtype=torch.float32)
                               for wps in env.scene.terrain.cfg.waypoints_list]
    if not hasattr(env, "_inner_list"):
        env._inner_list = [torch.tensor(inner, device=env.device, dtype=torch.float32)
                           for inner in env.scene.terrain.cfg.inner_list]

    deviations = torch.zeros(num_envs, n_horizon, device=env.device)
    map_levels = env._map_levels
    for map_level in torch.unique(map_levels):
        env_mask = map_levels == map_level
        if not env_mask.any():
            continue

        map_positions = pos_xy_world[env_mask]
        map_vel_x = vel_x[env_mask]
        waypoints_xy_world = env._waypoints_list[map_level][:, :2]
        inner_xy_world = env._inner_list[map_level][:, :2]

        current_idx, _ = find_frenet_coord_along_waypoints(inner_xy_world, map_positions)

        if CONFIG["env_config"]["DYNAMIC_LOOKAHEAD"]:
            s_length = CONFIG["env_config"]["LEN_S_IDX"]
            s_horizon = map_vel_x * t_horizon
            idx_horizon = s_horizon / s_length
            horizon_indices = _calculate_horizon_indices(
                current_idx, idx_horizon, len(waypoints_xy_world), n_horizon, env.device
            )
        else:
            lookahead_steps = torch.linspace(0, delta_s_idx * n_horizon, n_horizon + 1, device=env.device)
            horizon_indices = (current_idx.unsqueeze(-1) + lookahead_steps.unsqueeze(0)) % len(waypoints_xy_world)
            horizon_indices = horizon_indices.long()

        seg_starts = waypoints_xy_world[horizon_indices[:, :-1]]
        seg_ends = waypoints_xy_world[horizon_indices[:, 1:]]
        track_dirs = seg_ends - seg_starts
        car_offsets = map_positions.unsqueeze(1) - seg_starts
        cross_products = track_dirs[:, :, 0] * car_offsets[:, :, 1] - track_dirs[:, :, 1] * car_offsets[:, :, 0]
        signs = torch.sign(cross_products)
        segment_lengths = torch.norm(track_dirs, dim=2)
        distances = torch.abs(cross_products) / (segment_lengths + 1e-6)
        deviations[env_mask] = signs * distances  # normalization left at 1
    return deviations


def heading_error_horizon(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    delta_s_idx: int = 10,
    n_horizon: int = 5,
    t_horizon: int = 5,
) -> torch.Tensor:
    """Heading error to future points (shortest signed angle difference).

    Returns:
        [num_envs, n_horizon]
    """
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
    vel_x = mdp.base_lin_vel(env)[..., 0]
    heading_w = env.scene[asset_cfg.name].data.heading_w
    num_envs = pos_xy_world.shape[0]

    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_waypoints_list"):
        env._waypoints_list = [torch.tensor(wps, device=env.device, dtype=torch.float32)
                               for wps in env.scene.terrain.cfg.waypoints_list]
    if not hasattr(env, "_inner_list"):
        env._inner_list = [torch.tensor(inner, device=env.device, dtype=torch.float32)
                           for inner in env.scene.terrain.cfg.inner_list]

    heading_errors = torch.zeros(num_envs, n_horizon, device=env.device)
    map_levels = env._map_levels
    for map_level in torch.unique(map_levels):
        env_mask = map_levels == map_level
        if not env_mask.any():
            continue

        map_positions = pos_xy_world[env_mask]
        map_vel_x = vel_x[env_mask]
        map_headings = heading_w[env_mask]

        waypoints_xy_world = env._waypoints_list[map_level][:, :2]
        inner_xy_world = env._inner_list[map_level][:, :2]

        current_idx, _ = find_frenet_coord_along_waypoints(inner_xy_world, map_positions)

        if CONFIG["env_config"]["DYNAMIC_LOOKAHEAD"]:
            s_length = CONFIG["env_config"]["LEN_S_IDX"]
            s_horizon = map_vel_x * t_horizon
            idx_horizon = s_horizon / s_length
            horizon_indices = _calculate_horizon_indices(
                current_idx, idx_horizon, len(waypoints_xy_world), n_horizon, env.device
            )
        else:
            lookahead_steps = torch.linspace(0, delta_s_idx * n_horizon, n_horizon + 1, device=env.device)
            horizon_indices = (current_idx.unsqueeze(-1) + lookahead_steps.unsqueeze(0)) % len(waypoints_xy_world)
            horizon_indices = horizon_indices.long()

        lookahead_points = waypoints_xy_world[horizon_indices[:, 1:]]
        desired_headings = torch.atan2(
            lookahead_points[:, :, 1] - map_positions[:, 1].unsqueeze(-1),
            lookahead_points[:, :, 0] - map_positions[:, 0].unsqueeze(-1),
        )
        current_heading_errors = torch.atan2(
            torch.sin(desired_headings - map_headings.unsqueeze(-1)),
            torch.cos(desired_headings - map_headings.unsqueeze(-1)),
        )
        heading_errors[env_mask] = current_heading_errors  # normalization left at 1
    return heading_errors


def d_lat_horizon(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    delta_s_idx: int = 10,
    n_horizon: int = 5,
    t_horizon: int = 5,
) -> torch.Tensor:
    """Left/right lateral available space to track bounds at future points.

    Returns:
        [num_envs, n_horizon * 2]  (interleaved [left, right] per horizon step)
    """
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
    vel_x = mdp.base_lin_vel(env)[..., 0]
    num_envs = pos_xy_world.shape[0]

    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_waypoints_list"):
        env._waypoints_list = [torch.tensor(wps, device=env.device, dtype=torch.float32)
                               for wps in env.scene.terrain.cfg.waypoints_list]
    if not hasattr(env, "_inner_list"):
        env._inner_list = [torch.tensor(inner, device=env.device, dtype=torch.float32)
                           for inner in env.scene.terrain.cfg.inner_list]
    if not hasattr(env, "_d_lat_list"):
        env._d_lat_list = [torch.tensor(d, device=env.device, dtype=torch.float32)
                           for d in env.scene.terrain.cfg.d_lat_list]

    d_lat_results = torch.zeros(num_envs, n_horizon * 2, device=env.device)
    map_levels = env._map_levels
    for map_level in torch.unique(map_levels):
        env_mask = map_levels == map_level
        if not env_mask.any():
            continue

        map_positions = pos_xy_world[env_mask]
        map_vel_x = vel_x[env_mask]
        waypoints_xy_world = env._waypoints_list[map_level][:, :2]
        inner_xy_world = env._inner_list[map_level][:, :2]
        d_lat = env._d_lat_list[map_level][:, :2]

        current_idx, _ = find_frenet_coord_along_waypoints(inner_xy_world, map_positions)

        if CONFIG["env_config"]["DYNAMIC_LOOKAHEAD"]:
            s_length = CONFIG["env_config"]["LEN_S_IDX"]
            s_horizon = map_vel_x * t_horizon
            idx_horizon = s_horizon / s_length
            horizon_indices = _calculate_horizon_indices(
                current_idx, idx_horizon, len(waypoints_xy_world), n_horizon, env.device
            )
        else:
            lookahead_steps = torch.linspace(0, delta_s_idx * n_horizon, n_horizon + 1, device=env.device)
            horizon_indices = (current_idx.unsqueeze(-1) + lookahead_steps.unsqueeze(0)) % len(waypoints_xy_world)
            horizon_indices = horizon_indices.long()

        next_d_lat_horizon = d_lat[horizon_indices[:, 1:], :]
        d_lat_results[env_mask] = next_d_lat_horizon.reshape(-1, n_horizon * 2)  # normalization left at 1
    return d_lat_results


def kappa_radpm_horizon(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    delta_s_idx: int = 10,
    n_horizon: int = 5,
    t_horizon: int = 5,
) -> torch.Tensor:
    """Curvature (kappa) at future indices along the track.

    Returns:
        [num_envs, n_horizon]
    """
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
    vel_x = mdp.base_lin_vel(env)[..., 0]
    num_envs = pos_xy_world.shape[0]

    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_waypoints_list"):
        env._waypoints_list = [torch.tensor(wps, device=env.device, dtype=torch.float32)
                               for wps in env.scene.terrain.cfg.waypoints_list]
    if not hasattr(env, "_inner_list"):
        env._inner_list = [torch.tensor(inner, device=env.device, dtype=torch.float32)
                           for inner in env.scene.terrain.cfg.inner_list]
    if not hasattr(env, "_kappa_radpm_list"):
        env._kappa_radpm_list = [torch.tensor(k, device=env.device, dtype=torch.float32)
                                 for k in env.scene.terrain.cfg.kappa_radpm_list]

    kappa_results = torch.zeros(num_envs, n_horizon, device=env.device)
    map_levels = env._map_levels
    for map_level in torch.unique(map_levels):
        env_mask = map_levels == map_level
        if not env_mask.any():
            continue

        map_positions = pos_xy_world[env_mask]
        map_vel_x = vel_x[env_mask]
        waypoints_xy_world = env._waypoints_list[map_level][:, :2]
        inner_xy_world = env._inner_list[map_level][:, :2]
        kappa_radpm = env._kappa_radpm_list[map_level][:]

        current_idx, _ = find_frenet_coord_along_waypoints(inner_xy_world, map_positions)

        if CONFIG["env_config"]["DYNAMIC_LOOKAHEAD"]:
            s_length = CONFIG["env_config"]["LEN_S_IDX"]
            s_horizon = map_vel_x * t_horizon
            idx_horizon = s_horizon / s_length
            horizon_indices = _calculate_horizon_indices(
                current_idx, idx_horizon, len(waypoints_xy_world), n_horizon, env.device
            )
        else:
            lookahead_steps = torch.linspace(0, delta_s_idx * n_horizon, n_horizon + 1, device=env.device)
            horizon_indices = (current_idx.unsqueeze(-1) + lookahead_steps.unsqueeze(0)) % len(waypoints_xy_world)
            horizon_indices = horizon_indices.long()

        # NOTE: original code indexed [:, :1]; we keep it as-is to preserve behavior.
        next_kappa_radpm_horizon = kappa_radpm[horizon_indices[:, :1]].squeeze(-1)
        kappa_results[env_mask] = next_kappa_radpm_horizon  # normalization left at 1
    return kappa_results


def delta_psi_rad_horizon(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    delta_s_idx: int = 10,
    n_horizon: int = 5,
    t_horizon: int = 5,
) -> torch.Tensor:
    """Heading change (Δψ) between consecutive future horizon points.

    Returns:
        [num_envs, n_horizon]
    """
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
    vel_x = mdp.base_lin_vel(env)[..., 0]
    num_envs = pos_xy_world.shape[0]

    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_waypoints_list"):
        env._waypoints_list = [torch.tensor(wps, device=env.device, dtype=torch.float32)
                               for wps in env.scene.terrain.cfg.waypoints_list]
    if not hasattr(env, "_inner_list"):
        env._inner_list = [torch.tensor(inner, device=env.device, dtype=torch.float32)
                           for inner in env.scene.terrain.cfg.inner_list]
    if not hasattr(env, "_psi_rad_list"):
        env._psi_rad_list = [torch.tensor(psi, device=env.device, dtype=torch.float32)
                             for psi in env.scene.terrain.cfg.psi_rad_list]

    delta_psi_results = torch.zeros(num_envs, n_horizon, device=env.device)
    map_levels = env._map_levels
    for map_level in torch.unique(map_levels):
        env_mask = map_levels == map_level
        if not env_mask.any():
            continue

        map_positions = pos_xy_world[env_mask]
        map_vel_x = vel_x[env_mask]
        waypoints_xy_world = env._waypoints_list[map_level][:, :2]
        inner_xy_world = env._inner_list[map_level][:, :2]
        psi_rad = env._psi_rad_list[map_level][:]

        current_idx, _ = find_frenet_coord_along_waypoints(inner_xy_world, map_positions)

        if CONFIG["env_config"]["DYNAMIC_LOOKAHEAD"]:
            s_length = CONFIG["env_config"]["LEN_S_IDX"]
            s_horizon = map_vel_x * t_horizon
            idx_horizon = s_horizon / s_length
            horizon_indices = _calculate_horizon_indices(
                current_idx, idx_horizon, len(waypoints_xy_world), n_horizon, env.device
            )
        else:
            lookahead_steps = torch.linspace(0, delta_s_idx * n_horizon, n_horizon + 1, device=env.device)
            horizon_indices = (current_idx.unsqueeze(-1) + lookahead_steps.unsqueeze(0)) % len(waypoints_xy_world)
            horizon_indices = horizon_indices.long()

        # Keep original behavior (explicit recomputation + consecutive difference)
        lookahead_steps = torch.linspace(delta_s_idx, delta_s_idx * n_horizon, n_horizon, device=env.device)
        horizon_indices = (current_idx.unsqueeze(-1) + lookahead_steps.unsqueeze(0)) % len(waypoints_xy_world)
        horizon_indices = horizon_indices.long()

        delta_psi = (psi_rad[horizon_indices[:, 1:]] - psi_rad[horizon_indices[:, :-1]]).squeeze(-1)
        delta_psi_results[env_mask] = delta_psi  # normalization left at 1
    return delta_psi_results


# --------------------------------------------------------------------------
# Ego–opponent relative features
# --------------------------------------------------------------------------

def opponent_relative_info_history(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    opponent_cfg: SceneEntityCfg = SceneEntityCfg("opponent"),
    position_std_noise: float = 0.0,
    velocity_std_noise: float = 0.0,
) -> torch.Tensor:
    """Relative kinematics between ego and opponent, with rolling histories.

    Features tracked per step (and returned as concatenated histories):
      - s index difference (meters along centerline, signed)
      - d difference (meters lateral)
      - v_x difference (m/s)
      - signed cross-positioning (left/right of ego heading to opponent)
      - opponent d (absolute lateral offset)

    Returns:
        [num_envs, history_len * 5]
    """
    ego_pos_xy_world = mdp.root_pos_w(env=env, asset_cfg=asset_cfg)[..., :2]
    ego_heading_w = env.scene[asset_cfg.name].data.heading_w
    ego_vel_x = mdp.base_lin_vel(env=env, asset_cfg=asset_cfg)[..., 0]

    opp_pos_xy_world = mdp.root_pos_w(env=env, asset_cfg=opponent_cfg)[..., :2]

    if position_std_noise > 0.0:
        opp_pos_xy_world = opp_pos_xy_world + torch.normal(
            mean=0.0, std=position_std_noise, size=opp_pos_xy_world.shape, device=opp_pos_xy_world.device
        )

    # Histories & caches
    if not hasattr(env, "_s_idx_diff_history"):
        env._s_idx_diff_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
    if not hasattr(env, "_d_diff_history"):
        env._d_diff_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
    if not hasattr(env, "_vx_diff_history"):
        env._vx_diff_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
    if not hasattr(env, "_heading_diff_history"):
        env._heading_diff_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
    if not hasattr(env, "_cross_pos_history"):
        env._cross_pos_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
    if not hasattr(env, "_opponent_d_history"):
        env._opponent_d_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
    if not hasattr(env, "_opponent_speed"):
        env._opponent_speed = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "_opponent_d_dot"):
        env._opponent_d_dot = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "_opponent_heading"):
        env._opponent_heading = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "_opponent_vel_scaling_lvl"):
        env._opponent_vel_scaling_lvl = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_waypoints_list"):
        env._waypoints_list = [
            torch.tensor(wps, device=env.device, dtype=torch.float32) for wps in env.scene.terrain.cfg.waypoints_list
        ]

    # Use opponent speed/heading tracked by events (optionally noisy)
    opp_vel_x = env._opponent_speed
    if velocity_std_noise > 0.0:
        opp_vel_x = opp_vel_x + torch.normal(mean=0.0, std=velocity_std_noise, size=opp_vel_x.shape, device=opp_vel_x.device)
    opp_heading = env._opponent_heading

    map_levels = env._map_levels
    unique_map_levels = torch.unique(map_levels)

    num_envs = ego_pos_xy_world.shape[0]
    s_idx_diff_opp_ego = torch.ones(num_envs, device=env.device, dtype=torch.float32) \
        * CONFIG["env_config"]["OPPONENT_INIT_DISTANCE_IDX_MAX"] * CONFIG["env_config"]["LEN_S_IDX"]
    t_diff_opp_ego = torch.zeros(num_envs, device=env.device, dtype=torch.float32)
    time_to_collision = torch.zeros(num_envs, device=env.device, dtype=torch.float32)

    cross_pos_opp_ego = torch.zeros(num_envs, device=env.device, dtype=torch.float32)
    d_diff_opp_ego = torch.zeros(num_envs, device=env.device, dtype=torch.float32)

    vx_diff_opp_ego = torch.zeros(num_envs, device=env.device, dtype=torch.float32)
    opp_ego_heading_diff = torch.zeros(num_envs, device=env.device, dtype=torch.float32)

    opp_d = torch.zeros(num_envs, device=env.device, dtype=torch.float32)

    for map_level in unique_map_levels:
        env_mask = map_levels == map_level
        if not env_mask.any():
            continue

        ego_map_positions = ego_pos_xy_world[env_mask]
        ego_map_headings = ego_heading_w[env_mask]
        ego_map_vel_x = ego_vel_x[env_mask]
        opp_map_positions = opp_pos_xy_world[env_mask]
        opp_map_vel_x = opp_vel_x[env_mask]
        opp_map_headings = opp_heading[env_mask]

        waypoints_xy_world = env._waypoints_list[map_level][:, :2]

        # Frenet: indices (s) and lateral offsets (d)
        num_waypoints = len(waypoints_xy_world)
        ego_s_idx, ego_d = find_frenet_coord_along_waypoints(waypoints_xy_world, ego_map_positions)
        opp_s_idx, opp_d_map = find_frenet_coord_along_waypoints(waypoints_xy_world, opp_map_positions)

        opp_d[env_mask] = opp_d_map
        d_diff_opp_ego[env_mask] = opp_d_map - ego_d

        # Signed shortest difference on a circular track
        s_idx_diff_raw = opp_s_idx - ego_s_idx
        s_idx_diff_signed = (s_idx_diff_raw + num_waypoints // 2) % num_waypoints - (num_waypoints // 2)
        s_idx_diff_opp_ego[env_mask] = s_idx_diff_signed * CONFIG["env_config"]["LEN_S_IDX"]

        # L/R positioning of opponent with respect to ego heading (cross product sign)
        ego_heading_vec = torch.stack([torch.cos(ego_map_headings), torch.sin(ego_map_headings)], dim=1)
        dist = torch.norm(ego_map_positions - opp_map_positions, dim=1)
        ego_to_opp = (opp_map_positions - ego_map_positions)
        cross_pos = (ego_heading_vec[:, 0] * ego_to_opp[:, 1] - ego_heading_vec[:, 1] * ego_to_opp[:, 0]) / (dist + 1e-1)
        cross_pos_opp_ego[env_mask] = torch.clamp(cross_pos, -1.0, 1.0)

        # Relative longitudinal velocity and timing
        vx_diff_opp_ego[env_mask] = opp_map_vel_x - ego_map_vel_x
        t_diff_opp_ego[env_mask] = torch.clamp(s_idx_diff_opp_ego[env_mask] / (vx_diff_opp_ego[env_mask] + 1e-3), -10.0, 10.0)
        time_to_collision[env_mask] = torch.clamp(s_idx_diff_opp_ego[env_mask] / (-vx_diff_opp_ego[env_mask] + 1e-3), 0, 10)

        # Heading difference (not currently added to output history to preserve behavior)
        opp_ego_heading_diff[env_mask] = torch.atan2(
            torch.sin(opp_map_headings - ego_map_headings),
            torch.cos(opp_map_headings - ego_map_headings),
        )

    # Push to histories (shift right, write newest at index 0)
    env._s_idx_diff_history[:, 1:] = env._s_idx_diff_history[:, :-1].clone()
    env._s_idx_diff_history[:, 0] = s_idx_diff_opp_ego

    env._d_diff_history[:, 1:] = env._d_diff_history[:, :-1].clone()
    env._d_diff_history[:, 0] = d_diff_opp_ego

    env._vx_diff_history[:, 1:] = env._vx_diff_history[:, :-1].clone()
    env._vx_diff_history[:, 0] = vx_diff_opp_ego

    env._cross_pos_history[:, 1:] = env._cross_pos_history[:, :-1].clone()
    env._cross_pos_history[:, 0] = cross_pos_opp_ego

    env._opponent_d_history[:, 1:] = env._opponent_d_history[:, :-1].clone()
    env._opponent_d_history[:, 0] = opp_d

    # Return concatenated histories as one flat observation
    return torch.cat(
        [
            env._s_idx_diff_history.reshape(num_envs, -1),
            env._d_diff_history.reshape(num_envs, -1),
            env._vx_diff_history.reshape(num_envs, -1),
            env._cross_pos_history.reshape(num_envs, -1),
            env._opponent_d_history.reshape(num_envs, -1),
        ],
        dim=1,
    )


def gaps_info_history(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    opponent_cfg: SceneEntityCfg = SceneEntityCfg("opponent"),
) -> torch.Tensor:
    """Distances from opponent to inner/outer bounds along the track, as histories.

    Returns:
        [num_envs, history_len * 2]  (inner gap history | outer gap history)
    """
    ego_pos_xy_world = mdp.root_pos_w(env=env, asset_cfg=asset_cfg)[..., :2]  # not used directly, kept for parity
    opp_pos_xy_world = mdp.root_pos_w(env=env, asset_cfg=opponent_cfg)[..., :2]

    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_waypoints_list"):
        env._waypoints_list = [
            torch.tensor(wps, device=env.device, dtype=torch.float32) for wps in env.scene.terrain.cfg.waypoints_list
        ]
    if not hasattr(env, "_outer_list"):
        env._outer_list = [
            torch.tensor(outer, device=env.device, dtype=torch.float32) for outer in env.scene.terrain.cfg.outer_list
        ]
    if not hasattr(env, "_inner_list"):
        env._inner_list = [
            torch.tensor(inner, device=env.device, dtype=torch.float32) for inner in env.scene.terrain.cfg.inner_list
        ]
    if not hasattr(env, "_gap_inner_history"):
        env._gap_inner_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
    if not hasattr(env, "_gap_outer_history"):
        env._gap_outer_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)

    map_levels = env._map_levels
    unique_map_levels = torch.unique(map_levels)

    num_envs = opp_pos_xy_world.shape[0]
    gap_inner = torch.zeros(num_envs, device=env.device, dtype=torch.float32)
    gap_outer = torch.zeros(num_envs, device=env.device, dtype=torch.float32)

    for map_level in unique_map_levels:
        env_mask = map_levels == map_level
        if not env_mask.any():
            continue

        opp_map_positions = opp_pos_xy_world[env_mask]
        inner_xy_world = env._inner_list[map_level][:, :2]
        outer_xy_world = env._outer_list[map_level][:, :2]

        # Frenet distance to inner/outer boundaries (absolute)
        _, inner_dist = find_frenet_coord_along_waypoints(inner_xy_world, opp_map_positions)
        _, outer_dist = find_frenet_coord_along_waypoints(outer_xy_world, opp_map_positions)
        gap_inner[env_mask] = torch.abs(inner_dist)
        gap_outer[env_mask] = torch.abs(outer_dist)

    env._gap_inner_history[:, 1:] = env._gap_inner_history[:, :-1].clone()
    env._gap_inner_history[:, 0] = gap_inner

    env._gap_outer_history[:, 1:] = env._gap_outer_history[:, :-1].clone()
    env._gap_outer_history[:, 0] = gap_outer

    return torch.cat(
        [env._gap_inner_history.reshape(num_envs, -1), env._gap_outer_history.reshape(num_envs, -1)], dim=1
    )


def opponent_frenet_coordinates_history(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    opponent_cfg: SceneEntityCfg = SceneEntityCfg("opponent"),
) -> torch.Tensor:
    """Opponent progress and lateral offset (s in [0,1] normalized, d in meters), as histories.

    Returns:
        [num_envs, history_len * 2]  (s_history | d_history)
    """
    ego_pos_xy_world = mdp.root_pos_w(env=env, asset_cfg=asset_cfg)[..., :2]  # unused but parallel to other fns
    opp_pos_xy_world = mdp.root_pos_w(env=env, asset_cfg=opponent_cfg)[..., :2]

    if not hasattr(env, "_opponent_s_history"):
        env._opponent_s_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
    if not hasattr(env, "_opponent_d_history"):
        env._opponent_d_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_waypoints_list"):
        env._waypoints_list = [
            torch.tensor(wps, device=env.device, dtype=torch.float32) for wps in env.scene.terrain.cfg.waypoints_list
        ]

    map_levels = env._map_levels
    unique_map_levels = torch.unique(map_levels)

    num_envs = opp_pos_xy_world.shape[0]
    opp_s = torch.zeros(num_envs, device=env.device, dtype=torch.float32)
    opp_d = torch.zeros(num_envs, device=env.device, dtype=torch.float32)

    for map_level in unique_map_levels:
        env_mask = map_levels == map_level
        if not env_mask.any():
            continue

        opp_map_positions = opp_pos_xy_world[env_mask]
        waypoints_xy_world = env._waypoints_list[map_level][:, :2]
        num_waypoints = len(waypoints_xy_world)

        opp_s_idx, opp_d_map = find_frenet_coord_along_waypoints(waypoints_xy_world, opp_map_positions)
        opp_s[env_mask] = (opp_s_idx + 1) / num_waypoints  # normalized progress
        opp_d[env_mask] = opp_d_map

    env._opponent_s_history[:, 1:] = env._opponent_s_history[:, :-1].clone()
    env._opponent_s_history[:, 0] = opp_s

    env._opponent_d_history[:, 1:] = env._opponent_d_history[:, :-1].clone()
    env._opponent_d_history[:, 0] = opp_d

    return torch.cat([env._opponent_s_history.reshape(num_envs, -1), env._opponent_d_history.reshape(num_envs, -1)], dim=1)
