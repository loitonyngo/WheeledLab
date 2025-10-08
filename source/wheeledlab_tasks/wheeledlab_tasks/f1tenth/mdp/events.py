"""events.py

Reset utilities and runtime opponent motion logic for the F1TENTH tasks.

This module groups the code that:
  1) Initializes per-env histories/buffers used by observations, rewards,
     terminations, and curriculum logic.
  2) Resets the ego (robot) and, for the overtake task, the opponent actor.
  3) Advances the opponent along precomputed s-trajectories at runtime.

Design notes
------------
- **Non-invasive clean-up**: The logic is left intact to avoid regressions.
  Changes are limited to comments, naming clarifications, and small efficiency
  touches (e.g., use of `torch.as_tensor` where appropriate) that do not alter
  behavior.
- **Idempotent init**: `_init_common_histories` and `_init_opponent_histories`
  only create buffers if they are missing. This lets the code be called from
  different reset entry points without duplication or errors.
- **Shape conventions**: Histories follow `(num_envs, history_len[, ...])`.
- **Frames**: Positions from `mdp.root_pos_w` are in *world* frame; most
  track logic operates in the *track* frame (waypoint space). Be mindful when
  adding new terms.

Config
------
`CONFIG = load_config()` is assumed to expose an `env_config` mapping with all
parameters used below (history lengths, curriculum thresholds, opponent trajs,
move cadence, etc.).
"""

from __future__ import annotations

import numpy as np
import torch
import isaaclab.utils.math as math_utils

from typing import Sequence

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.terrains import TerrainImporter
from ..utils import find_frenet_coord_along_waypoints
import isaaclab.envs.mdp as mdp

from wheeledlab_tasks.config_loader import load_config

CONFIG = load_config()
ENV = CONFIG["env_config"]  # shorthand used throughout

# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _init_common_histories(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Initialize histories/buffers that are common across reset functions.

    Idempotent: it only creates attributes that don't exist yet, so calling it
    multiple times is safe.
    """
    # Map selection for each parallel env (which racetrack/map this env uses).
    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # Waypoint index at reset (used to compute progress relative to spawn).
    if not hasattr(env, "_initial_waypoint_indices"):
        env._initial_waypoint_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # Running sum of progress in waypoint indices (signed) for curriculum/rewards.
    if not hasattr(env, "_total_progress_indices"):
        env._total_progress_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # Flag to mark that an environment has just been reset (used once after reset).
    if not hasattr(env, "_reset_env_bool"):
        env._reset_env_bool = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    # Reward/termination progress history (circular buffer semantics are handled elsewhere).
    if not hasattr(env, "_progress_history_indices"):
        env._rew_history_length = ENV["REW_HISTORY_LENGTH"]
        env._progress_history_indices = torch.zeros(
            (env.num_envs, env._rew_history_length), dtype=torch.long, device=env.device
        )

    # Action/observation history lengths (constants for buffer shapes).
    if not hasattr(env, "_action_history_length"):
        env._action_history_length = ENV["ACTION_HISTORY_LENGTH"]
    if not hasattr(env, "_obs_history_length"):
        env._obs_history_length = ENV["OBS_HISTORY_LENGTH"]

    # Store the last N actions: shape (num_envs, N, 2) for [throttle, steering].
    if not hasattr(env, "_action_history"):
        env._action_history = torch.zeros(
            (env.num_envs, env._action_history_length, 2), dtype=torch.float32, device=env.device
        )

    # Observation histories commonly used by observation terms.
    for name in [
        "_base_lin_vel_x_history",
        "_base_lin_vel_y_history",
        "_base_ang_vel_z_history",
        "_target_velocity_history",
        "_target_steering_angle_history",
    ]:
        if not hasattr(env, name):
            setattr(
                env,
                name,
                torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device),
            )

    # Collision history with walls (used by rewards/terminations/curriculum).
    if not hasattr(env, "_wall_collision_history"):
        env._rew_history_length = ENV["REW_HISTORY_LENGTH"]
        env._wall_collision_history = torch.ones(
            (env.num_envs, env._rew_history_length), dtype=torch.long, device=env.device
        )


# ---------------------------------------------------------------------------
# Reset helpers (ego/opponent)
# ---------------------------------------------------------------------------

def _reset_ego_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset: RigidObject | Articulation,
    valid_poses: Sequence,
    current_idx: torch.Tensor,
) -> None:
    """Reset the ego (robot) state for a subset of envs.

    Parameters
    ----------
    env_ids: indices of the parallel envs to reset
    asset: ego asset handle (robot)
    valid_poses: list of InitialPoseCfg-like objects with `.pos`, `.rot_euler_xyz_deg`,
                 `.lin_vel`, `.ang_vel`
    current_idx: waypoint indices associated with the sampled poses (track frame)
    """
    # Compose pose tensors (track frame) and velocities.
    posns = torch.stack([torch.as_tensor(x.pos, device=env.device) for x in valid_poses]).float()
    oris = torch.stack(
        [
            math_utils.quat_from_euler_xyz(*torch.deg2rad(torch.as_tensor(x.rot_euler_xyz_deg, device=env.device)))
            for x in valid_poses
        ]
    ).float()
    lin_vels = torch.stack([torch.as_tensor(x.lin_vel, device=env.device) for x in valid_poses]).float()
    ang_vels = torch.stack([torch.as_tensor(x.ang_vel, device=env.device) for x in valid_poses]).float()

    # Shift by asset default root state (world frame offsets per env).
    positions = posns + asset.data.default_root_state[env_ids, :3]
    orientations = oris
    lin_vels = lin_vels + asset.data.default_root_state[env_ids, 7:10]
    ang_vels = ang_vels + asset.data.default_root_state[env_ids, 10:13]

    # Write pose and velocity to the simulator.
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.cat([lin_vels, ang_vels], dim=-1), env_ids=env_ids)

    # Reset per-env histories for these env_ids.
    env._initial_waypoint_indices[env_ids] = current_idx.clone()
    env._reset_env_bool[env_ids] = torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
    # Make cumulative progress zero at spawn by subtracting the spawn index.
    env._total_progress_indices[env_ids] = -current_idx.clone()

    env._progress_history_indices[env_ids, :] = torch.zeros(
        env._rew_history_length, dtype=torch.long, device=env.device
    )
    env._action_history[env_ids, :, :] = torch.zeros(
        (len(env_ids), env._action_history_length, 2), dtype=torch.float32, device=env.device
    )
    env._base_lin_vel_x_history[env_ids, :] = torch.zeros(
        (len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device
    )
    env._base_lin_vel_y_history[env_ids, :] = torch.zeros(
        (len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device
    )
    env._base_ang_vel_z_history[env_ids, :] = torch.zeros(
        (len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device
    )
    env._target_velocity_history[env_ids, :] = torch.zeros(
        (len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device
    )
    env._target_steering_angle_history[env_ids, :] = torch.zeros(
        (len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device
    )
    env._wall_collision_history[env_ids, :] = torch.zeros(
        (len(env_ids), env._rew_history_length), dtype=torch.long, device=env.device
    )


def _reset_opponent_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    opponent_asset: RigidObject | Articulation,
    opp_valid_poses: Sequence,
) -> None:
    """Reset the opponent's pose and velocity for a subset of envs."""
    opp_posns = torch.stack([torch.as_tensor(x.pos, device=env.device) for x in opp_valid_poses]).float()
    opp_oris = torch.stack(
        [
            math_utils.quat_from_euler_xyz(*torch.deg2rad(torch.as_tensor(x.rot_euler_xyz_deg, device=env.device)))
            for x in opp_valid_poses
        ]
    ).float()
    # Opponent starts with given lin/ang vel; current logic zeroes lin_vels.
    opp_lin_vels = torch.stack([torch.as_tensor(x.lin_vel, device=env.device) for x in opp_valid_poses]).float() * 0
    opp_ang_vels = torch.stack([torch.as_tensor(x.ang_vel, device=env.device) for x in opp_valid_poses]).float()

    opp_positions = opp_posns + opponent_asset.data.default_root_state[env_ids, :3]
    opp_orientations = opp_oris
    opp_lin_vels = opp_lin_vels + opponent_asset.data.default_root_state[env_ids, 7:10]
    opp_ang_vels = opp_ang_vels + opponent_asset.data.default_root_state[env_ids, 10:13]

    opponent_asset.write_root_pose_to_sim(torch.cat([opp_positions, opp_orientations], dim=-1), env_ids=env_ids)
    opponent_asset.write_root_velocity_to_sim(torch.cat([opp_lin_vels, opp_ang_vels], dim=-1), env_ids=env_ids)


def _init_opponent_histories(
    env: ManagerBasedEnv, env_ids: torch.Tensor, ego_current_idx: torch.Tensor, opp_current_idx: torch.Tensor
) -> None:
    """Initialize and reset opponent-specific histories.

    This covers trajectory parameters, per-env scalings, counters, and
    buffers that feed observation/reward terms for the overtake task.
    """
    # Distance between opponent and ego in s-index space at the previous step
    # (initialized to a large separation to avoid spurious signals on first tick).
    if not hasattr(env, "_prev_delta_s_opp_ego"):
        env._prev_delta_s_opp_ego = torch.ones(env.num_envs, dtype=torch.float32, device=env.device) \
            * ENV["OPPONENT_INIT_DISTANCE_IDX_MAX"] * ENV["LEN_S_IDX"]

    # Traversability (track) history (1: inside, 0: outside), used by terms.
    if not hasattr(env, "_traversability_history"):
        env._rew_history_length = ENV["REW_HISTORY_LENGTH"]
        env._traversability_history = torch.ones(
            (env.num_envs, env._rew_history_length), dtype=torch.long, device=env.device
        )

    # Dynamic opponent properties (types, speed gains, trajectory blend, etc.).
    if not hasattr(env, "_opponent_type"):
        env._opponent_type = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_opponent_vel_scaling"):
        env._opponent_vel_scaling = torch.ones(env.num_envs, dtype=torch.float16, device=env.device)
    if not hasattr(env, "_opponent_trajectory_alpha"):
        env._opponent_trajectory_alpha = torch.ones(env.num_envs, dtype=torch.float16, device=env.device)
    if not hasattr(env, "_opponent_overtaken_bool"):
        env._opponent_overtaken_bool = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_opponent_speed"):
        env._opponent_speed = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "_opponent_d_dot"):
        env._opponent_d_dot = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "_opponent_heading"):
        env._opponent_heading = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "_opponent_always_ahead"):
        env._opponent_always_ahead = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Event histories used in curriculum tuning and diagnostics.
    if not hasattr(env, "_opponent_collision_history"):
        env._rew_history_length = ENV["REW_HISTORY_LENGTH"]
        env._opponent_collision_history = torch.zeros(
            (env.num_envs, env._rew_history_length), dtype=torch.long, device=env.device
        )

    # Relative-state histories (delays for obs features and dense rewards).
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
    if not hasattr(env, "_gap_inner_history"):
        env._gap_inner_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
    if not hasattr(env, "_gap_outer_history"):
        env._gap_outer_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)

    if not hasattr(env, "_opponent_overtaken_history"):
        env._rew_history_length = ENV["REW_HISTORY_LENGTH"]
        env._opponent_overtaken_history = torch.zeros(
            (env.num_envs, env._rew_history_length), dtype=torch.long, device=env.device
        )

    # Global curriculum counters and initial speed scaling level.
    if not hasattr(env, "_opponent_overtaken_counter"):
        env._opponent_overtaken_counter = 0
        env._opponent_collision_counter = 0
        env._wall_collision_counter = 0
        env._opponent_vel_scaling_lvl = ENV["OPP_INIT_VEL_SCALING"]

    # Opponent s/d histories for observation features.
    if not hasattr(env, "_opponent_s_history"):
        env._opponent_s_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
    if not hasattr(env, "_opponent_d_history"):
        env._opponent_d_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)

    # Per-env lateral XY shift for the opponent trajectory (randomized at reset).
    if not hasattr(env, "_opponent_traj_x_shift") or not hasattr(env, "_opponent_traj_y_shift"):
        env._opponent_traj_x_shift = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        env._opponent_traj_y_shift = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # --- Reset values for the just-reset env_ids ---
    env._prev_delta_s_opp_ego[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=env.device)

    # Randomly assign opponent type per env in {0: center, 1: IQP, 2: SP}.
    env._opponent_type[env_ids] = torch.as_tensor(
        np.floor(np.random.rand(len(env_ids)) * 3), device=env.device, dtype=torch.long
    )

    # Per-env random scaling around the current curriculum level.
    env._opponent_vel_scaling[env_ids] = torch.normal(
        mean=torch.ones(len(env_ids), dtype=torch.float16, device=env.device) * env._opponent_vel_scaling_lvl,
        std=ENV["OPP_STD_VEL_SCALING"],
    ).clamp(min=ENV["OPP_MIN_VEL_SCALING"], max=env._opponent_vel_scaling_lvl)

    # Trajectory blend parameter (only used when OPP_GENERALIZATION is enabled).
    if ENV["OPP_GENERALIZATION"]:
        env._opponent_trajectory_alpha[env_ids] = torch.as_tensor(
            np.random.rand(len(env_ids)), device=env.device, dtype=torch.float16
        )
    else:
        env._opponent_trajectory_alpha[env_ids] = torch.ones(len(env_ids), dtype=torch.float16, device=env.device)

    # Reset per-env flags/aux values.
    env._opponent_overtaken_bool[env_ids] = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    env._opponent_speed[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=env.device)
    env._opponent_d_dot[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=env.device)
    env._opponent_heading[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=env.device)

    # Some envs keep the opponent always ahead by design (curriculum knob).
    env._opponent_always_ahead[env_ids] = (
        torch.rand(len(env_ids), device=env.device) < ENV["OPPONENT_ALWAYS_AHEAD_PERCENTAGE"]
    )

    # Clear short histories for just-reset envs.
    env._opponent_collision_history[env_ids, :] = torch.zeros(
        (len(env_ids), env._rew_history_length), dtype=torch.long, device=env.device
    )
    env._opponent_overtaken_history[env_ids, :] = torch.zeros(
        (len(env_ids), env._rew_history_length), dtype=torch.long, device=env.device
    )
    env._s_idx_diff_history[env_ids, :] = torch.zeros(
        (len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device
    )
    env._d_diff_history[env_ids, :] = torch.zeros(
        (len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device
    )
    env._vx_diff_history[env_ids, :] = torch.zeros(
        (len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device
    )
    env._heading_diff_history[env_ids, :] = torch.zeros(
        (len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device
    )
    env._cross_pos_history[env_ids, :] = torch.zeros(
        (len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device
    )
    env._gap_inner_history[env_ids, :] = torch.zeros(
        (len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device
    )
    env._gap_outer_history[env_ids, :] = torch.zeros(
        (len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device
    )
    env._opponent_s_history[env_ids, :] = torch.zeros(
        (len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device
    )
    env._opponent_d_history[env_ids, :] = torch.zeros(
        (len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device
    )

    # Random lateral shifts of the opponent trajectory in track frame.
    shift_rng = ENV["OPPONENT_TRAJ_XY_SHIFT_MAX"]
    env._opponent_traj_x_shift[env_ids] = torch.rand(len(env_ids), device=env.device) * shift_rng * 2 - shift_rng
    env._opponent_traj_y_shift[env_ids] = torch.rand(len(env_ids), device=env.device) * shift_rng * 2 - shift_rng


# ---------------------------------------------------------------------------
# Public reset functions
# ---------------------------------------------------------------------------

def reset_root_state_random(
    env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    """Reset the ego at a random valid pose sampled from the track waypoints."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    _init_common_histories(env, env_ids)

    # Randomly assign a map level per env (uniform over available waypoint sets).
    env._map_levels[env_ids] = torch.as_tensor(
        np.floor(np.random.rand(len(env_ids)) * len(env.cfg.scene.terrain.waypoints_list)),
        device=env.device,
        dtype=torch.long,
    )

    valid_poses, current_idx_np = terrain.cfg.generate_random_poses_from_waypoints(
        env=env, env_ids=env_ids, num_poses=len(env_ids), max_radius_offset=0.5
    )
    current_idx = torch.as_tensor(current_idx_np, dtype=torch.long, device=env.device)

    _reset_ego_state(env, env_ids, asset, valid_poses, current_idx)


def reset_root_state_random_with_opponent(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    opponent_asset_cfg: SceneEntityCfg = SceneEntityCfg("opponent"),
) -> None:
    """Reset ego and opponent with consistent random poses for overtake task."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    opponent_asset: RigidObject | Articulation = env.scene[opponent_asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    _init_common_histories(env, env_ids)

    # Retain original behavior: map levels based on traversability list length.
    env._map_levels[env_ids] = torch.as_tensor(
        np.floor(np.random.rand(len(env_ids)) * len(env.cfg.scene.terrain.traversability_hashmap_list)),
        device=env.device,
        dtype=torch.long,
    )

    ego_valid_poses, ego_idx_np, opp_valid_poses, opp_idx_np = terrain.cfg.generate_random_poses_from_waypoints_with_opponent(
        env=env, env_ids=env_ids, num_poses=len(env_ids), max_radius_offset=0.5
    )
    current_idx = torch.as_tensor(ego_idx_np, dtype=torch.long, device=env.device)
    opp_current_idx = torch.as_tensor(opp_idx_np, dtype=torch.long, device=env.device)

    _reset_ego_state(env, env_ids, asset, ego_valid_poses, current_idx)
    _reset_opponent_state(env, env_ids, opponent_asset, opp_valid_poses)
    _init_opponent_histories(env, env_ids, current_idx, opp_current_idx)


def reset_root_state_start_idx(
    env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    """Reset the ego agent at specific start indices (deterministic)."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # 1) Ensure histories exist
    _init_common_histories(env, env_ids)

    # 2) Sample poses from start indices
    valid_poses, current_idx_np = terrain.cfg.generate_start_idx_poses(env=env, env_ids=env_ids, num_poses=len(env_ids))
    current_idx = torch.as_tensor(current_idx_np, dtype=torch.long, device=env.device)

    # 3) Reset ego (robot) state
    _reset_ego_state(env, env_ids, asset, valid_poses, current_idx)

    # Progress accumulator starts from zero after a deterministic reset
    env._total_progress_indices[env_ids] = torch.zeros(len(env_ids), dtype=torch.long, device=env.device)


# ---------------------------------------------------------------------------
# Opponent motion (s-trajectory based)
# ---------------------------------------------------------------------------

def move_opponent_s_based(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("opponent"),
) -> None:
    """Advance the opponent along a precomputed s-trajectory.

    Behavior summary:
    - Keep the opponent static for the first `STATIC_OPPONENT_UNTIL_EP` episodes.
    - Periodically adjust the *global* opponent speed level
      (`env._opponent_vel_scaling_lvl`) using recent success/collision stats.
    - For each map level present in `env_ids`, move the opponent along the
      chosen trajectory (centerline / min-curvature / speed-profile).
    - If `OPP_GENERALIZATION` is on, the trajectory type is per-env; otherwise
      a single global traj from CONFIG is used.
    - When `env._opponent_always_ahead` is true, the opponent stays a fixed
      s-index offset ahead during the early phase of the episode.
    """
    # Early exit during curriculum warm-up (opponent stays static)
    num_episodes = env.common_step_counter // env.max_episode_length
    if num_episodes < ENV["STATIC_OPPONENT_UNTIL_EP"]:
        return None

    # Ensure attributes/buffers exist (compatible with reset helpers)
    if not hasattr(env, "_opponent_type"):
        env._opponent_type = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_opponent_vel_scaling"):
        env._opponent_vel_scaling = torch.ones(env.num_envs, dtype=torch.float16, device=env.device)
    if not hasattr(env, "_opponent_trajectory_alpha"):
        env._opponent_trajectory_alpha = torch.ones(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # Cache per-map trajectories on the env as tensors (built once)
    if not hasattr(env, "_opp_traj_center_list"):
        env._opp_traj_center_list = [
            torch.as_tensor(center, device=env.device, dtype=torch.float32)
            for center in env.scene.terrain.cfg.opp_traj_center_list
        ]
    if not hasattr(env, "_opp_traj_iqp_list"):
        env._opp_traj_iqp_list = [
            torch.as_tensor(iqp, device=env.device, dtype=torch.float32)
            for iqp in env.scene.terrain.cfg.opp_traj_iqp_list
        ]
    if not hasattr(env, "_opp_traj_sp_list"):
        env._opp_traj_sp_list = [
            torch.as_tensor(sp, device=env.device, dtype=torch.float32)
            for sp in env.scene.terrain.cfg.opp_traj_sp_list
        ]

    # Histories required for curriculum adjustments
    if not hasattr(env, "_opponent_overtaken_history"):
        env._rew_history_length = ENV["REW_HISTORY_LENGTH"]
        env._opponent_overtaken_history = torch.zeros(
            (env.num_envs, env._rew_history_length), dtype=torch.long, device=env.device
        )
    if not hasattr(env, "_opponent_collision_history"):
        env._rew_history_length = ENV["REW_HISTORY_LENGTH"]
        env._opponent_collision_history = torch.zeros(
            (env.num_envs, env._rew_history_length), dtype=torch.long, device=env.device
        )
    if not hasattr(env, "_wall_collision_history"):
        env._rew_history_length = ENV["REW_HISTORY_LENGTH"]
        env._wall_collision_history = torch.zeros(
            (env.num_envs, env._rew_history_length), dtype=torch.long, device=env.device
        )

    # Curriculum: adjust global opponent speed at fixed cadence
    if env.common_step_counter % ENV["VEL_SCALING_CHECK_STEPS"] == 0:
        denom = env._opponent_collision_counter + env._opponent_overtaken_counter + env._wall_collision_counter + 1
        success_ratio = env._opponent_overtaken_counter / denom
        if success_ratio > ENV["OT_COLLISION_RATIO_LVL_UP"]:
            env._opponent_vel_scaling_lvl += ENV["OPP_VEL_SCALING_INCREMENT"]
        elif success_ratio < ENV["OT_COLLISION_RATIO_LVL_DOWN"]:
            env._opponent_vel_scaling_lvl -= ENV["OPP_VEL_SCALING_DECREASE"]
        # Reset counters regardless of branch
        env._opponent_overtaken_counter = 0
        env._opponent_collision_counter = 0
        env._wall_collision_counter = 0

    # Enforce minimum velocity scaling level
    env._opponent_vel_scaling_lvl = max(env._opponent_vel_scaling_lvl, ENV["OPP_MIN_VEL_SCALING"])

    # Optional periodic counter reset (legacy)
    if env.common_step_counter % 500 == 0:
        env._opponent_overtaken_counter = 0
        env._opponent_collision_counter = 0
        env._wall_collision_counter = 0

    # Current positions in world frame (XY only)
    asset = env.scene[asset_cfg.name]
    ego_position_xy = mdp.root_pos_w(env=env, asset_cfg=SceneEntityCfg("robot"))[:, :2]
    opp_position_xy = mdp.root_pos_w(env=env, asset_cfg=SceneEntityCfg("opponent"))[:, :2]

    # Output buffers (copy then modify): preserve Z/vels, update XY + yaw
    new_positions = asset.data.root_pos_w.clone()
    new_orientations = asset.data.root_quat_w.clone()
    # (lin/ang velocities are unused here but kept to preserve behavior)
    new_lin_velocities = torch.zeros_like(asset.data.root_lin_vel_w)
    new_ang_velocities = torch.zeros_like(asset.data.root_ang_vel_w)

    # Batch by map level for efficiency
    opponent_types = env._opponent_type[env_ids]
    map_levels = env._map_levels[env_ids]
    unique_map_levels = torch.unique(map_levels)

    for map_level in unique_map_levels:
        map_mask = map_levels == map_level
        if not map_mask.any():
            continue

        # (kept for clarity / debugging)
        _waypoints_xy_world = env._waypoints_list[map_level][:, :2]

        level_opp_types = opponent_types[map_mask]
        unique_opp_types = torch.unique(level_opp_types)

        if ENV["OPP_GENERALIZATION"]:
            # Each env within this map level chooses a trajectory family
            for opp_type in unique_opp_types:
                type_mask = level_opp_types == opp_type
                if not type_mask.any():
                    continue
                combined_mask = map_mask & (opponent_types == opp_type)

                # Pick trajectory tensor based on opponent type
                if int(opp_type) == 0:
                    traj = env._opp_traj_center_list[map_level]
                elif int(opp_type) == 1:
                    traj = env._opp_traj_iqp_list[map_level]
                elif int(opp_type) == 2:
                    traj = env._opp_traj_sp_list[map_level]
                else:
                    raise ValueError(f"Unknown opponent type: {int(opp_type)}")

                traj_xy, traj_psi, traj_vel = traj[:, :2], traj[:, 2], traj[:, 3]
                traj_xy_shift = torch.stack(
                    [env._opponent_traj_x_shift[combined_mask], env._opponent_traj_y_shift[combined_mask]], dim=1
                )

                # Closest indices on reference paths (ego/opponent)
                ego_idx, _ = find_frenet_coord_along_waypoints(traj_xy, ego_position_xy[combined_mask])
                opp_idx, _ = find_frenet_coord_along_waypoints(traj_xy, opp_position_xy[combined_mask] - traj_xy_shift)

                # Number of s-indices to move this tick (>= 1)
                move_steps = (
                    traj_vel[opp_idx] * ENV["OPP_MOVE_DT"] * (env._opponent_vel_scaling[combined_mask]) / ENV["LEN_S_IDX"]
                ).int()

                # Choose next waypoint index according to early-phase and always-ahead logic
                next_idx = torch.where(
                    env._opponent_always_ahead[combined_mask],
                    (ego_idx + ENV["OPPONENT_INIT_DISTANCE_IDX_MIN"]) % len(traj_xy),
                    torch.where(
                        env.episode_length_buf[combined_mask] < ENV["OPPONENT_STARTS_MOVING_AFTER_STEP"],
                        opp_idx,
                        (opp_idx + torch.where(move_steps > 0, move_steps, 1)) % len(traj_xy),
                    ),
                )

                # Candidate next pose in track frame
                pos = traj_xy[next_idx] + traj_xy_shift
                psi = traj_psi[next_idx]

                # Update world-frame pose (apply per-env world origin)
                new_positions[combined_mask, 0] = pos[:, 0] + env.scene.env_origins[combined_mask, 0]
                new_positions[combined_mask, 1] = pos[:, 1] + env.scene.env_origins[combined_mask, 1]

                # Cache track-frame XY for features/rewards
                env._opponent_xy_position[combined_mask, :] = pos.clone()

                # Instantaneous speed and heading estimates
                disp = pos - opp_position_xy[combined_mask]
                env._opponent_speed[combined_mask] = torch.norm(disp, dim=1) / ENV["OPP_MOVE_DT"]
                env._opponent_heading[combined_mask] = torch.atan2(disp[:, 1], disp[:, 0])

                # Yaw-only quaternion (w, x, y, z)
                new_orientations[combined_mask] = torch.stack(
                    [torch.cos(psi / 2.0), torch.zeros_like(psi), torch.zeros_like(psi), torch.sin(psi / 2.0)], dim=1
                )

        else:
            # Single global trajectory family from config
            if ENV["OPP_TRAJ"] == "mincurv":
                traj = env._opp_traj_iqp_list[map_level]
            elif ENV["OPP_TRAJ"] == "sp":
                traj = env._opp_traj_sp_list[map_level]
            elif ENV["OPP_TRAJ"] == "centerline":
                traj = env._opp_traj_center_list[map_level]
            else:
                traj = env._opp_traj_iqp_list[map_level]

            traj_xy, traj_psi, traj_vel = traj[:, :2], traj[:, 2], traj[:, 3]
            traj_xy_shift = torch.stack(
                [env._opponent_traj_x_shift[map_mask], env._opponent_traj_y_shift[map_mask]], dim=1
            )

            ego_idx, _ = find_frenet_coord_along_waypoints(traj_xy, ego_position_xy[map_mask])
            opp_idx, _ = find_frenet_coord_along_waypoints(traj_xy, opp_position_xy[map_mask] - traj_xy_shift)

            move_steps = (traj_vel[opp_idx] * ENV["OPP_MOVE_DT"] * (env._opponent_vel_scaling[map_mask]) / ENV["LEN_S_IDX"]).int()

            next_idx = torch.where(
                env._opponent_always_ahead[map_mask],
                (ego_idx + ENV["OPPONENT_INIT_DISTANCE_IDX_MIN"]) % len(traj_xy),
                torch.where(
                    env.episode_length_buf[map_mask] < ENV["OPPONENT_STARTS_MOVING_AFTER_STEP"],
                    opp_idx,
                    (opp_idx + torch.where(move_steps > 0, move_steps, 1)) % len(traj_xy),
                ),
            )

            pos = traj_xy[next_idx] + traj_xy_shift
            psi = traj_psi[next_idx]
            # vel = traj_vel[next_idx]  # (kept for potential debugging/telemetry)

            new_positions[map_mask, 0] = pos[:, 0] + env.scene.env_origins[map_mask, 0]
            new_positions[map_mask, 1] = pos[:, 1] + env.scene.env_origins[map_mask, 1]

            disp = pos - opp_position_xy[map_mask]
            env._opponent_speed[map_mask] = torch.norm(disp, dim=1) / ENV["OPP_MOVE_DT"]
            env._opponent_heading[map_mask] = torch.atan2(disp[:, 1], disp[:, 0])

            new_orientations[map_mask] = torch.stack(
                [torch.cos(psi / 2.0), torch.zeros_like(psi), torch.zeros_like(psi), torch.sin(psi / 2.0)], dim=1
            )

    # Apply updates in a single simulator call for all selected envs
    asset.write_root_pose_to_sim(torch.cat([new_positions, new_orientations], dim=1), env_ids=env_ids)
