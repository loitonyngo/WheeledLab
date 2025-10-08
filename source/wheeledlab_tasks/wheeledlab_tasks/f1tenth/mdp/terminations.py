"""
Episode termination conditions for the overtaking task.

This module defines boolean termination predicates (one tensor boolean per env)
that the training loop can OR/AND to decide when an episode ends.

Design goals
------------
- Functions are idempotent and safe to call every step.
- They allocate and update per-env history buffers on first use.
- They support multiple maps by batching envs with the same map_level.

Conventions
-----------
- "Wall collision" means the robot comes within a fixed radius of either the
  inner or outer track boundary.
- Opponent-related terminations use the same s/d metrics and histories defined in
  observations/rewards (Δs along the track and Δd lateral offset).
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


def wall_collision(env: ManagerBasedEnv) -> torch.Tensor:
    """Terminate when the robot is too close to the inner or outer wall.

    How it works
    ------------
    For each env:
      1) Using current world XY position, find the closest point on the inner
         and outer boundaries (via Frenet nearest search).
      2) If either absolute distance is below HARD_WALL_COLLISION_RADIUS,
         mark a collision (True).
      3) Store a short collision history buffer, update counters, and log extras.

    Returns
    -------
    torch.BoolTensor [num_envs]:
        True for envs considered to be in collision this step (index defined by
        WALL_COLLISION_CHECK_IDX in the recent history buffer).
    """
    pos_xy_world = mdp.root_pos_w(env)[..., :2]

    # Lazily allocate common state
    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    if not hasattr(env, "_wall_collision_history"):
        env._rew_history_length = CONFIG["env_config"]["REW_HISTORY_LENGTH"]
        env._wall_collision_history = torch.zeros(
            (env.num_envs, env._rew_history_length), dtype=torch.long, device=env.device
        )

    if not hasattr(env, "_outer_list"):
        env._outer_list = [
            torch.tensor(outer, device=env.device, dtype=torch.float32)
            for outer in env.scene.terrain.cfg.outer_list
        ]
    if not hasattr(env, "_inner_list"):
        env._inner_list = [
            torch.tensor(inner, device=env.device, dtype=torch.float32)
            for inner in env.scene.terrain.cfg.inner_list
        ]

    # Group envs by map level (each map has its own boundary tensors)
    map_levels = env._map_levels
    unique_map_levels = torch.unique(map_levels)

    collision_bool = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    for map_level in unique_map_levels:
        env_mask = map_levels == map_level
        if not env_mask.any():
            continue

        map_positions = pos_xy_world[env_mask]
        inner_xy_world = env._inner_list[map_level][:, :2]
        outer_xy_world = env._outer_list[map_level][:, :2]

        # Efficient nearest-boundary distance using Frenet helper (vectorized).
        _, dist_from_inner = find_frenet_coord_along_waypoints(inner_xy_world, map_positions)
        _, dist_from_outer = find_frenet_coord_along_waypoints(outer_xy_world, map_positions)

        collision_with_inner = torch.abs(dist_from_inner) < CONFIG["env_config"]["HARD_WALL_COLLISION_RADIUS"]
        collision_with_outer = torch.abs(dist_from_outer) < CONFIG["env_config"]["HARD_WALL_COLLISION_RADIUS"]

        collisions_in_map = (collision_with_inner | collision_with_outer).long()
        collision_bool[env_mask] = collisions_in_map

    # Shift collision history and insert current flag
    env._wall_collision_history[:, 1:] = env._wall_collision_history[:, :-1].clone()
    env._wall_collision_history[:, 0] = collision_bool

    # Initialize counters if missing (kept consistent with rest of codebase)
    if not hasattr(env, "_opponent_overtaken_counter"):
        env._opponent_overtaken_counter = 0
        env._opponent_collision_counter = 0
        env._wall_collision_counter = 0

    # Count the number of collisions at the configured check index
    env._wall_collision_counter += torch.sum(
        env._wall_collision_history[:, CONFIG["env_config"]["WALL_COLLISION_CHECK_IDX"]].float()
    )

    # ---- Logging for diagnostics (W&B, play_policy), guarded to avoid surprises ----
    env.extras["log"]["Info/wall_collision_counter"] = env._wall_collision_counter
    env.extras["log"]["Info/wall_collision_step"] = torch.sum(
        env._wall_collision_history[:, CONFIG["env_config"]["WALL_COLLISION_CHECK_IDX"]].float()
    )
    if hasattr(env, "_action_history"):
        env.extras["log"]["Info/effort_action_steering"] = torch.mean(env._action_history[:, :, 1] ** 2, dim=1)
        env.extras["log"]["Info/effort_action_throttle"] = torch.mean(env._action_history[:, :, 0] ** 2, dim=1)
        env.extras["log"]["Info/var_action_steering"]    = torch.var(env._action_history[:, :, 1], dim=1)
        env.extras["log"]["Info/var_action_throttle"]    = torch.var(env._action_history[:, :, 0], dim=1)
    # -------------------------------------------------------------------------------

    # The termination flag used downstream is the value at a specific history index
    return env._wall_collision_history[:, CONFIG["env_config"]["WALL_COLLISION_CHECK_IDX"]].bool()


def opponent_collision(env: ManagerBasedEnv) -> torch.Tensor:
    """Terminate when ego is colliding with the opponent (Δs & Δd proximity).

    Conditions (OR of two cases; copied from original, behavior preserved)
    ----------------------------------------------------------------------
    Case A (general near alignment & cross-position small):
      |Δs| < OPP_FRONT_COLLISION_RADIUS
      |Δd| < OPP_LAT_COLLISION_RADIUS
      mean(cross_pos_history) < CROSS_POS_LIM

    OR

    Case B (tight proximity fallback):
      |Δs| < 0.55 AND |Δd| < 0.35

    where:
      - Δs, Δd come from histories computed in observation code
      - cross_pos_history is a left/right positioning cue w.r.t ego heading

    Returns
    -------
    torch.BoolTensor [num_envs]:
        True for envs considered colliding at OPPONENT_COLLISION_CHECK_IDX in history.
    """
    num_episodes = env.common_step_counter // env.max_episode_length
    if num_episodes < CONFIG["env_config"]["IGNORE_OPPONENT_UNTIL_EP"]:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    # Histories used by this predicate (allocate if missing)
    if not hasattr(env, "_opponent_collision_history"):
        env._rew_history_length = CONFIG["env_config"]["REW_HISTORY_LENGTH"]
        env._opponent_collision_history = torch.zeros(
            (env.num_envs, env._rew_history_length), dtype=torch.long, device=env.device
        )
    if not hasattr(env, "_cross_pos_history"):
        env._cross_pos_history = torch.zeros(
            (env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device
        )
    if not hasattr(env, "_s_idx_diff_history"):
        env._s_idx_diff_history = torch.ones(
            (env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device
        )
    if not hasattr(env, "_d_diff_history"):
        env._d_diff_history = torch.ones(
            (env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device
        )

    # Main condition (unchanged from original)
    near_front_lat = (
        (torch.abs(env._s_idx_diff_history[:, 0]) < CONFIG["env_config"]["OPP_FRONT_COLLISION_RADIUS"])
        & (torch.abs(env._d_diff_history[:, 0]) < CONFIG["env_config"]["OPP_LAT_COLLISION_RADIUS"])
        & (torch.mean(env._cross_pos_history[:, :], dim=1) < CONFIG["env_config"]["CROSS_POS_LIM"])
    )
    tight_fallback = (torch.abs(env._s_idx_diff_history[:, 0]) < 0.55) & (torch.abs(env._d_diff_history[:, 0]) < 0.35)

    opp_collision = near_front_lat | tight_fallback

    # Update collision history
    env._opponent_collision_history[:, 1:] = env._opponent_collision_history[:, :-1].clone()
    env._opponent_collision_history[:, 0] = opp_collision

    # Defensive init of counters (mirrors use elsewhere)
    if not hasattr(env, "_opponent_collision_counter"):
        env._opponent_collision_counter = 0
    env._opponent_collision_counter += torch.sum(
        env._opponent_collision_history[:, CONFIG["env_config"]["OPPONENT_COLLISION_CHECK_IDX"]].float()
    )

    return env._opponent_collision_history[:, CONFIG["env_config"]["OPPONENT_COLLISION_CHECK_IDX"]].bool()


def opponent_overtaken(env: ManagerBasedEnv) -> torch.Tensor:
    """Terminate when an overtake has been fully completed.

    Logic
    -----
    We rely on a rolling boolean `env._opponent_overtaken_history` maintained
    elsewhere (e.g., in the reward). If every element in the history window is 1
    for an env, we consider the overtake 'completed' and return True.

    Also increments an overtake counter used for logging.
    """
    num_episodes = env.common_step_counter // env.max_episode_length
    if num_episodes < CONFIG["env_config"]["IGNORE_OPPONENT_UNTIL_EP"]:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    if not hasattr(env, "_opponent_overtaken_bool"):
        env._opponent_overtaken_bool = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_opponent_overtaken_counter"):
        env._opponent_overtaken_counter = 0
    if not hasattr(env, "_opponent_overtaken_history"):
        env._rew_history_length = CONFIG["env_config"]["REW_HISTORY_LENGTH"]
        env._opponent_overtaken_history = torch.zeros(
            (env.num_envs, env._rew_history_length), dtype=torch.long, device=env.device
        )

    overtake_completed = env._opponent_overtaken_history.min(dim=1).values == 1
    env._opponent_overtaken_counter += torch.sum(overtake_completed.float())
    return overtake_completed


def far_from_opponent(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    opponent_cfg: SceneEntityCfg = SceneEntityCfg("opponent"),
) -> torch.Tensor:
    """Optional termination when the opponent is too far ahead/behind in s.

    Computes Δs = (s_opp - s_ego) (shortest around the loop). If Δs exceeds a
    fixed threshold, returns True (episode can end due to
    separation being too large to be interesting).

    Returns
    -------
    torch.BoolTensor [num_envs]
    """
    ego_position_xy = mdp.root_pos_w(env=env, asset_cfg=asset_cfg)[..., :2]
    opp_position_xy = mdp.root_pos_w(env=env, asset_cfg=opponent_cfg)[..., :2]

    if not hasattr(env, "_map_levels"):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_waypoints_list"):
        # Ensure waypoint lists exist if this function is called early.
        env._waypoints_list = [
            torch.tensor(wps, device=env.device, dtype=torch.float32)
            for wps in env.scene.terrain.cfg.waypoints_list
        ]

    map_levels = env._map_levels
    unique_map_levels = torch.unique(map_levels)

    delta_s_opp_ego = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    for map_level in unique_map_levels:
        env_mask = map_levels == map_level
        if not env_mask.any():
            continue

        waypoints_world = env._waypoints_list[map_level][:, :2]
        num_waypoints = len(waypoints_world)

        ego_idx, _ = find_frenet_coord_along_waypoints(waypoints_world, ego_position_xy[env_mask])
        opp_idx, _ = find_frenet_coord_along_waypoints(waypoints_world, opp_position_xy[env_mask])

        delta_s_opp_ego[env_mask] = (
            (opp_idx - ego_idx + num_waypoints // 2) % num_waypoints - num_waypoints // 2
        ).float()

    far_bool = delta_s_opp_ego > CONFIG["env_config"]["OPPONENT_FAR_AWAY_IDX"]  # index-threshold; keep original semantics
    return far_bool.bool()


def time_out(env: ManagerBasedEnv) -> torch.Tensor:
    """Terminate when the per-episode step counter reaches the maximum."""
    return env.episode_length_buf >= env.max_episode_length
