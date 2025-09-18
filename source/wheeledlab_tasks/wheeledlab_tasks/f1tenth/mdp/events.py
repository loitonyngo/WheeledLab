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
    
#--------
# Common helpers
# ---------------------------------------------------------------------------
        
def _init_common_histories(env: ManagerBasedEnv,
                           env_ids: torch.Tensor):
    """Initialize histories/buffers that are common across reset functions."""
    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    if not hasattr(env, '_initial_waypoint_indices'):
        env._initial_waypoint_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    if not hasattr(env, '_total_progress_indices'):
        env._total_progress_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        
    if not hasattr(env, '_reset_env_bool'):
        env._reset_env_bool = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    if not hasattr(env, '_progress_history_indices'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._progress_history_indices = torch.zeros(
            (env.num_envs, env._rew_history_length),
            dtype=torch.long,
            device=env.device,
        )

    if not hasattr(env, '_action_history_length'):
        env._action_history_length = CONFIG['env_config']['ACTION_HISTORY_LENGTH']

    if not hasattr(env, '_action_history'):
        env._action_history = torch.zeros(
            (env.num_envs, env._action_history_length, 2),
            dtype=torch.float32,
            device=env.device,
        )

    if not hasattr(env, '_obs_history_length'):
        env._obs_history_length = CONFIG['env_config']['OBS_HISTORY_LENGTH']

    for name in ['_base_lin_vel_x_history', '_base_lin_vel_y_history',
                 '_base_ang_vel_z_history', '_target_velocity_history', '_target_steering_angle_history']:
        if not hasattr(env, name):
            setattr(env, name, torch.zeros(
                (env.num_envs, env._obs_history_length),
                dtype=torch.float32,
                device=env.device,
            ))

    if not hasattr(env, '_wall_collision_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._wall_collision_history = torch.ones(
            (env.num_envs, env._rew_history_length),
            dtype=torch.long,
            device=env.device,
        )


def _reset_ego_state(env: ManagerBasedEnv, 
                     env_ids: torch.Tensor,
                     asset, 
                     valid_poses, 
                     current_idx):
    
    """Reset the ego (robot) state."""
    posns = torch.stack([torch.tensor(x.pos, device=env.device) for x in valid_poses]).float()
    oris = torch.stack([
        math_utils.quat_from_euler_xyz(*torch.deg2rad(torch.tensor(x.rot_euler_xyz_deg, device=env.device)))
        for x in valid_poses
    ]).float()
    lin_vels = torch.stack([torch.tensor(x.lin_vel, device=env.device) for x in valid_poses]).float()
    ang_vels = torch.stack([torch.tensor(x.ang_vel, device=env.device) for x in valid_poses]).float()

    positions = posns + asset.data.default_root_state[env_ids, :3]
    orientations = oris
    lin_vels += asset.data.default_root_state[env_ids, 7:10]
    ang_vels += asset.data.default_root_state[env_ids, 10:13]

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.cat([lin_vels, ang_vels], dim=-1), env_ids=env_ids)

    # Reset histories for these env_ids
    env._initial_waypoint_indices[env_ids] = current_idx.clone()  
    env._reset_env_bool[env_ids] = torch.ones(len(env_ids), dtype=bool, device=env.device)
    env._total_progress_indices[env_ids] = -current_idx.clone()  
    env._progress_history_indices[env_ids, :] =  torch.zeros(env._rew_history_length, dtype=torch.long, device=env.device)
    env._action_history[env_ids, :, :] = torch.zeros((len(env_ids), env._action_history_length, 2), dtype=torch.float32, device=env.device)
    env._base_lin_vel_x_history[env_ids, :] = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._base_lin_vel_y_history[env_ids, :] = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._base_ang_vel_z_history[env_ids, :] = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._target_velocity_history[env_ids, :] = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._target_steering_angle_history[env_ids, :] = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)

    env._wall_collision_history[env_ids, :] = torch.zeros((len(env_ids), env._rew_history_length), dtype=torch.long, device=env.device)



# ---------------------------------------------------------------------------
# Opponent-specific helpers
# ---------------------------------------------------------------------------

def _reset_opponent_state(env: ManagerBasedEnv, 
                          env_ids: torch.Tensor,
                          opponent_asset, 
                          opp_valid_poses):
    """Reset the opponent's pose and velocity."""
    opp_posns = torch.stack([torch.tensor(x.pos, device=env.device) for x in opp_valid_poses]).float()
    opp_oris = torch.stack([
        math_utils.quat_from_euler_xyz(*torch.deg2rad(torch.tensor(x.rot_euler_xyz_deg, device=env.device)))
        for x in opp_valid_poses
    ]).float()
    opp_lin_vels = torch.stack([torch.tensor(x.lin_vel, device=env.device) for x in opp_valid_poses]).float() * 0
    opp_ang_vels = torch.stack([torch.tensor(x.ang_vel, device=env.device) for x in opp_valid_poses]).float()

    opp_positions = opp_posns + opponent_asset.data.default_root_state[env_ids, :3]
    opp_orientations = opp_oris
    opp_lin_vels += opponent_asset.data.default_root_state[env_ids, 7:10]
    opp_ang_vels += opponent_asset.data.default_root_state[env_ids, 10:13]

    opponent_asset.write_root_pose_to_sim(torch.cat([opp_positions, opp_orientations], dim=-1), env_ids=env_ids)
    opponent_asset.write_root_velocity_to_sim(torch.cat([opp_lin_vels, opp_ang_vels], dim=-1), env_ids=env_ids)


def _init_opponent_histories(env: ManagerBasedEnv, env_ids: torch.Tensor, ego_current_idx, opp_current_idx):
    """Initialize and reset opponent-specific histories."""
    if not hasattr(env, '_prev_delta_s_opp_ego'):
        env._prev_delta_s_opp_ego = torch.ones(
            env.num_envs, dtype=torch.float32, device=env.device
        ) * CONFIG['env_config']['OPPONENT_INIT_DISTANCE_IDX_MAX'] * CONFIG['env_config']['LEN_S_IDX']

    if not hasattr(env, '_traversability_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._traversability_history = torch.ones(
            (env.num_envs, env._rew_history_length),
            dtype=torch.long,
            device=env.device,
        )

    # Opponent dynamic properties
    if not hasattr(env, '_opponent_type'):
        env._opponent_type = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, '_opponent_vel_scaling'):
        env._opponent_vel_scaling = torch.ones(env.num_envs, dtype=torch.float16, device=env.device)
    if not hasattr(env, '_opponent_trajectory_alpha'):
        env._opponent_trajectory_alpha = torch.ones(env.num_envs, dtype=torch.float16, device=env.device)
    if not hasattr(env, '_opponent_overtaken_bool'):
        env._opponent_overtaken_bool = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, '_opponent_speed'):
        env._opponent_speed = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, '_opponent_d_dot'):
        env._opponent_d_dot = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, '_opponent_heading'):
        env._opponent_heading = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, '_opponent_always_ahead'):
        env._opponent_always_ahead = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, '_opponent_collision_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._opponent_collision_history = torch.zeros(
            (env.num_envs, env._rew_history_length),
            dtype=torch.long,
            device=env.device,
        )
    if not hasattr(env, '_s_idx_diff_history'):
        env._s_idx_diff_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
    if not hasattr(env, '_d_diff_history'):
        env._d_diff_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
    if not hasattr(env, '_vx_diff_history'):
        env._vx_diff_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
    if not hasattr(env, '_heading_diff_history'):
        env._heading_diff_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
    if not hasattr(env, '_cross_pos_history'):
        env._cross_pos_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
    if not hasattr(env, '_gap_inner_history'):
        env._gap_inner_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
    if not hasattr(env, '_gap_outer_history'):
        env._gap_outer_history = torch.zeros((env.num_envs, env._obs_history_length), dtype=torch.float32, device=env.device)
        
    if not hasattr(env, '_opponent_overtaken_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._opponent_overtaken_history = torch.zeros(
            (env.num_envs, env._rew_history_length),
            dtype=torch.long,
            device=env.device,
        )    
    if not hasattr(env, '_opponent_overtaken_counter'):
        env._opponent_overtaken_counter = 0
        env._opponent_collision_counter = 0
        env._wall_collision_counter = 0

        env._opponent_vel_scaling_lvl = CONFIG['env_config']['OPP_INIT_VEL_SCALING']
        
    if not hasattr(env, '_opponent_s_history'):
        env._opponent_s_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )
    if not hasattr(env, '_opponent_d_history'):
        env._opponent_d_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )

    if not hasattr(env, '_opponent_traj_x_shift') or not hasattr(env, '_opponent_traj_y_shift'):
        env._opponent_traj_x_shift = torch.zeros(
            env.num_envs, 
            dtype=torch.float32,
            device=env.device
        )        
        env._opponent_traj_y_shift = torch.zeros(
            env.num_envs, 
            dtype=torch.float32,
            device=env.device
        )                
        
    # Reset for this batch
    env._prev_delta_s_opp_ego[env_ids]  = torch.zeros(len(env_ids), dtype=torch.float32, device=env.device)
    env._opponent_type[env_ids] = torch.tensor(
        np.floor(np.random.rand(len(env_ids)) * 3),
        device=env.device,
        dtype=torch.long,
    )

    env._opponent_vel_scaling[env_ids] = torch.normal(
        mean=torch.ones(len(env_ids), dtype=torch.float16, device=env.device) * env._opponent_vel_scaling_lvl,
        std=CONFIG['env_config']['OPP_STD_VEL_SCALING'],
    ).clamp(min=CONFIG['env_config']['OPP_MIN_VEL_SCALING'], max=env._opponent_vel_scaling_lvl)

    # define type of trajectory (defined, generalized)
    if CONFIG['env_config']['OPP_GENERALIZATION']:
        env._opponent_trajectory_alpha[env_ids] = torch.tensor(
            np.random.rand(len(env_ids)),
            device=env.device,
            dtype=torch.float16,
        )
    else:
        env._opponent_trajectory_alpha[env_ids] = torch.ones(len(env_ids), dtype=torch.float16, device=env.device)


    env._opponent_overtaken_bool[env_ids]        = torch.zeros(len(env_ids), dtype=bool, device=env.device)
    env._opponent_speed[env_ids]                 = torch.zeros(len(env_ids), dtype=torch.float32, device=env.device)
    env._opponent_d_dot[env_ids]                 = torch.zeros(len(env_ids), dtype=torch.float32, device=env.device)
    env._opponent_heading[env_ids]               = torch.zeros(len(env_ids), dtype=torch.float32, device=env.device)
    
    env._opponent_always_ahead[env_ids]          = torch.rand(len(env_ids), device=env.device) < CONFIG['env_config']['OPPONENT_ALWAYS_AHEAD_PERCENTAGE']
        
    env._opponent_collision_history[env_ids, :]  = torch.zeros((len(env_ids), env._rew_history_length), dtype=torch.long, device=env.device)
    env._opponent_overtaken_history[env_ids, :]  = torch.zeros((len(env_ids), env._rew_history_length), dtype=torch.long, device=env.device)
    env._s_idx_diff_history[env_ids, :]          = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._d_diff_history[env_ids, :]              = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._vx_diff_history[env_ids, :]             = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._heading_diff_history[env_ids, :]        = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._cross_pos_history[env_ids, :]           = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._gap_inner_history[env_ids, :]           = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._gap_outer_history[env_ids, :]           = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._opponent_s_history[env_ids, :]          = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._opponent_d_history[env_ids, :]          = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)

    env._opponent_traj_x_shift[env_ids] = torch.rand(len(env_ids), device=env.device)*CONFIG['env_config']['OPPONENT_TRAJ_XY_SHIFT_MAX']*2 - CONFIG['env_config']['OPPONENT_TRAJ_XY_SHIFT_MAX']
    env._opponent_traj_y_shift[env_ids] = torch.rand(len(env_ids), device=env.device)*CONFIG['env_config']['OPPONENT_TRAJ_XY_SHIFT_MAX']*2 - CONFIG['env_config']['OPPONENT_TRAJ_XY_SHIFT_MAX']

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def reset_root_state_random(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    _init_common_histories(env, env_ids)
    env._map_levels[env_ids] = torch.tensor(np.floor(np.random.rand(len(env_ids))*len(env.cfg.scene.terrain.waypoints_list)), device = env.device, dtype=torch.long)

    valid_poses, current_idx_np = terrain.cfg.generate_random_poses_from_waypoints(
        env=env, env_ids=env_ids, num_poses=len(env_ids), max_radius_offset=0.5
    )
    current_idx = torch.tensor(current_idx_np, dtype=torch.long, device=env.device)

    _reset_ego_state(env, env_ids, asset, valid_poses, current_idx)


def reset_root_state_random_with_opponent(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    opponent_asset_cfg: SceneEntityCfg = SceneEntityCfg("opponent"),
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    opponent_asset: RigidObject | Articulation = env.scene[opponent_asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    _init_common_histories(env)
    env._map_levels[env_ids] = torch.tensor(np.floor(np.random.rand(len(env_ids))*len(env.cfg.scene.terrain.traversability_hashmap_list)), device = env.device, dtype=torch.long)

    ego_valid_poses, ego_idx_np, opp_valid_poses, opp_idx_np = terrain.cfg.generate_random_poses_from_waypoints_with_opponent(
        env=env, env_ids=env_ids, num_poses=len(env_ids), max_radius_offset=0.5
    )
    current_idx = torch.tensor(ego_idx_np, dtype=torch.long, device=env.device)
    opp_current_idx = torch.tensor(opp_idx_np, dtype=torch.long, device=env.device)
    
    _reset_ego_state(env, env_ids, asset, ego_valid_poses, current_idx)
    _reset_opponent_state(env, env_ids, opponent_asset, opp_valid_poses)
    _init_opponent_histories(env, env_ids, current_idx, opp_current_idx)

def reset_root_state_start_idx(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the ego agent at specific start indices (deterministic)."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # 1. Ensure histories exist
    _init_common_histories(env)

    # 2. Sample poses from start indices
    valid_poses, current_idx_np = terrain.cfg.generate_start_idx_poses(
        env=env, env_ids=env_ids, num_poses=len(env_ids)
    )
    current_idx = torch.tensor(current_idx_np, dtype=torch.long, device=env.device)

    # 3. Reset ego (robot) state
    _reset_ego_state(env, env_ids, asset, valid_poses, current_idx)

    # Reset those for selected envs
    env._total_progress_indices[env_ids] = torch.zeros(
        len(env_ids), dtype=torch.long, device=env.device
    )

def move_opponent_s_based(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("opponent")
):
    num_episodes = env.common_step_counter // env.max_episode_length
    if num_episodes < CONFIG['env_config']['STATIC_OPPONENT_UNTIL_EP']:
        return None
    
    # Initialize attributes if they don't exist
    if not hasattr(env, '_opponent_type'):
        env._opponent_type = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, '_opponent_vel_scaling'):
        env._opponent_vel_scaling = torch.ones(env.num_envs, dtype=torch.float16, device=env.device)
    if not hasattr(env, '_opponent_trajectory_alpha'):
        env._opponent_trajectory_alpha = torch.ones(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    if not hasattr(env, '_opp_traj_center_list'):
        env._opp_traj_center_list = [
            torch.tensor(center, device=env.device, dtype=torch.float32)
            for center in env.scene.terrain.cfg.opp_traj_center_list
        ]

    if not hasattr(env, '_opp_traj_iqp_list'):
        env._opp_traj_iqp_list = [
            torch.tensor(iqp, device=env.device, dtype=torch.float32)
            for iqp in env.scene.terrain.cfg.opp_traj_iqp_list
        ]

    if not hasattr(env, '_opp_traj_sp_list'):
        env._opp_traj_sp_list = [
            torch.tensor(sp, device=env.device, dtype=torch.float32)
            for sp in env.scene.terrain.cfg.opp_traj_sp_list
        ]
        
    if not hasattr(env, '_opponent_overtaken_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._opponent_overtaken_history = torch.zeros(
            (env.num_envs, env._rew_history_length),
            dtype=torch.long,
            device=env.device,
        )    
    if not hasattr(env, '_opponent_collision_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._opponent_collision_history = torch.zeros(
            (env.num_envs, env._rew_history_length),
            dtype=torch.long,
            device=env.device,
        )
    if not hasattr(env, '_wall_collision_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._wall_collision_history = torch.zeros(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
        
    num_episodes = env.common_step_counter // env.max_episode_length
    if env.common_step_counter % CONFIG['env_config']['VEL_SCALING_CHECK_STEPS'] == 0:
        if (env._opponent_overtaken_counter/(env._opponent_collision_counter+env._opponent_overtaken_counter+env._wall_collision_counter+1)) > CONFIG['env_config']['OT_COLLISION_RATIO_LVL_UP']:
            env._opponent_vel_scaling_lvl += CONFIG['env_config']['OPP_VEL_SCALING_INCREMENT']
            env._opponent_overtaken_counter = 0
            env._opponent_collision_counter = 0
            env._wall_collision_counter = 0
        elif (env._opponent_overtaken_counter/(env._opponent_collision_counter+env._opponent_overtaken_counter+env._wall_collision_counter+1)) < CONFIG['env_config']['OT_COLLISION_RATIO_LVL_DOWN']: 
            env._opponent_overtaken_counter = 0
            env._opponent_collision_counter = 0
            env._wall_collision_counter = 0
            env._opponent_vel_scaling_lvl -= CONFIG['env_config']['OPP_VEL_SCALING_DECREASE']
        else:
            env._opponent_overtaken_counter = 0
            env._opponent_collision_counter = 0
            env._wall_collision_counter = 0
            
    env._opponent_vel_scaling_lvl = max(env._opponent_vel_scaling_lvl, CONFIG['env_config']['OPP_MIN_VEL_SCALING'])
    
    if env.common_step_counter % 500 == 0:
        env._opponent_overtaken_counter = 0
        env._opponent_collision_counter = 0
        env._wall_collision_counter = 0
    
    asset = env.scene[asset_cfg.name]
    ego_position_xy = mdp.root_pos_w(env=env, asset_cfg=SceneEntityCfg("robot"))[:, :2]
    opp_position_xy = mdp.root_pos_w(env=env, asset_cfg=SceneEntityCfg("opponent"))[:, :2]
    
    # Initialize new positions (preserve Z)
    new_positions = asset.data.root_pos_w.clone()
    new_orientations = asset.data.root_quat_w.clone()
    new_lin_velocities = torch.zeros_like(asset.data.root_lin_vel_w)
    new_ang_velocities = torch.zeros_like(asset.data.root_ang_vel_w)
    
    # Get opponent types and map levels for the given env_ids
    opponent_types = env._opponent_type[env_ids]
    map_levels = env._map_levels[env_ids]
    unique_map_levels = torch.unique(map_levels)
    
    for map_level in unique_map_levels:
        map_mask = (map_levels == map_level)
        if not map_mask.any():
            continue
        
        waypoints_xy_world = env._waypoints_list[map_level][:, :2]
        
        level_opp_types = opponent_types[map_mask]
        unique_opp_types = torch.unique(level_opp_types)
        
        if CONFIG['env_config']['OPP_GENERALIZATION']:

            for opp_type in unique_opp_types:
                type_mask = (level_opp_types == opp_type)
                if not type_mask.any():
                    continue
                combined_mask = map_mask & (opponent_types == opp_type)

                # Select two base trajectories to blend
                if opp_type == 0:  # Centerline dominant
                    traj = env._opp_traj_center_list[map_level]
                elif opp_type == 1:  # IQP dominant
                    traj = env._opp_traj_iqp_list[map_level]
                elif opp_type == 2:  # IQP dominant
                    traj = env._opp_traj_sp_list[map_level]
                else:
                    raise ValueError(f"Unknown opponent type: {opp_type}")

                traj_xy, traj_psi, traj_vel = traj[:, :2], traj[:, 2], traj[:, 3]
                traj_xy_shift = torch.stack([
                    env._opponent_traj_x_shift[combined_mask],
                    env._opponent_traj_y_shift[combined_mask]
                ], dim=1)
                # Closest indices for ego and opponent
                ego_idx, _ = find_frenet_coord_along_waypoints(traj_xy, ego_position_xy[combined_mask])
                opp_idx, _ = find_frenet_coord_along_waypoints(traj_xy, opp_position_xy[combined_mask] - traj_xy_shift)

                # Compute move steps
                move_steps = (
                    traj_vel[opp_idx] 
                    * CONFIG['env_config']['OPP_MOVE_DT'] 
                    * (env._opponent_vel_scaling[combined_mask]) 
                    / CONFIG['env_config']['LEN_S_IDX']
                ).int()

                # Next index: if always ahead, place at fixed offset; else advance along trajectory
                next_idx = torch.where(
                    env._opponent_always_ahead[combined_mask],
                    (ego_idx + CONFIG['env_config']['OPPONENT_INIT_DISTANCE_IDX_MIN']) % len(traj_xy),
                    torch.where(
                        env.episode_length_buf[combined_mask] < CONFIG['env_config']['OPPONENT_STARTS_MOVING_AFTER_STEP'],
                        opp_idx,
                        (opp_idx + move_steps + np.random.randint(low=1, high=2)) % len(traj_xy)
                    )
                )

                # Candidate next state
                pos = traj_xy[next_idx] + traj_xy_shift
                psi = traj_psi[next_idx]

                # Update positions (XY only, preserve Z offset + small jitter)
                new_positions[combined_mask, 0] = pos[:, 0] + env.scene.env_origins[combined_mask, 0] 
                new_positions[combined_mask, 1] = pos[:, 1] + env.scene.env_origins[combined_mask, 1] 

                # Speed as displacement / dt
                disp = pos - opp_position_xy[combined_mask]
                env._opponent_speed[combined_mask] = torch.norm(disp, dim=1) / CONFIG['env_config']['OPP_MOVE_DT']

                # Heading from displacement
                env._opponent_heading[combined_mask] = torch.atan2(disp[:, 1], disp[:, 0])

                # Orientation quaternion (yaw only)
                new_orientations[combined_mask] = torch.stack([
                    torch.cos(psi / 2.0),           # w
                    torch.zeros_like(psi),          # x
                    torch.zeros_like(psi),          # y
                    torch.sin(psi / 2.0)            # z
                ], dim=1)
        
        
                # if opp_type == 0:  # Centerline dominant
                #     traj_a = env._opp_traj_center_list[map_level]
                #     traj_b = env._opp_traj_iqp_list[map_level]
                # elif opp_type == 1:  # IQP dominant
                #     traj_a = env._opp_traj_iqp_list[map_level]
                #     traj_b = env._opp_traj_sp_list[map_level]
                # elif opp_type == 2:  # IQP dominant
                #     traj_a = env._opp_traj_sp_list[map_level]
                #     traj_b = env._opp_traj_center_list[map_level]
                # else:
                #     raise ValueError(f"Unknown opponent type: {opp_type}")

                # # Split into components
                # traj_a_xy, traj_a_psi, traj_a_vel = traj_a[:, :2], traj_a[:, 2], traj_a[:, 3]
                # traj_b_xy, traj_b_psi, traj_b_vel = traj_b[:, :2], traj_b[:, 2], traj_b[:, 3]

                # # Find closest indices for ego (used to place opponent ahead if needed)
                # ego_idx_a, _ = find_frenet_coord_along_waypoints(traj_a_xy, ego_position_xy[combined_mask])
                # ego_idx_b, _ = find_frenet_coord_along_waypoints(traj_b_xy, ego_position_xy[combined_mask])

                # # Find closest indices for opponent
                # opp_idx_a, _ = find_frenet_coord_along_waypoints(traj_a_xy, opp_position_xy[combined_mask])
                # opp_idx_b, _ = find_frenet_coord_along_waypoints(traj_b_xy, opp_position_xy[combined_mask])

                # # Compute move steps separately
                # move_steps_a = (traj_a_vel[opp_idx_a] * CONFIG['env_config']['OPP_MOVE_DT'] * 
                #             (env._opponent_vel_scaling[combined_mask])/CONFIG['env_config']['LEN_S_IDX']).int()
                # move_steps_b = (traj_b_vel[opp_idx_b] * CONFIG['env_config']['OPP_MOVE_DT'] * 
                #             (env._opponent_vel_scaling[combined_mask])/CONFIG['env_config']['LEN_S_IDX']).int()

                # # Next indices depending on "always ahead"
                # next_idx_a = torch.where(
                #                         env._opponent_always_ahead[combined_mask],
                #                         (ego_idx_a + CONFIG['env_config']['OPPONENT_INIT_DISTANCE_IDX_MIN']) % len(traj_a_xy),
                #                             torch.where(env.episode_length_buf[combined_mask] <CONFIG['env_config']['OPPONENT_STARTS_MOVING_AFTER_STEP'], 
                #                             (opp_idx_a + 1) % len(traj_a_xy), 
                #                             (opp_idx_a + move_steps_a + np.random.randint(low=1, high=2)) % len(traj_a_xy)
                #                         )
                #                         )
                

                # next_idx_b = torch.where(
                #                         env._opponent_always_ahead[combined_mask],
                #                         (ego_idx_b + CONFIG['env_config']['OPPONENT_INIT_DISTANCE_IDX_MIN']) % len(traj_b_xy),
                #                             torch.where(env.episode_length_buf[combined_mask] < CONFIG['env_config']['OPPONENT_STARTS_MOVING_AFTER_STEP'], 
                #                             (opp_idx_b + 1) % len(traj_b_xy), 
                #                             (opp_idx_b + move_steps_b + np.random.randint(low=1, high=2)) % len(traj_b_xy)
                #                         )
                #                         )       

                
                
                # # Candidate next states
                # pos_a = traj_a_xy[next_idx_a]
                # pos_b = traj_b_xy[next_idx_b]
                # psi_a = traj_a_psi[next_idx_a]
                # psi_b = traj_b_psi[next_idx_b]
                # vel_a = traj_a_vel[next_idx_a]
                # vel_b = traj_b_vel[next_idx_b]

                # # Blend with alpha
                # alpha = env._opponent_trajectory_alpha[combined_mask].unsqueeze(-1)
                # blended_pos = alpha * pos_a + (1 - alpha) * pos_b

                # heading_a = torch.stack([torch.cos(psi_a), torch.sin(psi_a)], dim=-1)
                # heading_b = torch.stack([torch.cos(psi_b), torch.sin(psi_b)], dim=-1)
                # blended_heading = alpha * heading_a + (1 - alpha) * heading_b
                # blended_psi = torch.atan2(blended_heading[:,1], blended_heading[:,0])

                # # blended_vel = (alpha.squeeze() * vel_a + (1 - alpha.squeeze()) * vel_b)

                # # Update positions (X,Y only; preserve Z)
                # new_positions[combined_mask, 0] = blended_pos[:,0] + env.scene.env_origins[combined_mask, 0] + torch.rand_like(blended_pos[:,0])*0.05 + env._opponent_traj_x_shift[combined_mask]
                # new_positions[combined_mask, 1] = blended_pos[:,1] + env.scene.env_origins[combined_mask, 1] + torch.rand_like(blended_pos[:,1])*0.05 + env._opponent_traj_y_shift[combined_mask]

                # # Compute speed as displacement / dt
                # disp = blended_pos - opp_position_xy[combined_mask]
                # env._opponent_speed[combined_mask] = torch.norm(disp, dim=1) / CONFIG['env_config']['OPP_MOVE_DT']

                # # Heading from displacement (safer than blended heading if you want kinematics)
                # env._opponent_heading[combined_mask] = torch.atan2(disp[:,1], disp[:,0])

                # # TODO: if you want to update _opponent_d_dot, you’d need to project blended_pos into Frenet coords too

                # # Convert yaw to quaternion (for now keep simple, identity rotation in x/y)
                # new_orientations[combined_mask] = torch.stack([
                #     torch.cos(blended_psi/2.0),   # w
                #     torch.zeros_like(blended_psi),# x
                #     torch.zeros_like(blended_psi),# y
                #     torch.sin(blended_psi/2.0)    # z
                # ], dim=1)
                
        else:
            # Opponent always follows the IQP trajectory
            if CONFIG['env_config']['OPP_TRAJ'] == "mincurv":
                traj = env._opp_traj_iqp_list[map_level]
            elif CONFIG['env_config']['OPP_TRAJ'] == "sp":
                traj = env._opp_traj_sp_list[map_level]
            elif CONFIG['env_config']['OPP_TRAJ'] == "centerline":
                traj = env._opp_traj_center_list[map_level]                
            else:
                traj = env._opp_traj_iqp_list[map_level]
                
            traj_xy, traj_psi, traj_vel = traj[:, :2], traj[:, 2], traj[:, 3]
            traj_xy_shift = torch.stack([
            env._opponent_traj_x_shift[map_mask],
            env._opponent_traj_y_shift[map_mask]
        ], dim=1)
            # Closest indices for ego and opponent
            ego_idx, _ = find_frenet_coord_along_waypoints(traj_xy, ego_position_xy[map_mask])
            opp_idx, _ = find_frenet_coord_along_waypoints(traj_xy, opp_position_xy[map_mask] - traj_xy_shift)

            # Compute move steps
            move_steps = (
                traj_vel[opp_idx] 
                * CONFIG['env_config']['OPP_MOVE_DT'] 
                * (env._opponent_vel_scaling[map_mask]) 
                / CONFIG['env_config']['LEN_S_IDX']
            ).int()

            # Next index: if always ahead, place at fixed offset; else advance along trajectory
            next_idx = torch.where(
                env._opponent_always_ahead[map_mask],
                (ego_idx + CONFIG['env_config']['OPPONENT_INIT_DISTANCE_IDX_MIN']) % len(traj_xy),
                torch.where(
                    env.episode_length_buf[map_mask] < CONFIG['env_config']['OPPONENT_STARTS_MOVING_AFTER_STEP'],
                    opp_idx,
                    (opp_idx + move_steps + np.random.randint(low=1, high=2)) % len(traj_xy)
                )
            )

            # Candidate next state
            pos = traj_xy[next_idx] + traj_xy_shift
            psi = traj_psi[next_idx]
            vel = traj_vel[next_idx]

            # Update positions (XY only, preserve Z offset + small jitter)
            new_positions[map_mask, 0] = pos[:, 0] + env.scene.env_origins[map_mask, 0] 
            new_positions[map_mask, 1] = pos[:, 1] + env.scene.env_origins[map_mask, 1] 

            # Speed as displacement / dt
            disp = pos - opp_position_xy[map_mask]
            env._opponent_speed[map_mask] = torch.norm(disp, dim=1) / CONFIG['env_config']['OPP_MOVE_DT']

            # Heading from displacement
            env._opponent_heading[map_mask] = torch.atan2(disp[:, 1], disp[:, 0])

            # Orientation quaternion (yaw only)
            new_orientations[map_mask] = torch.stack([
                torch.cos(psi / 2.0),           # w
                torch.zeros_like(psi),          # x
                torch.zeros_like(psi),          # y
                torch.sin(psi / 2.0)            # z
            ], dim=1)
        
    # Apply all updates at once
    asset.write_root_pose_to_sim(
        torch.cat([new_positions, new_orientations], dim=1), 
        env_ids=env_ids
    )

# def opponent_vel_scaling_level_up(
#         env: ManagerBasedEnv,
#         env_ids: torch.Tensor,
#         asset_cfg: SceneEntityCfg = SceneEntityCfg("opponent")
#         ):
            
#         if (env._opponent_overtaken_counter/(env._opponent_collision_counter+env._opponent_overtaken_counter+env._wall_collision_counter+1))> CONFIG['env_config']['OT_COLLISION_RATIO_LVL_UP']:
#             env._opponent_vel_scaling_lvl += 0.1
        
#         env._opponent_overtaken_counter = 0
#         env._opponent_collision_counter = 0
#         env._wall_collision_counter = 0
    
    
# def enhanced_braking(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     k_p: float= 1,
#     k_d: float= 0,
#     min_speed_correction: float = -0.5,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ):
#     """Stable velocity control with capped acceleration boost"""
#     asset: RigidObject | Articulation = env.scene[asset_cfg.name]
#     dt = torch.tensor(env.step_dt, device=env_ids.device)

#     # error_velocity_history = env._action_history[env_ids, :, 0]*8 - env._base_lin_vel_x_history[env_ids, :]
#     if CONFIG['env_config']['INCREMENTAL_MODE']:
#         error_velocity_history = env._action_history[env_ids, :, 0]
#     else:
#         error_velocity_history = env._action_history[env_ids, :, 0]*8 - env._base_lin_vel_x_history[env_ids, :]
        
#     error_velocity_p = error_velocity_history[env_ids, 0]
#     error_velocity_d = error_velocity_history[env_ids, 0]-error_velocity_history[env_ids, 1]
    
#     error_braking_eff = torch.where(error_velocity_history[env_ids, -1] < -0.5, error_velocity_history[env_ids, -1], 0)
#     # pid controller to adjust velocity...
#     velocity_adjustment = (k_p * error_velocity_p + k_d * error_velocity_d /dt + error_braking_eff*0.5)*env._base_lin_vel_x_history[env_ids, 0]/10
    
#     velocity_adjustment = torch.clamp(velocity_adjustment, min=min_speed_correction, max=0)
    
#     if not hasattr(env, '_last_velocity_adjustment'):
#         env._last_velocity_adjustment = torch.zeros(env.num_envs, 
#                                     dtype=torch.float32,
#                                     device=env.device)

#     # velocity_adjustment = torch.where((velocity_adjustment+env._last_velocity_adjustment)>0.10, 0.10-env._last_velocity_adjustment, velocity_adjustment)
#     # velocity_adjustment = torch.where((velocity_adjustment+env._last_velocity_adjustment)<-1.0, -1-env._last_velocity_adjustment, velocity_adjustment)

#     env._last_velocity_adjustment = velocity_adjustment
#     # diff_velocity_adjustment = velocity_adjustment - env._last_velocity_adjustment
#     # max_diff_velocity_adjustment = 0.4
    
#     # velocity_adjustment_cropped = torch.where(abs(diff_velocity_adjustment)<max_diff_velocity_adjustment, velocity_adjustment, torch.where(diff_velocity_adjustment<0, env._last_velocity_adjustment - max_diff_velocity_adjustment, env._last_velocity_adjustment + max_diff_velocity_adjustment))
#     # # Get current velocities in world frame (only x,y components)
#     vel_w = asset.data.root_lin_vel_w[env_ids, :2]    
#     # Get heading angle (yaw) in world frame
#     heading_w = asset.data.heading_w[env_ids]

#     # Create forward direction vector from heading angle
#     forward_dir = torch.stack([torch.cos(heading_w), torch.sin(heading_w)], dim=-1)
#     # Project velocity onto forward direction to get forward speed component
#     forward_speed = torch.sum(vel_w * forward_dir, dim=-1, keepdim=True)
#     # braking = (error_velocity_history[env_ids, 0] -  error_velocity_history[env_ids, 1]) < 0 
#     # above_tc = (error_velocity_history[env_ids, 0] -  error_velocity_history[env_ids, 1]) >= 0 and forward_speed > 0.75
#     # Calculate new velocity by reducing forward speed when decelerating
#     new_forward_speed = torch.where(
#     ((error_velocity_history[env_ids, 0] < 1) & (env._base_lin_vel_x_history[env_ids, 0] > 0.75)).unsqueeze(-1),
#         forward_speed + velocity_adjustment.unsqueeze(-1),
#         forward_speed
#     )

#     new_forward_speed = new_forward_speed * 0.70 + env._base_lin_vel_x_history[env_ids, 0].unsqueeze(-1) * 0.3
#     modified_vel_w  = new_forward_speed * forward_dir

#     full_vel_w = torch.cat([
#         modified_vel_w,
#         asset.data.root_lin_vel_w[env_ids, 2:3],
#         asset.data.root_ang_vel_w[env_ids, :]
#     ], dim=-1)
    
#     asset.write_root_velocity_to_sim(full_vel_w, env_ids=env_ids)

# def enhanced_tc(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     k_p: float= 1,
#     k_d: float= 0,
#     k_i: float= 1,
#     max_speed_correction: float = 0.2,
#     tc_speed_threshold: float = 1.4,
#     tc_coefficient: float = 0.5,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ):
#     """Stable velocity control with capped acceleration boost"""
#     asset: RigidObject | Articulation = env.scene[asset_cfg.name]
#     dt = torch.tensor(env.step_dt, device=env_ids.device)

#     if CONFIG['env_config']['INCREMENTAL_MODE']:
#         error_velocity_history = env._action_history[env_ids, :, 0]
#     else:
#         error_velocity_history = env._action_history[env_ids, :, 0]*8 - env._base_lin_vel_x_history[env_ids, :]
    
#     error_velocity_p = error_velocity_history[env_ids, 0]
#     error_velocity_d = error_velocity_history[env_ids, 0]-error_velocity_history[env_ids, 1]
#     error_velocity_i = torch.sum(error_velocity_history[env_ids, :])
#     # pid controller to adjust velocity...
#     velocity_adjustment = (k_p * error_velocity_p + k_d * error_velocity_d /dt + k_i * error_velocity_i * dt)*env._base_lin_vel_x_history[env_ids, 0]/10
    
#     velocity_adjustment =   torch.where(
#                             ((env._base_lin_vel_x_history[env_ids, 0] > tc_speed_threshold)).unsqueeze(-1),
#                             torch.clamp(velocity_adjustment, min=0, max=max_speed_correction).unsqueeze(-1),
#                             torch.clamp(velocity_adjustment, min=0, max=max_speed_correction*tc_coefficient).unsqueeze(-1)
#                             )
    
#     if not hasattr(env, '_last_velocity_adjustment'):
#         env._last_velocity_adjustment = torch.zeros(env.num_envs, 
#                                     dtype=torch.float32,
#                                     device=env.device)
#     env._last_velocity_adjustment = velocity_adjustment
    
#     # # Get current velocities in world frame (only x,y components)
#     vel_w = asset.data.root_lin_vel_w[env_ids, :2]    
#     # Get heading angle (yaw) in world frame
#     heading_w = asset.data.heading_w[env_ids]

#     # Create forward direction vector from heading angle
#     forward_dir = torch.stack([torch.cos(heading_w), torch.sin(heading_w)], dim=-1)
#     # Project velocity onto forward direction to get forward speed component
#     forward_speed = torch.sum(vel_w * forward_dir, dim=-1, keepdim=True)
#     # braking = (error_velocity_history[env_ids, 0] -  error_velocity_history[env_ids, 1]) < 0 
#     # above_tc = (error_velocity_history[env_ids, 0] -  error_velocity_history[env_ids, 1]) >= 0 and forward_speed > 0.75
#     # Calculate new velocity by reducing forward speed when decelerating
#     new_forward_speed = torch.where(
#         ((error_velocity_history[env_ids, 0] > 0.5) & (env._base_lin_vel_x_history[env_ids, 0] > 0.5)).unsqueeze(-1),
#         forward_speed + velocity_adjustment,
#         forward_speed
#     )
    
#     new_forward_speed = new_forward_speed * 0.75 + env._base_lin_vel_x_history[env_ids, 0].unsqueeze(-1) * 0.25
#     modified_vel_w  = new_forward_speed * forward_dir

#     full_vel_w = torch.cat([
#         modified_vel_w,
#         asset.data.root_lin_vel_w[env_ids, 2:3],
#         asset.data.root_ang_vel_w[env_ids, :]
#     ], dim=-1)
    
#     asset.write_root_velocity_to_sim(full_vel_w, env_ids=env_ids)

# def enhanced_rotation(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     k_p: float = 1,
#     k_d: float = 0,
#     k_i: float = 1,
#     max_ang_vel: float = 1.8,
#     max_correction: float = 0.75,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ):
#     """Stable velocity control with capped acceleration boost"""
#     asset: RigidObject | Articulation = env.scene[asset_cfg.name]
#     dt = torch.tensor(env.step_dt, device=env_ids.device)

#     command_steering_history = env._action_history[env_ids, :, 1]
#     base_ang_vel_z_history = env._base_ang_vel_z_history[env_ids, :]
    
#     # Current angular velocity
#     current_ang_vel_z = asset.data.root_ang_vel_w[env_ids, 2]
    
#     # Only apply correction when absolute angular velocity is less than 2 rad/s
#     apply_correction = torch.abs(env._base_ang_vel_z_history[env_ids, 1]) < max_ang_vel
    
#     error_p = env._action_history[env_ids, 0, 1]/0.4
#     error_d = (env._action_history[env_ids, 0, 1] - env._action_history[env_ids, 1, 1])/dt/0.4
#     error_i = torch.sum(env._action_history[env_ids, :, 1])/0.4*dt
#     # Calculate adjustment
#     ang_vel_z_adjustment = k_p * error_p + k_d*error_d + k_i *error_i
#     ang_vel_z_adjustment = torch.clamp(ang_vel_z_adjustment, min=-max_correction, max=max_correction)
    
#     # Apply adjustment only where condition is met
#     new_ang_vel_z = torch.where(
#         apply_correction.unsqueeze(-1),
#         current_ang_vel_z.unsqueeze(-1) + ang_vel_z_adjustment.unsqueeze(-1),
#         current_ang_vel_z.unsqueeze(-1)
#     )
    
#     # Apply smoothing filter
#     new_ang_vel_z = new_ang_vel_z * 0.5 + env._base_ang_vel_z_history[env_ids, 0].unsqueeze(-1) * 0.3 + env._base_ang_vel_z_history[env_ids, 1].unsqueeze(-1) * 0.15 + env._base_ang_vel_z_history[env_ids, 2].unsqueeze(-1) * 0.05

#     full_vel_w = torch.cat([
#         asset.data.root_lin_vel_w[env_ids, :],
#         asset.data.root_ang_vel_w[env_ids, 0:2],
#         new_ang_vel_z  # Ensure proper shape
#     ], dim=-1)
    
#     asset.write_root_velocity_to_sim(full_vel_w, env_ids=env_ids)

# def enhanced_vy(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     k_p: float = 1,
#     k_d: float = 0,
#     k_i: float = 1,
#     max_correction: float = 0.2,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ):
#     """Stable velocity control with capped acceleration boost"""
#     asset: RigidObject | Articulation = env.scene[asset_cfg.name]
#     dt = torch.tensor(env.step_dt, device=env_ids.device)

#     command_steering_history = env._action_history[env_ids, :, 1]
#     base_ang_vel_z_history = env._base_ang_vel_z_history[env_ids, :]
#     base_vel_x_history = env._base_lin_vel_x_history[env_ids, :]
    
#     # Current angular velocity
#     current_vel_y = asset.data.root_lin_vel_b[env_ids, 1]
    
#     # Only apply correction when absolute angular velocity is less than 2 rad/s
#     apply_correction = env._base_lin_vel_x_history[env_ids, 0] > 1
    
#     # Calculate adjustment
#     vel_y_adjustment = k_d*(env._action_history[env_ids, 0, 1] - env._action_history[env_ids, 1, 1])/dt/0.4
#     vel_y_adjustment = torch.clamp(vel_y_adjustment, min=-max_correction, max=max_correction)

#     # # Get current velocities in world frame (only x,y components)
#     vel_w = asset.data.root_lin_vel_w[env_ids, :2]    
#     # Get heading angle (yaw) in world frame
#     heading_w = asset.data.heading_w[env_ids]

#     # Create forward direction vector from heading angle
#     lateral_dir = torch.stack([-torch.sin(heading_w), torch.cos(heading_w)], dim=-1)
#     # Project velocity onto forward direction to get forward speed component
#     lateral_speed = torch.sum(vel_w * lateral_dir, dim=-1, keepdim=True)
#     # braking = (error_velocity_history[env_ids, 0] -  error_velocity_history[env_ids, 1]) < 0 
#     # above_tc = (error_velocity_history[env_ids, 0] -  error_velocity_history[env_ids, 1]) >= 0 and forward_speed > 0.75
#     # Calculate new velocity by reducing forward speed when decelerating
#     new_lateral_speed = torch.where(
#         (apply_correction).unsqueeze(-1),
#         lateral_speed + vel_y_adjustment.unsqueeze(-1),
#         lateral_speed
#     )
    
#     new_lateral_speed = new_lateral_speed * 0.90 + env._base_lin_vel_y_history[env_ids, 0].unsqueeze(-1) * 0.3 + env._base_lin_vel_y_history[env_ids, 1].unsqueeze(-1) * 0.1

#     full_vel_w = torch.cat([
#         asset.data.root_lin_vel_w[env_ids, 0:1],
#         new_lateral_speed,  # Update y-velocity
#         asset.data.root_lin_vel_w[env_ids, 2:3],  # Keep
#         asset.data.root_ang_vel_w[env_ids, :]
#     ], dim=-1)
    
#     asset.write_root_velocity_to_sim(full_vel_w, env_ids=env_ids)
     