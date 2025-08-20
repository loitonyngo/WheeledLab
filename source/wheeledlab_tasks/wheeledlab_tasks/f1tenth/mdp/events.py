import torch
import numpy as np
import isaaclab.utils.math as math_utils

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.terrains import TerrainImporter
from ..utils import find_frenet_coord_along_waypoints
import isaaclab.envs.mdp as mdp

import yaml
with open("/home/tongo/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/f1tenth/config/f1tenth_config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

def reset_root_state_random(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    # valid_posns_and_rots: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # access the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                    dtype=torch.long,
                                    device=env.device)
        
    env._map_levels[env_ids] = torch.tensor(np.floor(np.random.rand(len(env_ids))*len(env.cfg.scene.terrain.map_name_list)), device = env.device, dtype=torch.long)

    # valid_poses = terrain.cfg.generate_poses_from_init_points(env, env_ids)
    valid_poses, current_idx_np = terrain.cfg.generate_random_poses_from_waypoints(env=env, env_ids=env_ids, num_poses=len(env_ids), max_radius_offset=0.5)
    current_idx = torch.tensor(current_idx_np, dtype=torch.long, device=env.device)

    # Tensorizes the valid poses
    posns = torch.stack(list(map(lambda x: torch.tensor(x.pos, device=env.device), valid_poses))).float()
    oris = list(map(lambda x: torch.deg2rad(torch.tensor(x.rot_euler_xyz_deg, device=env.device)), valid_poses))
    oris = torch.stack([math_utils.quat_from_euler_xyz(*ori) for ori in oris]).float()
    lin_vels = torch.stack(list(map(lambda x: torch.tensor(x.lin_vel, device=env.device), valid_poses))).float()
    ang_vels = torch.stack(list(map(lambda x: torch.tensor(x.ang_vel, device=env.device), valid_poses))).float()

    positions = posns
    positions += asset.data.default_root_state[env_ids, :3]
    orientations = oris

    lin_vels = lin_vels
    lin_vels += asset.data.default_root_state[env_ids, 7:10]
    ang_vels = ang_vels
    ang_vels += asset.data.default_root_state[env_ids, 10:13]

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.cat([lin_vels, ang_vels], dim=-1), env_ids=env_ids)

    if not hasattr(env, '_initial_waypoint_indices'):
        env._initial_waypoint_indices = torch.zeros(env.num_envs, 
                                                dtype=torch.long,
                                                device=env.device)
        
    if not hasattr(env, '_reset_env_bool'):
        env._reset_env_bool = torch.ones(  # Tracks where to insert the next index
            env.num_envs,
            dtype=torch.bool,
            device=env.device
        )

    if not hasattr(env, '_progress_history_indices'):
        env._progress_history_length = CONFIG['env_config']['PROGRESS_HISTORY_LENGTH']  # Store last 10 waypoints
        env._progress_history_indices = torch.zeros(
            (env.num_envs, env._progress_history_length), 
            dtype=torch.long,
            device=env.device
        )
    if not hasattr(env, '_action_history_length'):
        env._action_history_length = CONFIG['env_config']['ACTION_HISTORY_LENGTH']

    if not hasattr(env, '_action_history'):
        env._action_history = torch.zeros(
            (env.num_envs, env._action_history_length, 2), 
            dtype=torch.float32,
            device=env.device
        )

    if not hasattr(env, '_obs_history_length'):
        env._obs_history_length = CONFIG['env_config']['OBS_HISTORY_LENGTH']

    if not hasattr(env, '_base_lin_vel_x_history'):
        env._base_lin_vel_x_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )

    if not hasattr(env, '_base_lin_vel_y_history'):
        env._base_lin_vel_y_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )

    if not hasattr(env, '_base_ang_vel_z_history'):
        env._base_ang_vel_z_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )

    if not hasattr(env, '_traversability_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._traversability_history = torch.ones(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )

    if not hasattr(env, '_target_velocity_history'):
        env._target_velocity_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )
        
        
    # set boolean so that it knows it just resetted
    env._initial_waypoint_indices[env_ids] = current_idx.clone()  
    env._reset_env_bool[env_ids] = torch.ones(len(env_ids), dtype=bool, device=env.device)
    env._progress_history_indices[env_ids, :] =  torch.zeros(env._progress_history_length, dtype=torch.long, device=env.device)
    env._action_history[env_ids, :, :] = torch.zeros((len(env_ids), env._action_history_length, 2), dtype=torch.float32, device=env.device)

    env._base_lin_vel_x_history[env_ids, :] = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._base_lin_vel_y_history[env_ids, :] = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._base_ang_vel_z_history[env_ids, :] = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._target_velocity_history[env_ids, :] = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)

    env._traversability_history[env_ids, :] = torch.ones((len(env_ids), env._rew_history_length), dtype=torch.long, device=env.device)

def reset_root_state_random_with_opponent(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    # valid_posns_and_rots: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    opponent_asset_cfg: SceneEntityCfg = SceneEntityCfg("opponent"),
):
    # access the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    opponent_asset: RigidObject | Articulation = env.scene[opponent_asset_cfg.name]

    terrain: TerrainImporter = env.scene.terrain

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                    dtype=torch.long,
                                    device=env.device)
    
    if not hasattr(env, '_prev_delta_s_opp_ego'):
        env._prev_delta_s_opp_ego = torch.ones(env.num_envs, 
                                dtype=torch.float32,
                                device=env.device)*CONFIG['env_config']['OPPONENT_INIT_DISTANCE_IDX']*CONFIG['env_config']['LEN_S_IDX']    
        
    env._map_levels[env_ids] = torch.tensor(np.floor(np.random.rand(len(env_ids))*len(env.cfg.scene.terrain.map_name_list)), device = env.device, dtype=torch.long)

    # valid_poses = terrain.cfg.generate_poses_from_init_points(env, env_ids)
    ego_valid_poses, ego_current_idx_np, opp_valid_poses, opp_current_idx_np = terrain.cfg.generate_random_poses_from_waypoints_with_opponent(env=env, env_ids=env_ids, num_poses=len(env_ids), max_radius_offset=0.5)
    current_idx = torch.tensor(ego_current_idx_np, dtype=torch.long, device=env.device)

    # Tensorizes the valid poses
    posns = torch.stack(list(map(lambda x: torch.tensor(x.pos, device=env.device), ego_valid_poses))).float()
    oris = list(map(lambda x: torch.deg2rad(torch.tensor(x.rot_euler_xyz_deg, device=env.device)), ego_valid_poses))
    oris = torch.stack([math_utils.quat_from_euler_xyz(*ori) for ori in oris]).float()
    lin_vels = torch.stack(list(map(lambda x: torch.tensor(x.lin_vel, device=env.device), ego_valid_poses))).float()
    ang_vels = torch.stack(list(map(lambda x: torch.tensor(x.ang_vel, device=env.device), ego_valid_poses))).float()

    positions = posns
    positions += asset.data.default_root_state[env_ids, :3]
    orientations = oris

    lin_vels = lin_vels
    lin_vels += asset.data.default_root_state[env_ids, 7:10]
    ang_vels = ang_vels
    ang_vels += asset.data.default_root_state[env_ids, 10:13]

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.cat([lin_vels, ang_vels], dim=-1), env_ids=env_ids)
    
    if not hasattr(env, '_initial_waypoint_indices'):
        env._initial_waypoint_indices = torch.zeros(env.num_envs, 
                                                dtype=torch.long,
                                                device=env.device)
        
    if not hasattr(env, '_reset_env_bool'):
        env._reset_env_bool = torch.ones(  # Tracks where to insert the next index
            env.num_envs,
            dtype=torch.bool,
            device=env.device
        )

    if not hasattr(env, '_progress_history_indices'):
        env._progress_history_length = CONFIG['env_config']['PROGRESS_HISTORY_LENGTH']  # Store last 10 waypoints
        env._progress_history_indices = torch.zeros(
            (env.num_envs, env._progress_history_length), 
            dtype=torch.long,
            device=env.device
        )
    if not hasattr(env, '_action_history_length'):
        env._action_history_length = CONFIG['env_config']['ACTION_HISTORY_LENGTH']

    if not hasattr(env, '_action_history'):
        env._action_history = torch.zeros(
            (env.num_envs, env._action_history_length, 2), 
            dtype=torch.float32,
            device=env.device
        )

    if not hasattr(env, '_obs_history_length'):
        env._obs_history_length = CONFIG['env_config']['OBS_HISTORY_LENGTH']

    if not hasattr(env, '_base_lin_vel_x_history'):
        env._base_lin_vel_x_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )

    if not hasattr(env, '_base_lin_vel_y_history'):
        env._base_lin_vel_y_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )

    if not hasattr(env, '_base_ang_vel_z_history'):
        env._base_ang_vel_z_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )

    if not hasattr(env, '_traversability_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._traversability_history = torch.ones(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )

    if not hasattr(env, '_wall_collision_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._wall_collision_history = torch.ones(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
        
    if not hasattr(env, '_target_velocity_history'):
        env._target_velocity_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )
        
        
    # set boolean so that it knows it just resetted
    env._initial_waypoint_indices[env_ids] = current_idx.clone()  
    env._reset_env_bool[env_ids] = torch.ones(len(env_ids), dtype=bool, device=env.device)
    env._progress_history_indices[env_ids, :] =  torch.zeros(env._progress_history_length, dtype=torch.long, device=env.device)
    env._action_history[env_ids, :, :] = torch.zeros((len(env_ids), env._action_history_length, 2), dtype=torch.float32, device=env.device)

    env._base_lin_vel_x_history[env_ids, :] = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._base_lin_vel_y_history[env_ids, :] = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._base_ang_vel_z_history[env_ids, :] = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._target_velocity_history[env_ids, :] = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)

    env._traversability_history[env_ids, :] = torch.ones((len(env_ids), env._rew_history_length), dtype=torch.long, device=env.device)
    env._wall_collision_history[env_ids, :] = torch.zeros((len(env_ids), env._rew_history_length), dtype=torch.long, device=env.device)

    env._prev_delta_s_opp_ego[env_ids]  = torch.ones(len(env_ids), 
                                dtype=torch.float32,
                                device=env.device)*CONFIG['env_config']['OPPONENT_INIT_DISTANCE_IDX']*CONFIG['env_config']['LEN_S_IDX']  

    # Tensorize opponent poses and velocities
    opp_posns = torch.stack(list(map(lambda x: torch.tensor(x.pos, device=env.device), opp_valid_poses))).float()
    opp_oris = torch.stack([
        math_utils.quat_from_euler_xyz(*torch.deg2rad(torch.tensor(x.rot_euler_xyz_deg, device=env.device)))
        for x in opp_valid_poses
    ]).float()
    opp_lin_vels = torch.stack(list(map(lambda x: torch.tensor(x.lin_vel, device=env.device), opp_valid_poses))).float()*0
    opp_ang_vels = torch.stack(list(map(lambda x: torch.tensor(x.ang_vel, device=env.device), opp_valid_poses))).float()

    # Apply opponent's default state offsets (assuming opponent asset has its own default state)
    opp_positions = opp_posns + opponent_asset.data.default_root_state[env_ids, :3]
    opp_orientations = opp_oris
    opp_lin_vels += opponent_asset.data.default_root_state[env_ids, 7:10]
    opp_ang_vels += opponent_asset.data.default_root_state[env_ids, 10:13]
    opponent_asset.write_root_pose_to_sim(torch.cat([opp_positions, opp_orientations], dim=-1), env_ids=env_ids)
    opponent_asset.write_root_velocity_to_sim(torch.cat([opp_lin_vels, opp_ang_vels], dim=-1), env_ids=env_ids)


    if not hasattr(env, '_opponent_type'):
        env._opponent_type = torch.zeros(
            env.num_envs, 
            dtype=torch.long,
            device=env.device
        )
    if not hasattr(env, '_opponent_vel_scaling'):
        env._opponent_vel_scaling = torch.ones(
            env.num_envs, 
            dtype=torch.float16,
            device=env.device
        )        

    if not hasattr(env, '_opponent_overtaken_bool'):
       env._opponent_overtaken_bool = torch.zeros(
            env.num_envs,  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.bool,
            device=env.device
        )
    
    if not hasattr(env, '_opponent_speed'):
       env._opponent_speed = torch.zeros(
            env.num_envs,  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
        )

    if not hasattr(env, '_opponent_d_dot'):
        env._opponent_d_dot = torch.zeros(
            env.num_envs,  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
        )

    if not hasattr(env, '_opponent_heading'):
        env._opponent_heading = torch.zeros(
            env.num_envs,  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
        )
         
    env._opponent_type[env_ids]           = torch.tensor(np.floor(np.random.rand(len(env_ids))*3), device = env.device, dtype=torch.long)
    env._opponent_vel_scaling[env_ids]    = torch.tensor(np.random.rand(len(env_ids)), device = env.device, dtype=torch.float16)
    env._opponent_overtaken_bool[env_ids] = torch.zeros(len(env_ids), dtype=bool, device=env.device)
    env._opponent_speed[env_ids]          = torch.zeros(len(env_ids), dtype=torch.float32, device=env.device)
    env._opponent_d_dot[env_ids]          = torch.zeros(len(env_ids), dtype=torch.float32, device=env.device)
    env._opponent_heading[env_ids]        = torch.zeros(len(env_ids), dtype=torch.float32, device=env.device)

# def reset_root_state_random_opponent(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     # valid_posns_and_rots: dict[str, tuple[float, float]],
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("opponent"),
# ):
#     # access the used quantities (to enable type-hinting)
#     asset: RigidObject | Articulation = env.scene[asset_cfg.name]
#     terrain: TerrainImporter = env.scene.terrain

#     if not hasattr(env, '_map_levels'):
#         env._map_levels = torch.zeros(env.num_envs, 
#                                     dtype=torch.long,
#                                     device=env.device)
        
#     if not hasattr(env, '_prev_delta_s_opp_ego'):
#         env._prev_delta_s_opp_ego = torch.ones(env.num_envs, 
#                                 dtype=torch.float32,
#                                 device=env.device)*CONFIG['env_config']['OPPONENT_INIT_DISTANCE_IDX']*CONFIG['env_config']['LEN_S_IDX']  
        
#     env._map_levels[env_ids] = torch.tensor(np.floor(np.random.rand(len(env_ids))*len(env.cfg.scene.terrain.traversability_hashmap_list)), device = env.device, dtype=torch.long)

#     # valid_poses = terrain.cfg.generate_poses_from_init_points(env, env_ids)
#     valid_poses, current_idx_np = terrain.cfg.generate_random_poses_from_waypoints(env=env, env_ids=env_ids, num_poses=len(env_ids), max_radius_offset=1.5)
#     current_idx = torch.tensor(current_idx_np, dtype=torch.long, device=env.device)

#     # Tensorizes the valid poses
#     posns = torch.stack(list(map(lambda x: torch.tensor(x.pos, device=env.device), valid_poses))).float()
#     oris = list(map(lambda x: torch.deg2rad(torch.tensor(x.rot_euler_xyz_deg, device=env.device)), valid_poses))
#     oris = torch.stack([math_utils.quat_from_euler_xyz(*ori) for ori in oris]).float()
#     lin_vels = torch.stack(list(map(lambda x: torch.tensor(x.lin_vel, device=env.device), valid_poses))).float()
#     ang_vels = torch.stack(list(map(lambda x: torch.tensor(x.ang_vel, device=env.device), valid_poses))).float()

#     positions = posns
#     positions += asset.data.default_root_state[env_ids, :3]
#     orientations = oris

#     lin_vels = lin_vels
#     lin_vels += asset.data.default_root_state[env_ids, 7:10]
#     ang_vels = ang_vels
#     ang_vels += asset.data.default_root_state[env_ids, 10:13]

#     # set into the physics simulation
#     asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
#     asset.write_root_velocity_to_sim(torch.cat([lin_vels, ang_vels], dim=-1), env_ids=env_ids)

#     env._prev_delta_s_opp_ego[env_ids]  = torch.ones(len(env_ids), 
#                                 dtype=torch.float32,
#                                 device=env.device)*CONFIG['env_config']['OPPONENT_INIT_DISTANCE_IDX']*CONFIG['env_config']['LEN_S_IDX']    
    
def reset_root_state_start_idx(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    # valid_posns_and_rots: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # access the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                    dtype=torch.long,
                                    device=env.device)
        
    env._map_levels[env_ids] = torch.tensor(np.floor(np.random.rand(len(env_ids))*len(env.cfg.scene.terrain.traversability_hashmap_list)), device = env.device, dtype=torch.long)

    # valid_poses = terrain.cfg.generate_poses_from_init_points(env, env_ids)
    valid_poses, current_idx_np = terrain.cfg.generate_start_idx_poses(env=env, env_ids=env_ids, num_poses=len(env_ids))
    current_idx = torch.tensor(current_idx_np, dtype=torch.long, device=env.device)

    # Tensorizes the valid poses
    posns = torch.stack(list(map(lambda x: torch.tensor(x.pos, device=env.device), valid_poses))).float()
    oris = list(map(lambda x: torch.deg2rad(torch.tensor(x.rot_euler_xyz_deg, device=env.device)), valid_poses))
    oris = torch.stack([math_utils.quat_from_euler_xyz(*ori) for ori in oris]).float()
    lin_vels = torch.stack(list(map(lambda x: torch.tensor(x.lin_vel, device=env.device), valid_poses))).float()
    ang_vels = torch.stack(list(map(lambda x: torch.tensor(x.ang_vel, device=env.device), valid_poses))).float()

    positions = posns
    positions += asset.data.default_root_state[env_ids, :3]
    orientations = oris

    lin_vels = lin_vels
    lin_vels += asset.data.default_root_state[env_ids, 7:10]
    ang_vels = ang_vels
    ang_vels += asset.data.default_root_state[env_ids, 10:13]

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.cat([lin_vels, ang_vels], dim=-1), env_ids=env_ids)

    if not hasattr(env, '_initial_waypoint_indices'):
        env._initial_waypoint_indices = torch.zeros(env.num_envs, 
                                                dtype=torch.long,
                                                device=env.device)
        
    if not hasattr(env, '_reset_env_bool'):
        env._reset_env_bool = torch.ones(  # Tracks where to insert the next index
            env.num_envs,
            dtype=torch.bool,
            device=env.device
        )

    if not hasattr(env, '_progress_history_indices'):
        env._progress_history_length = CONFIG['env_config']['PROGRESS_HISTORY_LENGTH']  # Store last 10 waypoints
        env._progress_history_indices = torch.zeros(
            (env.num_envs, env._progress_history_length), 
            dtype=torch.long,
            device=env.device
        )

    if not hasattr(env, '_action_history'):
        env.action_history_length = CONFIG['env_config']['ACTION_HISTORY_LENGTH']  # Store last 10 waypoints
        env._action_history = torch.zeros(
            (env.num_envs, env.action_history_length, 2), 
            dtype=torch.float32,
            device=env.device
        )

    if not hasattr(env, '_obs_history_length'):
        env._obs_history_length = CONFIG['env_config']['OBS_HISTORY_LENGTH']
        
    if not hasattr(env, '_base_lin_vel_x_history'):
        env._base_lin_vel_x_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )

    if not hasattr(env, '_base_lin_vel_y_history'):
        env._base_lin_vel_y_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )

    if not hasattr(env, '_base_ang_vel_z_history'):
        env._base_ang_vel_z_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )

    if not hasattr(env, '_traversability_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._traversability_history = torch.ones(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )

    if not hasattr(env, '_target_velocity_history'):
        env._target_velocity_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )
        
    # set boolean so that it knows it just resetted
    env._initial_waypoint_indices[env_ids] = current_idx.clone()  
    env._reset_env_bool[env_ids] = torch.ones(len(env_ids), dtype=bool, device=env.device)
    env._progress_history_indices[env_ids, :] =  torch.zeros(env._progress_history_length, dtype=torch.long, device=env.device)
    env._action_history[env_ids, :, :] = torch.zeros((len(env_ids), env._action_history_length, 2), dtype=torch.float32, device=env.device)

    env._base_lin_vel_x_history[env_ids, :] = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._base_lin_vel_y_history[env_ids, :] = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._base_ang_vel_z_history[env_ids, :] = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)
    env._target_velocity_history[env_ids, :] = torch.zeros((len(env_ids), env._obs_history_length), dtype=torch.float32, device=env.device)

    env._traversability_history[env_ids, :] = torch.ones((len(env_ids), env._rew_history_length), dtype=torch.long, device=env.device)

def move_opponent_vel_based(
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
    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    asset = env.scene[asset_cfg.name]
    opp_position_xy = mdp.root_pos_w(env=env, asset_cfg=SceneEntityCfg("opponent"))[:, :2]
    
    # Initialize velocity tensors
    new_position = torch.zeros_like(opp_position_xy)
    new_lin_velocities = torch.zeros_like(asset.data.root_lin_vel_w)
    new_ang_velocities = torch.zeros_like(asset.data.root_ang_vel_w)
    
    # Get opponent types and map levels for current env_ids
    opponent_types = env._opponent_type
    map_levels = env._map_levels
    unique_map_levels = torch.unique(map_levels)
    
    for map_level in unique_map_levels:
        # Mask for environments in this map level
        map_mask = (map_levels == map_level)
        if not map_mask.any():
            continue
            
        # Get all positions for this map level at once
        current_positions = opp_position_xy[env_ids[map_mask]]
        
        # Get opponent types for this map level
        level_opp_types = opponent_types[map_mask]
        
        # Process each opponent type in this map level
        for opp_type in torch.unique(level_opp_types):
            type_mask = (level_opp_types == opp_type)
            if not type_mask.any():
                continue
                
            # Select the appropriate trajectory
            if opp_type == 0:  # Center
                waypoints = torch.tensor(
                    env.scene.terrain.cfg.opp_traj_center_list[map_level], 
                    device=env.device
                )
            elif opp_type == 1:  # IQP
                waypoints = torch.tensor(
                    env.scene.terrain.cfg.opp_traj_iqp_list[map_level], 
                    device=env.device
                )
            elif opp_type == 2:  # SP
                waypoints = torch.tensor(
                    env.scene.terrain.cfg.opp_traj_sp_list[map_level], 
                    device=env.device
                )
            else:
                raise ValueError(f"Unknown opponent type: {opp_type}")
            
            # Extract waypoint components
            waypoints_xy = waypoints[:, :2]
            waypoints_vel_x = waypoints[:, 3]
            
            # Find closest waypoints for all relevant environments at once
            current_indices, _ = find_frenet_coord_along_waypoints(
                waypoints_xy, 
                current_positions[type_mask]
            )
            
            # Calculate lookahead indices (wrap around if needed)
            lookahead_indices = (current_indices + 5) % len(waypoints_xy)
            
            # Get current and lookahead positions
            current_wp_pos = waypoints_xy[current_indices]
            lookahead_wp_pos = waypoints_xy[lookahead_indices]
            
            # Calculate direction vectors
            direction_vectors = lookahead_wp_pos - current_wp_pos
            
            # Calculate yaw using atan2 (y, x)
            target_yaws = torch.atan2(direction_vectors[:, 1], direction_vectors[:, 0])

            # if num_episodes < 25:
            #     vel_multiplier = 0.4
            # elif num_episodes < 50:
            vel_multiplier = 0.8 + 0.1*np.random.rand() - 0.1*np.random.rand()
                
            # Get target velocities
            # target_vels = waypoints_vel_x[current_indices] * env._opponent_vel_scaling[env_ids[map_mask][type_mask]] * vel_multiplier
            
            if num_episodes % 25:
                vel_multiplier += 0.05
            
            target_vels = waypoints_vel_x[current_indices] * 0.55

            # Convert to world frame velocities
            cos_yaws = torch.cos(target_yaws)
            sin_yaws = torch.sin(target_yaws)
            
            # Get the actual env_ids we're processing
            processing_ids = env_ids[map_mask][type_mask]
            
            # Set velocities
            new_lin_velocities[processing_ids, 0] = target_vels * cos_yaws
            new_lin_velocities[processing_ids, 1] = target_vels * sin_yaws
            new_lin_velocities[processing_ids, 2] = 0
    
    # Write velocities to sim
    asset.write_root_velocity_to_sim(
        torch.cat([new_lin_velocities, new_ang_velocities], dim=1),
        env_ids=env_ids
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
    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
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
        # Mask for environments in this map level (relative to env_ids)
        map_mask = (map_levels == map_level)
        
        if not map_mask.any():
            continue
        
        waypoints_xy_world = torch.tensor(
            env.scene.terrain.cfg.waypoints_list[map_level],
            device=env.device,
            dtype=torch.float32
        )[:, :2]
        
        # Get opponent types for this map level
        level_opp_types = opponent_types[map_mask]
        unique_opp_types = torch.unique(level_opp_types)
        
        for opp_type in unique_opp_types:
            # Mask for environments with this opponent type (relative to map_mask)
            type_mask = (level_opp_types == opp_type)            
            if not type_mask.any():
                continue
                
            # Select the appropriate trajectory
            if opp_type == 0:  # Center
                opp_traj = torch.tensor(
                    env.scene.terrain.cfg.opp_traj_center_list[map_level], 
                    device=env.device
                )
            elif opp_type == 1:  # IQP
                opp_traj = torch.tensor(
                    env.scene.terrain.cfg.opp_traj_iqp_list[map_level], 
                    device=env.device
                )
            elif opp_type == 2:  # SP
                opp_traj = torch.tensor(
                    env.scene.terrain.cfg.opp_traj_sp_list[map_level], 
                    device=env.device
                )
            else:
                raise ValueError(f"Unknown opponent type: {opp_type}")
            
            # waypoints = torch.tensor(
            #         env.scene.terrain.cfg.opp_traj_center_list[map_level], 
            #         device=env.device
            #     )
            opp_traj_xy = opp_traj[:, :2]
            opp_traj_psi = opp_traj[:, 2]
            opp_traj_vel_x = opp_traj[:, 3]
            
            ego_current_indices, _ = find_frenet_coord_along_waypoints(
                opp_traj_xy, 
                ego_position_xy[type_mask]  # Only positions for current_type_env_ids
            )
            
            # Find closest waypoints for all relevant environments at once
            opp_current_indices, _ = find_frenet_coord_along_waypoints(
                opp_traj_xy, 
                opp_position_xy[type_mask]  # Only positions for current_type_env_ids
            )

            _, opp_current_d = find_frenet_coord_along_waypoints(
                waypoints_xy_world, 
                opp_position_xy[type_mask]  # Only positions for current_type_env_ids
            )
            
            # Calculate next waypoint indices (with wrapping)
            # move_steps = (opp_traj_vel_x[opp_current_indices] * CONFIG['env_config']['OPP_MOVE_DT'] * 0.7 / CONFIG['env_config']['LEN_S_IDX']).int()

            # if num_episodes < 25:
            next_indices_traj = (ego_current_indices + CONFIG['env_config']['OPPONENT_INIT_DISTANCE_IDX'] ) % len(opp_traj_xy)
            # else:
            #     # move_steps = 1
            # next_indices_traj = (opp_current_indices + move_steps + np.random.randint(low=0, high=2)) % len(opp_traj_xy)
            
            # Update positions (X,Y only; preserve Z)
            new_positions[type_mask, 0] = opp_traj_xy[next_indices_traj, 0] + env.scene.env_origins[type_mask, 0] + torch.rand_like(opp_traj_xy[next_indices_traj, 0])*0.05
            new_positions[type_mask, 1] = opp_traj_xy[next_indices_traj, 1] + env.scene.env_origins[type_mask, 1] + torch.rand_like(opp_traj_xy[next_indices_traj, 1])*0.05
            
            env._opponent_speed[type_mask] = torch.norm(opp_traj_xy[next_indices_traj]-opp_position_xy[type_mask], p=2, dim=1)/CONFIG['env_config']['OPP_MOVE_DT']
            # env._opponent_heading = torch.atan2(opp_traj_xy[next_indices_traj, 1] - opp_position_xy[type_mask, 1], 
            #                                     opp_traj_xy[next_indices_traj, 0] - opp_position_xy[type_mask, 0])

            _, opp_next_d = find_frenet_coord_along_waypoints(
                waypoints_xy_world, 
                opp_traj_xy[next_indices_traj]  # Only positions for current_type_env_ids
            )
            
            env._opponent_d_dot[type_mask] = (opp_next_d - opp_current_d)/CONFIG['env_config']['OPP_MOVE_DT']
            
            # Convert yaw (psi) to quaternion for orientation
            
                #         if opp_type == 0:
                # a = 1
            target_yaw = opp_traj_psi[opp_current_indices]
            new_orientations[type_mask] = torch.stack([
                torch.ones_like(target_yaw),         # Quaternion w
                torch.zeros_like(target_yaw),        # Quaternion x
                torch.zeros_like(target_yaw),        # Quaternion y
                torch.zeros_like(target_yaw)         # Quaternion z
            ], dim=1)
    
    
    # Apply all updates at once
    asset.write_root_pose_to_sim(
        torch.cat([new_positions, new_orientations], dim=1), 
        env_ids=env_ids
    )
    
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
     