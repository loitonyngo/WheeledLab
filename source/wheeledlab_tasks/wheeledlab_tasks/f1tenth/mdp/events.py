import torch
import numpy as np
import isaaclab.utils.math as math_utils

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.terrains import TerrainImporter
from ..utils import find_frenet_coord_along_waypoints
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

def reset_root_state_random_opponent(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    # valid_posns_and_rots: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("opponent"),
):
    # access the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                    dtype=torch.long,
                                    device=env.device)
        
    if not hasattr(env, '_prev_delta_s_opp_ego'):
        env._prev_delta_s_opp_ego = torch.ones(env.num_envs, 
                                dtype=torch.float32,
                                device=env.device)*CONFIG['env_config']['OPPONENT_DETECTION_IDX']  
        
    env._map_levels[env_ids] = torch.tensor(np.floor(np.random.rand(len(env_ids))*len(env.cfg.scene.terrain.traversability_hashmap_list)), device = env.device, dtype=torch.long)

    # valid_poses = terrain.cfg.generate_poses_from_init_points(env, env_ids)
    valid_poses, current_idx_np = terrain.cfg.generate_random_poses_from_waypoints(env=env, env_ids=env_ids, num_poses=len(env_ids), max_radius_offset=1.5)
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

    env._prev_delta_s_opp_ego[env_ids]  = torch.ones(len(env_ids), 
                                dtype=torch.float32,
                                device=env.device)*CONFIG['env_config']['OPPONENT_DETECTION_IDX']  
    
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
