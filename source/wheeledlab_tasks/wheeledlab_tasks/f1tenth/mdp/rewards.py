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
    

def wall_collision_penalty(env):
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
    num_envs = pos_xy_world.shape[0]

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
    if not hasattr(env, '_outer_list'):
        env._outer_list = [
            torch.tensor(outer, device=env.device, dtype=torch.float32)
            for outer in env.scene.terrain.cfg.outer_list
        ]
    if not hasattr(env, '_inner_list'):
        env._inner_list = [
            torch.tensor(inner, device=env.device, dtype=torch.float32)
            for inner in env.scene.terrain.cfg.inner_list
        ]        
        
    # Get map levels for all environments
    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)

    collision_bool = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
    
    for map_level in unique_map_levels:
        # Create mask for environments using this map
        env_mask = (map_levels == map_level)
        num_envs_in_map = env_mask.sum()
        
        if num_envs_in_map == 0:
            continue
            
        # Get positions for these environments
        map_positions = pos_xy_world[env_mask]
        
        # Get waypoints for this map level
        inner_xy_world = env._inner_list[map_level][:, :2]
        outer_xy_world = env._outer_list[map_level][:, :2]

        # Find nearest waypoint for these environments
        nearest_to_inner_idx, dist_from_inner = find_frenet_coord_along_waypoints(inner_xy_world, map_positions)  # shape: [num_envs_in_map]
        nearest_to_outer_idx, dist_from_outer = find_frenet_coord_along_waypoints(outer_xy_world, map_positions)  # shape: [num_envs_in_map]


        # Check for collisions with inner and outer bounds
        collision_with_inner = abs(dist_from_inner) < CONFIG['env_config']['WALL_COLLISION_RADIUS']
        collision_with_outer = abs(dist_from_outer) < CONFIG['env_config']['WALL_COLLISION_RADIUS']
        
        # Combine collisions (OR operation - collision with either counts)
        collisions_in_map = collision_with_inner | collision_with_outer
        
        # Convert collisions to long dtype (0 or 1)
        collisions_in_map = collisions_in_map.long()
        
        # Update the collision_bool tensor for these environments
        collision_bool[env_mask] = collisions_in_map

    # vel_x = mdp.base_lin_vel(env)[:, 0]
    return torch.where(collision_bool.bool(), -1, 0)

def opponent_mean_delta_speed(env):
    # num_episodes = env.common_step_counter // env.max_episode_length
    # if num_episodes < CONFIG['env_config']['IGNORE_OPPONENT_UNTIL_EP']:
    #     return torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
   
    if not hasattr(env, '_vx_diff_history'):
        env._vx_diff_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )
        
    mean_vx_diff = torch.mean(env._vx_diff_history, dim=1)

    return -mean_vx_diff


def opponent_collision_penalty(env):
    # num_episodes = env.common_step_counter // env.max_episode_length
    # if num_episodes < CONFIG['env_config']['IGNORE_OPPONENT_UNTIL_EP']:
    #     return torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
   
    if not hasattr(env, '_opponent_vel_scaling_lvl'):
       env._opponent_vel_scaling_lvl = torch.zeros(
            env.num_envs,  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
        )       
    if not hasattr(env, '_s_idx_diff_history'):
        env._s_idx_diff_history = torch.ones(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )
    if not hasattr(env, '_d_diff_history'):
        env._d_diff_history = torch.ones(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )
        
    ego_position_xy = env.scene["robot"].data.root_pos_w[:, :2]
    opp_position_xy = env.scene["opponent"].data.root_pos_w[:, :2]
    dist = torch.norm(ego_position_xy - opp_position_xy, p=2, dim=1)
    
    opp_collision = (
        (abs(env._s_idx_diff_history[:, 0]) < CONFIG['env_config']['OPP_FRONT_COLLISION_RADIUS']) & 
        (abs(env._d_diff_history[:, 0]) < CONFIG['env_config']['OPP_LAT_COLLISION_RADIUS'])
    )
        
    return torch.where(opp_collision.bool(), -1, 0)


def opponent_overtake_delta_distance_reward(env):
    # num_episodes = env.common_step_counter // env.max_episode_length
    # if num_episodes < CONFIG['env_config']['IGNORE_OPPONENT_UNTIL_EP']:
    #     return torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    
    if not hasattr(env, '_prev_delta_s_opp_ego'):
        env._prev_delta_s_opp_ego = torch.zeros(env.num_envs, 
                                dtype=torch.float32,
                                device=env.device)*CONFIG['env_config']['OPPONENT_INIT_DISTANCE_IDX_MIN']*CONFIG['env_config']['LEN_S_IDX']  
        
    ego_position_xy = mdp.root_pos_w(env = env, asset_cfg = SceneEntityCfg("robot"))[..., :2]
    opp_position_xy = mdp.root_pos_w(env = env, asset_cfg = SceneEntityCfg("opponent"))[:, :2]
    dist = torch.norm(ego_position_xy - opp_position_xy, p=2, dim=1)

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
    if not hasattr(env, '_progress_history_indices'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']  # Store last 10 waypoints

        env._progress_history_indices = torch.zeros(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
        env._reset_env_bool = torch.ones(  # Tracks where to insert the next index
            env.num_envs,
            dtype=torch.bool,
            device=env.device
        )

    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)
    
    # Initialize outputs
    delta_s_opp_ego       = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    delta_delta_s_opp_ego = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # ego_behind_opp_bool  = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # current_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # Process each map level separately
    for map_level in unique_map_levels:
        env_mask = (map_levels == map_level)
        num_envs_in_map = env_mask.sum()
        
        if num_envs_in_map == 0:
            continue
            
        # Get waypoints for this map level
        waypoints_world = env._waypoints_list[map_level][:, :2]
        inner_xy_world = env._inner_list[map_level][:, :2]
        num_waypoints = len(waypoints_world)

        # Use the inner so that taking the inside is incentivezed ?
        ego_current_idx, _ = find_frenet_coord_along_waypoints(
            waypoints_world, 
            ego_position_xy[env_mask]
        )
        opp_current_idx, _ = find_frenet_coord_along_waypoints(
            waypoints_world, 
            opp_position_xy[env_mask]
        )

        delta_s_opp_ego[env_mask] = ((opp_current_idx-ego_current_idx + num_waypoints//2) % num_waypoints - num_waypoints // 2).float()
        delta_delta_s_opp_ego[env_mask] = env._prev_delta_s_opp_ego[env_mask] - delta_s_opp_ego[env_mask]*CONFIG['env_config']['LEN_S_IDX']  

        # ego_behind_opp_bool[env_mask] = delta_s_opp_ego[env_mask] > 0
        env._prev_delta_s_opp_ego[env_mask] = delta_s_opp_ego[env_mask]*CONFIG['env_config']['LEN_S_IDX']  

    if not hasattr(env, '_wall_collision_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._wall_collision_history = torch.zeros(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
    no_off_track = env._wall_collision_history.max(dim=1).values == 0
    if not hasattr(env, '_opponent_collision_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._opponent_collision_history = torch.zeros(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
    no_opp_collision = env._opponent_collision_history.max(dim=1).values == 0
    
    # New: Give reward for both closing gap AND extending lead
    reward = torch.where(
        delta_s_opp_ego < 0,  # If ego is ahead
        -delta_delta_s_opp_ego,  # Positive reward for increasing lead (since delta_s_opp_ego becomes more negative)
        delta_delta_s_opp_ego  # Original reward for closing gap
    )
    
    return torch.where(no_off_track & no_opp_collision, reward*CONFIG['env_config']['LEN_S_IDX'], 0)

def opponent_overtake_distance_reward(env):
    # num_episodes = env.common_step_counter // env.max_episode_length
    # if num_episodes < CONFIG['env_config']['IGNORE_OPPONENT_UNTIL_EP']:
    #     return torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    ego_position_xy = mdp.root_pos_w(env = env, asset_cfg = SceneEntityCfg("robot"))[..., :2]
    opp_position_xy = mdp.root_pos_w(env = env, asset_cfg = SceneEntityCfg("opponent"))[:, :2]
    dist = torch.norm(ego_position_xy - opp_position_xy, p=2, dim=1)

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
    if not hasattr(env, '_waypoints_list'):
        env._waypoints_list = [
            torch.tensor(wps, device=env.device, dtype=torch.float32)
            for wps in env.scene.terrain.cfg.waypoints_list
    ]
    if not hasattr(env, '_inner_list'):
        env._inner_list = [
            torch.tensor(inner, device=env.device, dtype=torch.float32)
            for inner in env.scene.terrain.cfg.inner_list
        ]
    
    if not hasattr(env, '_progress_history_indices'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']  # Store last 10 waypoints

        env._progress_history_indices = torch.zeros(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
        env._reset_env_bool = torch.ones(  # Tracks where to insert the next index
            env.num_envs,
            dtype=torch.bool,
            device=env.device
        )

    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)
    
    # Initialize outputs
    delta_s_opp_ego       = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # ego_behind_opp_bool  = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # current_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # Process each map level separately
    for map_level in unique_map_levels:
        env_mask = (map_levels == map_level)
        num_envs_in_map = env_mask.sum()
        
        if num_envs_in_map == 0:
            continue
            
        # Get waypoints for this map level
        waypoints_world = env._waypoints_list[map_level][:, :2]
        inner_xy_world = env._inner_list[map_level][:, :2]
        num_waypoints = len(waypoints_world)

        # Use the inner so that taking the inside is incentivezed ?
        ego_current_idx, _ = find_frenet_coord_along_waypoints(
            waypoints_world, 
            ego_position_xy[env_mask]
        )
        opp_current_idx, _ = find_frenet_coord_along_waypoints(
            waypoints_world, 
            opp_position_xy[env_mask]
        )

        opp_overtaken_idx = opp_current_idx+CONFIG['env_config']['OPPONENT_OVERTAKEN_IDX']
        delta_s_opp_ego[env_mask] = ((opp_overtaken_idx-ego_current_idx + num_waypoints//2) % num_waypoints - num_waypoints // 2).float()

        # ego_behind_opp_bool[env_mask] = delta_s_opp_ego[env_mask] > 0
        # env._prev_delta_s_opp_ego[env_mask] = delta_s_opp_ego[env_mask]*CONFIG['env_config']['LEN_S_IDX']  

    if not hasattr(env, '_wall_collision_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._wall_collision_history = torch.zeros(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
    no_off_track = env._wall_collision_history.max(dim=1).values == 0
    if not hasattr(env, '_opponent_collision_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._opponent_collision_history = torch.zeros(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
    no_opp_collision = env._opponent_collision_history.max(dim=1).values == 0

    return torch.where(no_off_track & no_opp_collision, -delta_s_opp_ego*CONFIG['env_config']['LEN_S_IDX'], 0)

def opponent_overtake_completed_reward(env):
    # num_episodes = env.common_step_counter // env.max_episode_length
    # if num_episodes < CONFIG['env_config']['IGNORE_OPPONENT_UNTIL_EP']:
    #     return torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    ego_position_xy = mdp.root_pos_w(env = env, asset_cfg = SceneEntityCfg("robot"))[..., :2]
    opp_position_xy = mdp.root_pos_w(env = env, asset_cfg = SceneEntityCfg("opponent"))[:, :2]
    dist = torch.norm(ego_position_xy - opp_position_xy, p=2, dim=1)

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
        
    if not hasattr(env, '_progress_history_indices'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']  # Store last 10 waypoints

        env._progress_history_indices = torch.zeros(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
        env._reset_env_bool = torch.ones(  # Tracks where to insert the next index
            env.num_envs,
            dtype=torch.bool,
            device=env.device
        )
        
    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)
    
    # Initialize outputs
    delta_s_opp_ego       = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    ego_progress          = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # ego_behind_opp_bool  = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # current_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # Process each map level separately
    for map_level in unique_map_levels:
        env_mask = (map_levels == map_level)
        num_envs_in_map = env_mask.sum()
        
        if num_envs_in_map == 0:
            continue
            
        # Get waypoints for this map level
        waypoints_world = env._waypoints_list[map_level][:, :2]
        inner_xy_world = env._inner_list[map_level][:, :2]
        num_waypoints = len(waypoints_world)

        
        # Use the inner so that taking the inside is incentivezed ?
        ego_current_idx, _ = find_frenet_coord_along_waypoints(
            waypoints_world, 
            ego_position_xy[env_mask]
        )
        opp_current_idx, _ = find_frenet_coord_along_waypoints(
            waypoints_world, 
            opp_position_xy[env_mask]
        )

        delta_s_opp_ego[env_mask] = ((opp_current_idx-ego_current_idx + num_waypoints//2) % num_waypoints - num_waypoints // 2).float()
        ego_progress[env_mask] = (ego_current_idx - env._progress_history_indices[env_mask, env._progress_history_checkpoint_idx]) % num_waypoints
        # ego_behind_opp_bool[env_mask] = delta_s_opp_ego[env_mask] > 0
        # env._prev_delta_s_opp_ego[env_mask] = delta_s_opp_ego[env_mask]*CONFIG['env_config']['LEN_S_IDX']  

    if not hasattr(env, '_wall_collision_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._wall_collision_history = torch.zeros(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
    no_off_track = env._wall_collision_history.max(dim=1).values == 0
    if not hasattr(env, '_opponent_collision_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._opponent_collision_history = torch.zeros(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
    no_opp_collision = env._opponent_collision_history.max(dim=1).values == 0
    
    env._opponent_overtaken_bool = torch.where(
        (delta_s_opp_ego < -CONFIG['env_config']['OPPONENT_OVERTAKEN_IDX']) & no_off_track & no_opp_collision & (ego_progress <= CONFIG['env_config']['MAX_PROGRESS_IDX']),
        torch.ones(env.num_envs, device=env.device, dtype=bool),
        torch.zeros(env.num_envs, device=env.device, dtype=bool)
    )

        
    if not hasattr(env, '_opponent_overtaken_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._opponent_overtaken_history = torch.zeros(
            (env.num_envs, env._rew_history_length),
            dtype=torch.long,
            device=env.device,
        )
        
    env._opponent_overtaken_history[:, 1:] = env._opponent_overtaken_history[:, :-1].clone()
    env._opponent_overtaken_history[:, 0] = env._opponent_overtaken_bool
    
    overtake_completed = env._opponent_overtaken_history.min(dim=1).values == 1
    env.extras['log']['Info/opponent_vel_scaling'] = env._opponent_vel_scaling_lvl
    if overtake_completed.any() if hasattr(overtake_completed, 'any') else len(overtake_completed) > 0:
        env.extras['log']['Info/opponent_vel_scaling_max_overtaken'] = max(env._opponent_vel_scaling[overtake_completed])
    else:
        env.extras['log']['Info/opponent_vel_scaling_max_overtaken'] = 0.0  # or some default value    env.extras['log']['Info/opponent_overtaken_counter'] = env._opponent_overtaken_counter 
    env.extras['log']['Info/opponent_overtaken_step'] = torch.sum(overtake_completed.float())
    env.extras['log']['Info/opponent_collision_counter'] = env._opponent_collision_counter 
    env.extras['log']['Info/opponent_collision_step'] = torch.sum(env._opponent_collision_history[:, CONFIG['env_config']['OPPONENT_COLLISION_CHECK_IDX']])
    env.extras['log']['Info/opponent_overtaken_collision_ratio'] = env._opponent_overtaken_counter/(env._opponent_collision_counter+env._opponent_overtaken_counter+env._wall_collision_counter+1)
    env.extras['log']['Info/opponent_overtaken_collision_ratio_step'] = torch.sum(overtake_completed.float())/(torch.sum(env._opponent_collision_history[:, CONFIG['env_config']['OPPONENT_COLLISION_CHECK_IDX']])+torch.sum(overtake_completed.float())+torch.sum(env._wall_collision_history[:, CONFIG['env_config']['WALL_COLLISION_CHECK_IDX']].float())+1)

    env.extras['delta_s_opp_ego'] = delta_s_opp_ego[0]
    
    return overtake_completed.float()*env._opponent_vel_scaling

def opponent_overtake_positioning_reward(env):
    num_episodes = env.common_step_counter // env.max_episode_length  
    if num_episodes < CONFIG['env_config']['IGNORE_OPPONENT_UNTIL_EP']:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    
    if not hasattr(env, '_prev_delta_s_opp_ego'):
        env._prev_delta_s_opp_ego = torch.zeros(env.num_envs, 
                                dtype=torch.float32,
                                device=env.device)
        
    ego_position_xy = mdp.root_pos_w(env = env, asset_cfg = SceneEntityCfg("robot"))[..., :2]
    opp_position_xy = mdp.root_pos_w(env = env, asset_cfg = SceneEntityCfg("opponent"))[:, :2]
    
    ego_heading_w = env.scene["robot"].data.heading_w
    ego_heading_vec = torch.stack([torch.cos(ego_heading_w), torch.sin(ego_heading_w)], dim=1)

    dist = torch.norm(ego_position_xy - opp_position_xy, p=2, dim=1)

    ego_opp_vec = (opp_position_xy - ego_position_xy)
    
    ego_opp_positioning_vec = (ego_heading_vec[:, 0] * ego_opp_vec[:, 1] 
                            - ego_heading_vec[:, 1] * ego_opp_vec[:, 0]) / (dist + 1)
    ego_opp_positioning_vec_norm = torch.clamp(ego_opp_positioning_vec, -.5, .5)
    
    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
    if not hasattr(env, '_progress_history_indices'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']  # Store last 10 waypoints

        env._progress_history_indices = torch.zeros(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
        env._reset_env_bool = torch.ones(  # Tracks where to insert the next index
            env.num_envs,
            dtype=torch.bool,
            device=env.device
        )
    if not hasattr(env, '_progress_history_checkpoint_idx'):
        env._progress_history_checkpoint_idx = CONFIG['env_config']['PROGRESS_HISTORY_CHECK_IDX']
    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)
    
    # Initialize outputs
    delta_s_opp_ego = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    delta_d_opp_ego = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    ego_not_ahead_bool  = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    far_way_opp_bool   = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    ego_progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # current_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # Process each map level separately
    for map_level in unique_map_levels:
        env_mask = (map_levels == map_level)
        num_envs_in_map = env_mask.sum()
        
        if num_envs_in_map == 0:
            continue
            
        # Get waypoints for this map level
        waypoints_world = env._waypoints_list[map_level][:, :2]
        num_waypoints = len(waypoints_world)

        # Find nearest waypoint for these environments
        ego_current_idx, ego_current_d = find_frenet_coord_along_waypoints(
            waypoints_world, 
            ego_position_xy[env_mask]
        )
        opp_current_idx, opp_current_d = find_frenet_coord_along_waypoints(
            waypoints_world, 
            opp_position_xy[env_mask]
        )

        delta_s_opp_ego[env_mask] = ((opp_current_idx-ego_current_idx + num_waypoints//2) % num_waypoints - num_waypoints // 2).float()
        delta_d_opp_ego[env_mask] = abs(ego_current_d-opp_current_d).float()
        
        ego_not_ahead_bool[env_mask] = (delta_s_opp_ego[env_mask]+5) > 0
        far_way_opp_bool[env_mask]    = abs(delta_s_opp_ego[env_mask]) > CONFIG['env_config']['OPPONENT_FAR_AWAY_IDX']
        # env._prev_delta_s_opp_ego[env_mask] = delta_s_opp_ego[env_mask]*CONFIG['env_config']['LEN_S_IDX']  

        # Calculate progress
        ego_progress[env_mask] = (ego_current_idx - env._progress_history_indices[env_mask, env._progress_history_checkpoint_idx]) % num_waypoints
        
        # Calculate progress bool
    
    
    if not hasattr(env, '_wall_collision_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._wall_collision_history = torch.zeros(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
    no_off_track = env._wall_collision_history.max(dim=1).values == 0
    if not hasattr(env, '_opponent_collision_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._opponent_collision_history = torch.zeros(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
    no_opp_collision = env._opponent_collision_history.max(dim=1).values == 0
    
    # positioning_penalty = (1.0 - torch.abs(ego_opp_positioning_vec_norm))*torch.exp(-delta_s_opp_ego*CONFIG['env_config']['LEN_S_IDX'])
    # positioning_reward  = torch.abs(ego_opp_positioning_vec_norm)*torch.exp(-((delta_s_opp_ego)**2)*CONFIG['env_config']['LEN_S_IDX']* 2)
    positioning_reward  = torch.abs(delta_d_opp_ego)*torch.exp(-((delta_s_opp_ego)**2)*CONFIG['env_config']['LEN_S_IDX']* 2)
    # behind_mask = delta_s_opp_ego > 0
    # ahead_mask  = delta_s_opp_ego <= 0
    # When behind → reward lateral offset (encourage overtaking attempt)
    # reward_behind = torch.abs(ego_opp_positioning_vec_norm) * torch.exp(-((delta_s_opp_ego)**2) * CONFIG['env_config']['LEN_S_IDX']*0.5)

    # # When ahead → reward staying centered in front (ego_opp_positioning_vec_norm ≈ 0)
    # reward_ahead = (1.0 - torch.abs(ego_opp_positioning_vec_norm)) * torch.exp(-((delta_s_opp_ego)**2) * CONFIG['env_config']['LEN_S_IDX']*0.5)

    # # Combine
    # positioning_reward = torch.where(behind_mask, reward_behind, reward_ahead)

    return torch.where(no_off_track & no_opp_collision & (ego_progress <= CONFIG['env_config']['MAX_PROGRESS_IDX']), positioning_reward, 0)

def upright_penalty(env, thresh_deg):
    rot_mat = math_utils.matrix_from_quat(mdp.root_quat_w(env))
    up_dot = rot_mat[:, 2, 2]
    up_dot = torch.rad2deg(torch.arccos(up_dot))
    penalty = torch.where(up_dot > thresh_deg, up_dot - thresh_deg, 0.)
    return penalty

def forward_vel(env):
    return mdp.base_lin_vel(env)[:, 0]

def progress_rew(env):
    """Reward for passing each new waypoint, handling lap transitions."""
    progress_bool, progress = progress_waypoint_bool(env)

    if not hasattr(env, '_wall_collision_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._wall_collision_history = torch.zeros(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
    no_off_track = env._wall_collision_history.max(dim=1).values == 0
    if not hasattr(env, '_opponent_collision_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._opponent_collision_history = torch.zeros(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
    no_opp_collision = env._opponent_collision_history.max(dim=1).values == 0
    
    return torch.where(progress_bool & no_off_track & no_opp_collision, progress*CONFIG['env_config']['LEN_S_IDX'], 0.0)

def progress_waypoint_bool(env):
    # Initialize buffer if first run
    if not hasattr(env, '_progress_history_indices'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']  # Store last 10 waypoints

        env._progress_history_indices = torch.zeros(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
        env._reset_env_bool = torch.ones(  # Tracks where to insert the next index
            env.num_envs,
            dtype=torch.bool,
            device=env.device
        )
    if not hasattr(env, '_progress_history_checkpoint_idx'):
        env._progress_history_checkpoint_idx = CONFIG['env_config']['PROGRESS_HISTORY_CHECK_IDX']
    if not hasattr(env, '_total_progress_indices'):
        env._total_progress_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # Get current positions and map levels
    position_xy_world = mdp.root_pos_w(env)[..., :2]
    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
    if not hasattr(env, '_waypoints_list'):
        env._waypoints_list = [
            torch.tensor(wps, device=env.device, dtype=torch.float32)
            for wps in env.scene.terrain.cfg.waypoints_list
        ]
    if not hasattr(env, '_initial_waypoint_indices'):
        env._initial_waypoint_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, '_total_progress_indices'):
        env._total_progress_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, '_wall_collision_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._wall_collision_history = torch.zeros(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
    no_off_track = env._wall_collision_history.max(dim=1).values == 0  
    
    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)
    
    # Initialize outputs
    progress_bool = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    # current_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    

         
    # Process each map level separately
    for map_level in unique_map_levels:
        env_mask = (map_levels == map_level)
        num_envs_in_map = env_mask.sum()
        
        if num_envs_in_map == 0:
            continue
        
        
        # Get waypoints for this map level
        waypoints_world = env._waypoints_list[map_level][:, :2]

        # Find nearest waypoint for these environments
        current_idx, _ = find_frenet_coord_along_waypoints(
            waypoints_world, 
            position_xy_world[env_mask]
        )
        
        num_waypoints = len(waypoints_world)
        
        # Store current indices and num_waypoints
        # current_indices[env_mask] = current_idx
        
        # Update history for these environments
        env._progress_history_indices[env_mask, 1:] = env._progress_history_indices[env_mask, :-1].clone()
        env._progress_history_indices[env_mask, 0] = current_idx
        
        # Calculate progress
        current_progress = (current_idx - env._progress_history_indices[env_mask, env._progress_history_checkpoint_idx]) % num_waypoints
        progress[env_mask] = current_progress
        # env._total_progress_indices[env_mask] += torch.where(no_off_track,
        #                                                     current_progress,
        #                                                     0)
        # Calculate progress bool
        progress_bool[env_mask] = (current_progress > 0) & (current_progress <= CONFIG['env_config']['MAX_PROGRESS_IDX']) & (env._reset_env_bool[env_mask] == False)
    
    # Reset flags
    env._reset_env_bool = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    asset = env.scene["robot"]
    
    if not hasattr(env, '_vel_y_calc'):
        env._vel_y_calc = torch.zeros(env.num_envs, 
                                        dtype=torch.float32,
                                        device=env.device)    
    
    if not hasattr(env, '_target_velocity_history'):
        env._obs_history_length = CONFIG['env_config']['OBS_HISTORY_LENGTH']
        env._target_velocity_history = torch.zeros(
            (env.num_envs, env._obs_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
            )
        env._target_velocity_history = torch.zeros(
            (env.num_envs, env._obs_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
            )
        
    env._vel_y_calc =  mdp.base_lin_vel(env)[:, 1]*mdp.base_lin_vel(env)[:, 0]*1.2
    
    ###########################
    # Store extras (using first map's waypoints count for simplicity), only necessary when you play policy, find a better way to implement it
    # env.extras['inner'] =  torch.tensor(env.scene.terrain.cfg.inner_list[map_level][current_idx], device=env.device)
    # env.extras['outer'] = torch.tensor(env.scene.terrain.cfg.outer_list[map_level][current_idx], device=env.device)
    env.extras['theta'] = asset.data.heading_w
    env.extras['pos_xy'] = position_xy_world
    env.extras['vel_x'] = asset.data.root_lin_vel_b[:,0]
    env.extras['vel_y'] = -asset.data.root_lin_vel_b[:,1]
    env.extras['target_velocity'] = env._target_velocity_history[:, 0]
    env.extras['target_steering'] = env._target_steering_angle_history[:, 0]

    env.extras['yaw_rate'] = asset.data.root_ang_vel_b[:,2]
    env.extras['s_idx'] = current_idx.clone()
    env.extras['time'] = torch.tensor(env.sim.current_time, device=env.device)
    env.extras['s_idx_max'] = torch.tensor(num_waypoints, device=env.device)
    env.extras['throttle_joints_applied_effort'] = asset.actuators['throttle_joints'].applied_effort
    ###########################
    
    env.extras['vel_y_calc'] = env._vel_y_calc

    env.extras['log']['Info/mean_speed'] = torch.mean(env._base_lin_vel_x_history)

    return progress_bool, progress


def negative_throttle_penalty(env):
    # last_throttle_action = mdp.last_action(env)[..., 0]*env.cfg.actions.throttle_steer.scale[0]
    last_throttle_action = mdp.last_action(env)[..., 0]

    return torch.where(last_throttle_action < 0., -1, 0.) # speed target

def var_throttle_penalty(env):
    # last_throttle_action = mdp.last_action(env)[..., 0]*env.cfg.actions.throttle_steer.scale[0]
    last_throttle_action = mdp.last_action(env)[..., 0]
    if not hasattr(env, '_action_history'):
        env._action_history_length = CONFIG['env_config']['ACTION_HISTORY_LENGTH']
        env._action_history = torch.zeros(
            (env.num_envs, env._action_history_length, 2),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
            )
    var_throttle = torch.var(env._action_history[:, :, 0], dim=1)
    
    return -var_throttle # speed target

def var_steering_penalty(env):

    if not hasattr(env, '_action_history'):
        env._action_history_length = CONFIG['env_config']['ACTION_HISTORY_LENGTH']
        env._action_history = torch.zeros(
            (env.num_envs, env._action_history_length, 2),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
            )
    var_steering = torch.var(env._action_history[:, :, 1], dim=1)
    
    return -var_steering # speed target

def var_throttle_rate_penalty(env):

    if not hasattr(env, '_action_history'):
        env._action_history_length = CONFIG['env_config']['ACTION_HISTORY_LENGTH']
        env._action_history = torch.zeros(
            (env.num_envs, env._action_history_length, 2),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
            )
    throttle_rate = torch.diff(env._action_history[:, :, 0], dim=1)
    var_throttle_rate = torch.var(throttle_rate, dim=1)

    return -var_throttle_rate # speed target

def delta_throttle_l2_penalty(env):
    # last_throttle_action = mdp.last_action(env)[..., 0]*env.cfg.actions.throttle_steer.scale[0]
    last_throttle_action = mdp.last_action(env)[..., 0]
    if not hasattr(env, '_action_history'):
        env._action_history_length = CONFIG['env_config']['ACTION_HISTORY_LENGTH']
        env._action_history = torch.zeros(
            (env.num_envs, env._action_history_length, 2),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
            )
        
    delta_throttle_l2 = -((env._action_history[:, 0, 0] - env._action_history[:, 1, 0]))**2
    
    return delta_throttle_l2 # speed target

def delta_steering_l2_penalty(env):
    # last_throttle_action = mdp.last_action(env)[..., 0]*env.cfg.actions.throttle_steer.scale[0]
    last_steering_action = mdp.last_action(env)[..., 1]
    if not hasattr(env, '_action_history'):
        env._action_history_length = CONFIG['env_config']['ACTION_HISTORY_LENGTH']
        env._action_history = torch.zeros(
            (env.num_envs, env._action_history_length, 2),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
            )
        
    delta_steering_l2_penalty = -((env._action_history[:, 0, 1] - env._action_history[:, 1, 1]))**2
    
    return delta_steering_l2_penalty # speed target

def effort_throttle_penalty(env):
    # last_throttle_action = mdp.last_action(env)[..., 0]*env.cfg.actions.throttle_steer.scale[0]

    effort_throttle_penalty = -(env._action_history[:, 0, 0])**2
    
    return effort_throttle_penalty # speed target

def effort_steering_penalty(env):
    # last_throttle_action = mdp.last_action(env)[..., 0]*env.cfg.actions.throttle_steer.scale[0]

    effort_steering_penalty = -(env._action_history[:, 0, 1])**2

    return effort_steering_penalty # speed target

def effort_target_steering_angle_penalty(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), mean_noise = 0, std_noise = 0) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if not hasattr(env, '_target_steering_angle_history'):
        env._obs_history_length = CONFIG['env_config']['OBS_HISTORY_LENGTH']
        env._target_steering_angle_history = torch.zeros(
            (env.num_envs, env._obs_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
            )    
    last_action = mdp.last_action(env)[..., 1]*CONFIG['env_config']['MAX_STEERING_ANGLE_INCREMENT']
    # # shift the history to the right and insert the last angular velocity at the beginning
    env._target_steering_angle_history[:, 1:] = env._target_steering_angle_history[:, :-1].clone()
    env._target_steering_angle_history[:, 0] = torch.clamp(env._target_steering_angle_history[:, 0] + last_action, min = -0.4, max=0.4)
    effort_steering_penalty = -(env._target_steering_angle_history[:, 0])**2

    return effort_steering_penalty

# def delta_throttle_l2_penalty(env):
#     # last_throttle_action = mdp.last_action(env)[..., 0]*env.cfg.actions.throttle_steer.scale[0]
#     last_throttle_action = mdp.last_action(env)[..., 0]
#     if not hasattr(env, '_action_history'):
#         env._action_history_length = CONFIG['env_config']['ACTION_HISTORY_LENGTH']
#         env._action_history = torch.zeros(
#             (env.num_envs, env._action_history_length, 2),  # Shape: (num_envs, history_length, n_actions)
#             dtype=torch.float32,
#             device=env.device
#             )
        
#     delta_throttle_l2 = -(env._action_history[:, 0, 0] - env._action_history[:, 1, 0])**2
    
#     return delta_throttle_l2 # speed target

def delta_speed_cmd_penalty(env):
    # last_throttle_action = mdp.last_action(env)[..., 0]*env.cfg.actions.throttle_steer.scale[0]
    last_throttle_action = mdp.last_action(env)[..., 0]
    last_speed_cmd = torch.clamp(last_throttle_action*CONFIG['env_config']['MAX_SPEED_SCALING']+CONFIG['env_config']['SPEED_OFFSET'], min=0, max=CONFIG['env_config']['MAX_SPEED_SCALING']) #-1,1 mapped to 0-8
    speed = mdp.base_lin_vel(env)[..., 0]
    if not hasattr(env, '_action_history'):
        env._action_history_length = CONFIG['env_config']['ACTION_HISTORY_LENGTH']
        env._action_history = torch.zeros(
            (env.num_envs, env._action_history_length, 2),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
            )
        
    offset_speed_cmd_penalty = -(last_speed_cmd - speed)**2
    
    return offset_speed_cmd_penalty # if speed command is above actual speed, penalize

def speed_target_rew(env, speed_target: float=1.):
    lin_vel = mdp.base_lin_vel(env)[..., :2]
    speed_dist = -((torch.norm(lin_vel, dim=-1) - speed_target) ** 2) + speed_target ** 2
    return torch.where(speed_dist > 0., speed_dist, 0.) # speed target


def steering_target_rew(env, steering_target: float=0.1):
    last_steering_action = mdp.last_action(env)[..., 1]*env.cfg.actions.throttle_steer.scale[1]
    steering_diff = -((last_steering_action - steering_target) ** 2) + steering_target
    return torch.where(steering_diff > 0., steering_diff, 0.) 

def low_speed_penalty(env):
    if not hasattr(env, '_base_lin_vel_x_history'):
        env._obs_history_length = CONFIG['env_config']['OBS_HISTORY_LENGTH']
        env._base_lin_vel_x_history = torch.zeros(
            (env.num_envs, env._obs_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
            )
    forward_speed_mean = torch.mean(env._base_lin_vel_x_history, dim=1)
    
    return torch.where(forward_speed_mean < 1, -1, 0.) # avoid standstill

def low_speed_penalty(env):
    if not hasattr(env, '_base_lin_vel_x_history'):
        env._obs_history_length = CONFIG['env_config']['OBS_HISTORY_LENGTH']
        env._base_lin_vel_x_history = torch.zeros(
            (env.num_envs, env._obs_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
            )
    forward_speed_mean = torch.mean(env._base_lin_vel_x_history, dim=1)
    
    return torch.where(forward_speed_mean < 1, -1, 0.) # avoid standstill

def average_vel(env):
    if not hasattr(env, '_base_lin_vel_x_history'):
        env._obs_history_length = CONFIG['env_config']['OBS_HISTORY_LENGTH']
        env._base_lin_vel_x_history = torch.zeros(
            (env.num_envs, env._obs_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
            )
        
    return torch.mean(env._base_lin_vel_x_history, dim=1)