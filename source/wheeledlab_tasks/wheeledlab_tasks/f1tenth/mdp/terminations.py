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
    
# def out_of_map(env):
#     poses = mdp.root_pos_w(env)
#     poses = poses[..., :2]
#     terrain = env.scene[SceneEntityCfg("terrain").name]
#     width = terrain.cfg.width_list[0]
#     height = terrain.cfg.height_list[0]
#     x_out_range = torch.logical_or(poses[..., 0] > width / 2, poses[..., 0] < -width / 2)
#     y_out_range = torch.logical_or(poses[..., 1] > height / 2, poses[..., 1] < -height / 2)
#     return torch.logical_or(x_out_range, y_out_range)


# def upright_bool(env, thresh_deg):
#     return upright_penalty(env, thresh_deg) > 0.0


# def is_not_traversable(env):
#     poses =mdp.root_pos_w(env)[..., :2]
#     if not hasattr(env, '_map_levels'):
#         env._map_levels = torch.zeros(env.num_envs, 
#                                 dtype=torch.long,
#                                 device=env.device)
#     map_levels     = env._map_levels
#     traversability = TraversabilityHashmapUtil().get_traversability(poses, map_levels)
#     num_episodes   = env.common_step_counter // env.max_episode_length
# #   delay the termination for the first 10 episodes
#     # if num_episodes < 50:
#     #     # return false (IS traversable)
#     #     return torch.zeros(env.num_envs, device=env.device) == 1
    
#     if not hasattr(env, '_traversability_history'):
#         env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
#         env._traversability_history = torch.ones(
#             (env.num_envs, env._rew_history_length), 
#             dtype=torch.long,
#             device=env.device
#         )
    
#     env._traversability_history[:, 1:] = env._traversability_history[:, :-1].clone()
#     env._traversability_history[:, 0] = traversability
#     delayed_traversability = env._traversability_history[:,-1]

#     return torch.logical_not(delayed_traversability)

def wall_collision(env):
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
    num_envs = pos_xy_world.shape[0]

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)

    if not hasattr(env, '_wall_collision_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._wall_collision_history = torch.ones(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
  
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
        inner_xy_world = torch.tensor(
            env.scene.terrain.cfg.inner_list[map_level],
            device=env.device,
            dtype=torch.float32
        )[:, :2]

        outer_xy_world = torch.tensor(
            env.scene.terrain.cfg.outer_list[map_level],
            device=env.device,
            dtype=torch.float32
        )[:, :2]

        # Calculate distances to ALL inner and outer points
        # Expand dimensions for broadcasting: [num_envs, 1, 2] - [1, num_points, 2] = [num_envs, num_points]
        dist_to_inner = torch.norm(map_positions[:, None, :] - inner_xy_world[None, :, :], p=2, dim=2)  # shape: [num_envs, num_inner_points]
        dist_to_outer = torch.norm(map_positions[:, None, :] - outer_xy_world[None, :, :], p=2, dim=2)  # shape: [num_envs, num_outer_points]

        # Find minimum distance to each boundary
        min_dist_to_inner = torch.min(dist_to_inner, dim=1)[0]  # shape: [num_envs]
        min_dist_to_outer = torch.min(dist_to_outer, dim=1)[0]  # shape: [num_envs]

        # Check for collisions with inner and outer bounds
        collision_with_inner = min_dist_to_inner < CONFIG['env_config']['WALL_COLLISION_RADIUS']
        collision_with_outer = min_dist_to_outer < CONFIG['env_config']['WALL_COLLISION_RADIUS']
                
        # Combine collisions (OR operation - collision with either counts)
        collisions_in_map = collision_with_inner | collision_with_outer
        
        # Convert collisions to long dtype (0 or 1)
        collisions_in_map = collisions_in_map.long()
        
        # Update the collision_bool tensor for these environments
        collision_bool[env_mask] = collisions_in_map

    env._wall_collision_history[:, 1:] = env._wall_collision_history[:, :-1].clone()
    env._wall_collision_history[:, 0] = collision_bool
    
    return  env._wall_collision_history[:, -1].bool()

def opponent_collision(env):
    num_episodes = env.common_step_counter // env.max_episode_length
    if num_episodes <  CONFIG['env_config']['IGNORE_OPPONENT_UNTIL_EP']:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    
    if not hasattr(env, '_prev_delta_s_opp_ego'):
        env._prev_delta_s_opp_ego = torch.ones(env.num_envs, 
                                dtype=torch.float32,
                                device=env.device)*CONFIG['env_config']['OPPONENT_INIT_DISTANCE_IDX']*CONFIG['env_config']['LEN_S_IDX']
        
        
    ego_position_xy = mdp.root_pos_w(env = env, asset_cfg = SceneEntityCfg("robot"))[..., :2]
    opp_position_xy = mdp.root_pos_w(env = env, asset_cfg = SceneEntityCfg("opponent"))[:, :2]
    
    dist = torch.norm(ego_position_xy - opp_position_xy, p=2, dim=1)
    opp_collision = dist < CONFIG['env_config']['OPP_COLLISION_RADIUS']

    return opp_collision.bool()

def opponent_overtaken(env):

    num_episodes = env.common_step_counter // env.max_episode_length
    if num_episodes < CONFIG['env_config']['IGNORE_OPPONENT_UNTIL_EP']:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    if not hasattr(env, '_opponent_overtaken_bool'):
       env._opponent_overtaken_bool = torch.zeros(
            env.num_envs,  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.bool,
            device=env.device
        )
       
    return env._opponent_overtaken_bool

def far_from_opponent(    
        env: ManagerBasedEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        opponent_cfg: SceneEntityCfg = SceneEntityCfg("opponent")
    ) -> torch.Tensor:
    
    # num_episodes = env.common_step_counter // env.max_episode_length
    # if num_episodes < CONFIG['env_config']['IGNORE_OPPONENT_UNTIL_EP']:
    #     return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        
    ego_position_xy = mdp.root_pos_w(env = env, asset_cfg = SceneEntityCfg("robot"))[..., :2]
    opp_position_xy = mdp.root_pos_w(env = env, asset_cfg = SceneEntityCfg("opponent"))[:, :2]

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
        
    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)
    
    # Initialize outputs
    delta_s_opp_ego       = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    far_from_opponent_bool = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # ego_behind_opp_bool  = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # current_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # Process each map level separately
    for map_level in unique_map_levels:
        env_mask = (map_levels == map_level)
        num_envs_in_map = env_mask.sum()
        
        if num_envs_in_map == 0:
            continue
            
        # Get waypoints for this map level
        waypoints_world = torch.tensor(
            env.scene.terrain.cfg.waypoints_list[map_level], 
            device=env.device
        )[:, :2]
        num_waypoints = len(waypoints_world)

        # Find nearest waypoint for these environments
        ego_current_idx, _ = find_frenet_coord_along_waypoints(
            waypoints_world, 
            ego_position_xy[env_mask]
        )
        opp_current_idx, _ = find_frenet_coord_along_waypoints(
            waypoints_world, 
            opp_position_xy[env_mask]
        )

        delta_s_opp_ego[env_mask] = ((opp_current_idx-ego_current_idx + num_waypoints//2) % num_waypoints - num_waypoints // 2).float()
    
    far_from_opponent_bool = torch.where(delta_s_opp_ego > 75,
                                                    torch.ones_like(delta_s_opp_ego, dtype=torch.bool),
                                                    torch.zeros_like(delta_s_opp_ego, dtype=torch.bool))  
        
    return far_from_opponent_bool.bool()
