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
    

def base_lin_vel_x(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), mean_noise = 0, std_noise = 0) -> torch.Tensor:
    """Root linear velocity in the asset's root frame. 2D, only x and y"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    noise = torch.empty(size=asset.data.root_lin_vel_b[:,0].unsqueeze(-1).shape, device=env.device).normal_(mean=mean_noise, std=std_noise)

    return (asset.data.root_lin_vel_b[:,0].unsqueeze(-1) + noise)

def base_lin_vel_y(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), mean_noise = 0, std_noise = 0) -> torch.Tensor:
    """Root linear velocity in the asset's root frame. 2D, only x and y"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    noise = torch.empty(size=asset.data.root_lin_vel_b[:,1].unsqueeze(-1).shape, device=env.device).normal_(mean=mean_noise, std=std_noise)

    return (asset.data.root_lin_vel_b[:,1].unsqueeze(-1) + noise)

def base_ang_vel_z(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), mean_noise = 0, std_noise = 0) -> torch.Tensor:
    """Root angular velocity in the asset's root frame. Only z, yaw rade"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    noise = torch.empty(size=asset.data.root_ang_vel_b[:,2].unsqueeze(-1).shape, device=env.device).normal_(mean=mean_noise, std=std_noise)

    return asset.data.root_ang_vel_b[:,2].unsqueeze(-1) + noise

def base_lin_vel_x_history(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), mean_noise = 0, std_noise = 0) -> torch.Tensor:
    """Root linear velocity in the asset's root frame. 2D, only x and y"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    noise = torch.empty(size=asset.data.root_ang_vel_b[:,2].unsqueeze(-1).shape, device=env.device).normal_(mean=mean_noise, std=std_noise)
    if not hasattr(env, '_base_lin_vel_x_history'):
        env._obs_history_length = CONFIG['env_config']['OBS_HISTORY_LENGTH']
        env._base_lin_vel_x_history = torch.zeros(
            (env.num_envs, env._obs_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
            )
    # shift the history to the right and insert the last angular velocity at the beginning
    env._base_lin_vel_x_history[:, 1:] = env._base_lin_vel_x_history[:, :-1].clone()
    env._base_lin_vel_x_history[:, 0] = asset.data.root_lin_vel_b[:,0]
    base_lin_vel_x_history = env._base_lin_vel_x_history
    return base_lin_vel_x_history

def base_lin_vel_y_history(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), mean_noise = 0, std_noise = 0) -> torch.Tensor:
    """Root linear velocity in the asset's root frame. 2D, only x and y"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    noise = torch.empty(size=asset.data.root_ang_vel_b[:,2].unsqueeze(-1).shape, device=env.device).normal_(mean=mean_noise, std=std_noise)
    if not hasattr(env, '_base_lin_vel_y_history'):
        env._obs_history_length = CONFIG['env_config']['OBS_HISTORY_LENGTH']
        env._base_lin_vel_y_history = torch.zeros(
            (env.num_envs, env._obs_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
            )
    # shift the history to the right and insert the last angular velocity at the beginning
    env._base_lin_vel_y_history[:, 1:] = env._base_lin_vel_y_history[:, :-1].clone()
    env._base_lin_vel_y_history[:, 0] = -asset.data.root_lin_vel_b[:,1]
    base_lin_vel_y_history = env._base_lin_vel_y_history
    return base_lin_vel_y_history

def base_ang_vel_z_history(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), mean_noise = 0, std_noise = 0) -> torch.Tensor:
    """Root angular velocity in the asset's root frame. Only z, yaw rade"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    noise = torch.empty(size=asset.data.root_ang_vel_b[:,2].unsqueeze(-1).shape, device=env.device).normal_(mean=mean_noise, std=std_noise)
    if not hasattr(env, '_base_ang_vel_z_history'):
        env._obs_history_length = CONFIG['env_config']['OBS_HISTORY_LENGTH']
        env._base_ang_vel_z_history = torch.zeros(
            (env.num_envs, env._obs_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
            )
    # shift the history to the right and insert the last angular velocity at the beginning
    env._base_ang_vel_z_history[:, 1:] = env._base_ang_vel_z_history[:, :-1].clone()
    env._base_ang_vel_z_history[:, 0] = asset.data.root_ang_vel_b[:,2]
    base_ang_vel_z_history = env._base_ang_vel_z_history
    
    return base_ang_vel_z_history

def target_velocity_history(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), mean_noise = 0, std_noise = 0) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    noise = torch.empty(size=asset.data.root_ang_vel_b[:,2].unsqueeze(-1).shape, device=env.device).normal_(mean=mean_noise, std=std_noise)
    if not hasattr(env, '_target_velocity_history'):
        env._obs_history_length = CONFIG['env_config']['OBS_HISTORY_LENGTH']
        env._target_velocity_history = torch.zeros(
            (env.num_envs, env._obs_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
            )
    
    last_action = mdp.last_action(env)[..., 0]*CONFIG['env_config']['MAX_SPEED_INCREMENT']
    # # shift the history to the right and insert the last angular velocity at the beginning
    env._target_velocity_history[:, 1:] = env._target_velocity_history[:, :-1].clone()
    env._target_velocity_history[:, 0] = torch.clamp(env._target_velocity_history[:, 0] + last_action, min = 0.0)
    target_velocity_history = env._target_velocity_history
    return target_velocity_history
#last action is from -1 and 1, not clipped
def action_history(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), mean_noise = 0, std_noise = 0) -> torch.Tensor:
    """Root angular velocity in the asset's root frame. Only z, yaw rade"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    noise = torch.empty(size=asset.data.root_ang_vel_b[:,2].unsqueeze(-1).shape, device=env.device).normal_(mean=mean_noise, std=std_noise)
    if not hasattr(env, '_action_history'):
        env._action_history_length = CONFIG['env_config']['ACTION_HISTORY_LENGTH']
        env._action_history = torch.zeros(
            (env.num_envs, env._action_history_length, 2),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
            )

    last_action = mdp.last_action(env)[..., :]
    # Shift the history to the right and insert the last action at the beginning
    env._action_history[:, 1:, :] = env._action_history[:, :-1, :].clone()
    env._action_history[:, 0, :] = last_action
    action_history_obs = env._action_history.reshape(-1, env._action_history_length * 2)  # Flatten the history for observation

    return action_history_obs 

def deviation_centerline_horizon(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    delta_s_idx: int = 10,
    n_horizon: int = 5,
    t_horizon: int = 5
) -> torch.Tensor:
    """
    Calculate signed lateral deviations from multiple reference waypoint segments.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        delta_s_idx: Base number of waypoints to look ahead
        n_horizon: Number of delta_s_idx points to return
        
    Returns:
        Tensor of signed deviations (positive = left of reference, negative = right)
        for each n_horizon segment. Shape: [num_envs, n_horizon]
    """
    # Get current state
    asset = env.scene[asset_cfg.name]
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
    vel_x = mdp.base_lin_vel(env)[..., 0]

    num_envs = pos_xy_world.shape[0]
    
    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
    
    # Get map levels for all environments
    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)
    
    # Initialize output tensor
    deviations = torch.zeros(num_envs, n_horizon, device=env.device)
    
    # Process each map level separately
    for map_level in unique_map_levels:
        # Create mask for environments using this map
        env_mask = (map_levels == map_level)
        num_envs_in_map = env_mask.sum()
        
        if num_envs_in_map == 0:
            continue
            
        # Get positions for these environments
        map_positions = pos_xy_world[env_mask]
        map_vel_x     = vel_x[env_mask]

        # Get waypoints for this map level
        waypoints_xy_world = torch.tensor(
            env.scene.terrain.cfg.waypoints_list[map_level],
            device=env.device,
            dtype=torch.float32
        )[:, :2]
        
        inner_xy_world = torch.tensor(
            env.scene.terrain.cfg.inner_list[map_level],
            device=env.device,
            dtype=torch.float32
        )[:, :2]
        # Find nearest waypoint for these environments
        current_idx, _ = find_frenet_coord_along_waypoints(inner_xy_world, map_positions)  # shape: [num_envs_in_map]
        
        if CONFIG['env_config']['DYNAMIC_LOOKAHEAD']:
            
            s_length  = CONFIG['env_config']['LEN_S_IDX'] # [m]
            s_horizon = map_vel_x * t_horizon 
            idx_horizon = s_horizon/s_length

            horizon_indices = calculate_horizon_indices(
                current_idx=current_idx,
                idx_horizon=idx_horizon,
                waypoints_length=len(waypoints_xy_world),
                n_horizon=n_horizon,
                device=env.device
            )
        else:
            lookahead_steps = torch.linspace(0, delta_s_idx*n_horizon, n_horizon+1, device=env.device)
            horizon_indices = (current_idx.unsqueeze(-1) + lookahead_steps.unsqueeze(0)) % len(waypoints_xy_world)
            horizon_indices = horizon_indices.long()

        # Get waypoint pairs for each segment [num_envs_in_map, n_horizon, 2, 2]
        segment_starts = waypoints_xy_world[horizon_indices[:, :-1]]
        segment_ends = waypoints_xy_world[horizon_indices[:, 1:]]
        
        # Calculate track directions [num_envs_in_map, n_horizon, 2]
        track_dirs = segment_ends - segment_starts
        
        # Calculate car offsets [num_envs_in_map, n_horizon, 2]
        car_offsets = map_positions.unsqueeze(1) - segment_starts
        
        # Cross products (track_dir × car_offset) [num_envs_in_map, n_horizon]
        cross_products = (track_dirs[:, :, 0] * car_offsets[:, :, 1] - 
                         track_dirs[:, :, 1] * car_offsets[:, :, 0])
        signs = torch.sign(cross_products)
        
        # Perpendicular distances [num_envs_in_map, n_horizon]
        segment_lengths = torch.norm(track_dirs, dim=2)
        distances = torch.abs(cross_products) / (segment_lengths + 1e-6)
        
        # Signed deviations and store in output
        signed_deviations = signs * distances
        norm_distance = 1
        deviations[env_mask] = signed_deviations / norm_distance

    return deviations

def heading_error_horizon(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    delta_s_idx: int = 10,
    n_horizon: int = 5,
    t_horizon: int = 5
) -> torch.Tensor:
    """
    Calculate heading errors between car's current orientation and multiple delta_s_idx waypoints,
    supporting multiple maps through env._map_levels.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        delta_s_idx: Base number of waypoints to look ahead
        n_horizon: Number of delta_s_idx points to return
        
    Returns:
        Tensor of heading errors in radians (range [-π, π]) for each n_horizon point.
        Shape: [num_envs, n_horizon]
    """
    # Get current state
    asset = env.scene[asset_cfg.name]
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
    vel_x = mdp.base_lin_vel(env)[..., 0]

    heading_w = asset.data.heading_w
    num_envs = pos_xy_world.shape[0]

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
        
    # Get map levels for all environments
    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)
    
    # Initialize output tensor
    heading_errors = torch.zeros(num_envs, n_horizon, device=env.device)
    
    # Process each map level separately
    for map_level in unique_map_levels:
        # Create mask for environments using this map
        env_mask = (map_levels == map_level)
        num_envs_in_map = env_mask.sum()
        
        if num_envs_in_map == 0:
            continue
            
        # Get positions and headings for these environments
        map_positions = pos_xy_world[env_mask]
        map_vel_x    = vel_x[env_mask]
        map_headings = heading_w[env_mask]
        
        # Get waypoints for this map level
        waypoints_xy_world = torch.tensor(
            env.scene.terrain.cfg.waypoints_list[map_level],
            device=env.device,
            dtype=torch.float32
        )[:, :2]
        
        inner_xy_world = torch.tensor(
            env.scene.terrain.cfg.inner_list[map_level],
            device=env.device,
            dtype=torch.float32
        )[:, :2]

        # Find nearest waypoint for these environments
        current_idx, _ = find_frenet_coord_along_waypoints(inner_xy_world, map_positions)  # shape: [num_envs_in_map]
        
        if CONFIG['env_config']['DYNAMIC_LOOKAHEAD']:
            
            s_length  = CONFIG['env_config']['LEN_S_IDX'] # [m]
            s_horizon = map_vel_x * t_horizon 
            idx_horizon = s_horizon/s_length

            horizon_indices = calculate_horizon_indices(
                current_idx=current_idx,
                idx_horizon=idx_horizon,
                waypoints_length=len(waypoints_xy_world),
                n_horizon=n_horizon,
                device=env.device
            )
        else:
            lookahead_steps = torch.linspace(0, delta_s_idx*n_horizon, n_horizon+1, device=env.device)
            horizon_indices = (current_idx.unsqueeze(-1) + lookahead_steps.unsqueeze(0)) % len(waypoints_xy_world)
            horizon_indices = horizon_indices.long()


        # Get delta_s_idx waypoints [num_envs_in_map, n_horizon, 2]
        lookahead_points = waypoints_xy_world[horizon_indices[:, 1:]]
        
        # Calculate desired heading vectors [num_envs_in_map, n_horizon]
        desired_headings = torch.atan2(
            lookahead_points[:, :, 1] - map_positions[:, 1].unsqueeze(-1),
            lookahead_points[:, :, 0] - map_positions[:, 0].unsqueeze(-1)
        )
        
        # Calculate smallest angle differences [num_envs_in_map, n_horizon]
        current_heading_errors = torch.atan2(
            torch.sin(desired_headings - map_headings.unsqueeze(-1)),
            torch.cos(desired_headings - map_headings.unsqueeze(-1))
        )
        
        # Store results in the output tensor
        norm_heading_errors = 1
        heading_errors[env_mask] = current_heading_errors / norm_heading_errors

    return heading_errors

def d_lat_horizon(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    delta_s_idx: int = 10,
    n_horizon: int = 5,
    t_horizon: int = 5
) -> torch.Tensor:
    """
    Calculate normalized lateral space from centerline to trackbounds, supporting multiple maps.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        delta_s_idx: Base number of waypoints to look ahead
        n_horizon: Number of delta_s_idx points to return
        
    Returns:
        Tensor of normalized lateral space (shape: [num_envs, n_horizon * 2])
    """
    # Get current state
    asset = env.scene[asset_cfg.name]
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
    vel_x = mdp.base_lin_vel(env)[..., 0]

    num_envs = pos_xy_world.shape[0]

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
        
    # Get map levels for all environments
    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)
    
    # Initialize output tensor (n_horizon * 2 for left/right bounds at each point)
    d_lat_results = torch.zeros(num_envs, n_horizon * 2, device=env.device)
    
    # Process each map level separately
    for map_level in unique_map_levels:
        # Create mask for environments using this map
        env_mask = (map_levels == map_level)
        num_envs_in_map = env_mask.sum()
        
        if num_envs_in_map == 0:
            continue
            
        # Get positions for these environments, shift by origin env
        map_positions = pos_xy_world[env_mask]
        map_vel_x     = vel_x[env_mask]
        # Get waypoints and track bounds for this map level
        waypoints_xy_world = torch.tensor(
            env.scene.terrain.cfg.waypoints_list[map_level],
            device=env.device,
            dtype=torch.float32
        )[:, :2]
        
        inner_xy_world = torch.tensor(
            env.scene.terrain.cfg.inner_list[map_level],
            device=env.device,
            dtype=torch.float32
        )[:, :2]

        d_lat = torch.tensor(
            env.scene.terrain.cfg.d_lat_list[map_level],
            device=env.device,
            dtype=torch.float32
        )

        # Find nearest waypoint for these environments
        current_idx, _ = find_frenet_coord_along_waypoints(inner_xy_world, map_positions)  # shape: [num_envs_in_map]
        
        if CONFIG['env_config']['DYNAMIC_LOOKAHEAD']:
            
            s_length  = CONFIG['env_config']['LEN_S_IDX'] # [m]
            s_horizon = map_vel_x * t_horizon 
            idx_horizon = s_horizon/s_length

            horizon_indices = calculate_horizon_indices(
                current_idx=current_idx,
                idx_horizon=idx_horizon,
                waypoints_length=len(waypoints_xy_world),
                n_horizon=n_horizon,
                device=env.device
            )
        else:
            lookahead_steps = torch.linspace(0, delta_s_idx*n_horizon, n_horizon+1, device=env.device)
            horizon_indices = (current_idx.unsqueeze(-1) + lookahead_steps.unsqueeze(0)) % len(waypoints_xy_world)
            horizon_indices = horizon_indices.long()

        # Gather d_lat values - shape: [num_envs_in_map, n_horizon, 2]
        next_d_lat_horizon = d_lat[horizon_indices[:, 1:], :]
        
        # Normalize and reshape to [num_envs_in_map, n_horizon * 2]
        norm_d_lat = 1
        norm_next_d_lat_horizon = next_d_lat_horizon.reshape(-1, n_horizon * 2) / norm_d_lat
        
        # Store results in the output tensor
        d_lat_results[env_mask] = norm_next_d_lat_horizon

    return d_lat_results


def kappa_radpm_horizon(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    delta_s_idx: int = 10,
    n_horizon: int = 5,
    t_horizon: int = 5
) -> torch.Tensor:
    """
    Calculate curvature (kappa) values for n_horizon points, supporting multiple maps.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        delta_s_idx: Base number of waypoints to look ahead
        n_horizon: Number of delta_s_idx points to return
        
    Returns:
        Tensor of normalized curvature values (shape: [num_envs, n_horizon])
    """

    asset = env.scene[asset_cfg.name]
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
    vel_x = mdp.base_lin_vel(env)[..., 0]

    num_envs = pos_xy_world.shape[0]

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
        
    # Get map levels for all environments
    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)
    
    # Initialize output tensor
    kappa_results = torch.zeros(num_envs, n_horizon, device=env.device)
    
    # Process each map level separately
    for map_level in unique_map_levels:
        # Create mask for environments using this map
        env_mask = (map_levels == map_level)
        num_envs_in_map = env_mask.sum()
        
        if num_envs_in_map == 0:
            continue
            
        # Get positions for these environments
        map_positions = pos_xy_world[env_mask]
        map_vel_x     = vel_x[env_mask]

        # Get waypoints and track data for this map level
        waypoints_xy_world = torch.tensor(
            env.scene.terrain.cfg.waypoints_list[map_level],
            device=env.device,
            dtype=torch.float32
        )[:, :2]
        
        inner_xy_world = torch.tensor(
            env.scene.terrain.cfg.inner_list[map_level],
            device=env.device,
            dtype=torch.float32
        )[:, :2]

        kappa_radpm = torch.tensor(
            env.scene.terrain.cfg.kappa_radpm_list[map_level],
            device=env.device,
            dtype=torch.float32
        )

        # Find nearest waypoint for these environments
        current_idx, _ = find_frenet_coord_along_waypoints(inner_xy_world, map_positions)  # shape: [num_envs_in_map]
        
        if CONFIG['env_config']['DYNAMIC_LOOKAHEAD']:
            s_length  = CONFIG['env_config']['LEN_S_IDX'] # [m]
            s_horizon = map_vel_x * t_horizon 
            idx_horizon = s_horizon/s_length

            horizon_indices = calculate_horizon_indices(
                current_idx=current_idx,
                idx_horizon=idx_horizon,
                waypoints_length=len(waypoints_xy_world),
                n_horizon=n_horizon,
                device=env.device
            )
        else:
            lookahead_steps = torch.linspace(0, delta_s_idx*n_horizon, n_horizon+1, device=env.device)
            horizon_indices = (current_idx.unsqueeze(-1) + lookahead_steps.unsqueeze(0)) % len(waypoints_xy_world)
            horizon_indices = horizon_indices.long()

        lookahead_steps = torch.linspace(delta_s_idx, delta_s_idx*n_horizon, n_horizon, device=env.device)
        horizon_indices = (current_idx.unsqueeze(-1) + lookahead_steps.unsqueeze(0)) % len(waypoints_xy_world)
        horizon_indices = horizon_indices.long()


        # Gather curvature values - shape: [num_envs_in_map, n_horizon]
        next_kappa_radpm_horizon = kappa_radpm[horizon_indices[:, :1]].squeeze(-1)
        
        # Normalize and store results
        norm_kappa_radpm = 1
        kappa_results[env_mask] = next_kappa_radpm_horizon / norm_kappa_radpm

    return kappa_results

def delta_psi_rad_horizon(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    delta_s_idx: int = 10,
    n_horizon: int = 5,
    t_horizon: int = 5
) -> torch.Tensor:
    """
    Calculate curvature (kappa) values for n_horizon points, supporting multiple maps.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        delta_s_idx: Base number of waypoints to look ahead
        n_horizon: Number of delta_s_idx points to return
        
    Returns:
        Tensor of normalized curvature values (shape: [num_envs, n_horizon])
    """

    asset = env.scene[asset_cfg.name]
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
    vel_x = mdp.base_lin_vel(env)[..., 0]

    num_envs = pos_xy_world.shape[0]

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
        
    # Get map levels for all environments
    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)
    
    # Initialize output tensor
    kappa_results = torch.zeros(num_envs, n_horizon, device=env.device)
    
    # Process each map level separately
    for map_level in unique_map_levels:
        # Create mask for environments using this map
        env_mask = (map_levels == map_level)
        num_envs_in_map = env_mask.sum()
        
        if num_envs_in_map == 0:
            continue
            
        # Get positions for these environments
        map_positions = pos_xy_world[env_mask]
        map_vel_x     = vel_x[env_mask]
        # Get waypoints and track data for this map level
        waypoints_xy_world = torch.tensor(
            env.scene.terrain.cfg.waypoints_list[map_level],
            device=env.device,
            dtype=torch.float32
        )[:, :2]
        
        inner_xy_world = torch.tensor(
            env.scene.terrain.cfg.inner_list[map_level],
            device=env.device,
            dtype=torch.float32
        )[:, :2]

        psi_rad = torch.tensor(
            env.scene.terrain.cfg.psi_rad_list[map_level],
            device=env.device,
            dtype=torch.float32
        )

        # Find nearest waypoint for these environments
        current_idx, _ = find_frenet_coord_along_waypoints(inner_xy_world, map_positions)  # shape: [num_envs_in_map]
        
        if CONFIG['env_config']['DYNAMIC_LOOKAHEAD']:
            
            s_length  = CONFIG['env_config']['LEN_S_IDX'] # [m]
            s_horizon = map_vel_x * t_horizon 
            idx_horizon = s_horizon/s_length

            horizon_indices = calculate_horizon_indices(
                current_idx=current_idx,
                idx_horizon=idx_horizon,
                waypoints_length=len(waypoints_xy_world),
                n_horizon=n_horizon,
                device=env.device
            )
        else:
            lookahead_steps = torch.linspace(0, delta_s_idx*n_horizon, n_horizon+1, device=env.device)
            horizon_indices = (current_idx.unsqueeze(-1) + lookahead_steps.unsqueeze(0)) % len(waypoints_xy_world)
            horizon_indices = horizon_indices.long()

        lookahead_steps = torch.linspace(delta_s_idx, delta_s_idx*n_horizon, n_horizon, device=env.device)
        horizon_indices = (current_idx.unsqueeze(-1) + lookahead_steps.unsqueeze(0)) % len(waypoints_xy_world)
        horizon_indices = horizon_indices.long()

        delta_psi_rad = (psi_rad[horizon_indices[:, 1:]] - psi_rad[horizon_indices[:, :-1]]).squeeze(-1)
                
        # Normalize and store results
        norm_psi_rad = 1
        delta_psi_rad[env_mask] = delta_psi_rad / norm_psi_rad

    return delta_psi_rad

def opponent_frenet_info(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    opponent_cfg: SceneEntityCfg = SceneEntityCfg("opponent")
) -> torch.Tensor:

    ego_pos_xy_world = mdp.root_pos_w(env=env, asset_cfg=asset_cfg)[..., :2]
    ego_heading_w = env.scene[asset_cfg.name].data.heading_w
    ego_vel_x = mdp.base_lin_vel(env=env, asset_cfg=asset_cfg)[..., 0]

    opp_pos_xy_world = mdp.root_pos_w(env=env, asset_cfg=opponent_cfg)[..., :2]
    opp_vel_x = mdp.base_lin_vel(env=env, asset_cfg=opponent_cfg)[..., 0]

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
        
    # Get map levels for all environments
    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)

    num_envs              = ego_pos_xy_world.shape[0]
    s_idx_diff_opp_ego    = torch.zeros(num_envs, device=env.device, dtype= torch.float32)*CONFIG['env_config']['OPPONENT_INIT_DISTANCE_IDX']*CONFIG['env_config']['LEN_S_IDX']
    cross_pos_opp_ego     = torch.zeros(num_envs, device = env.device, dtype=torch.float32)
    d_diff_opp_ego        = torch.zeros(num_envs, device = env.device, dtype=torch.float32)

    vx_diff_opp_ego       = torch.zeros(num_envs, device = env.device, dtype=torch.float32)
    # Process each map level separately

    # num_episodes = env.common_step_counter // env.max_episode_length
    # if num_episodes < CONFIG['env_config']['IGNORE_OPPONENT_UNTIL_EP']:
    #     return torch.stack([s_idx_diff_opp_ego, cross_pos_opp_ego, opp_detected],dim=1)

    for map_level in unique_map_levels:
        # Create mask for environments using this map
        env_mask = (map_levels == map_level)
        num_envs_in_map = env_mask.sum()
        
        if num_envs_in_map == 0:
            continue
            
        # Get positions for these environments
        ego_map_positions = ego_pos_xy_world[env_mask]
        ego_map_headings =  ego_heading_w[env_mask]
        ego_map_vel_x = ego_vel_x[env_mask]
        opp_map_positions = opp_pos_xy_world[env_mask]

        # Get waypoints and track data for this map level
        waypoints_xy_world = torch.tensor(
            env.scene.terrain.cfg.waypoints_list[map_level],
            device=env.device,
            dtype=torch.float32
        )[:, :2]
        
        num_waypoints = len(waypoints_xy_world)
        ego_current_s_idx, ego_current_d = find_frenet_coord_along_waypoints(waypoints_xy_world, ego_map_positions)
        opp_current_s_idx, opp_current_d = find_frenet_coord_along_waypoints(waypoints_xy_world, opp_map_positions)

        d_diff_opp_ego[env_mask]            = opp_current_d     - ego_current_d
        s_idx_diff_raw    = opp_current_s_idx - ego_current_s_idx
        s_idx_diff_signed = (s_idx_diff_raw + num_waypoints // 2) % num_waypoints - (num_waypoints // 2)

        # s_idx_diff_ot = torch.where((s_idx_diff_signed > 0) & (s_idx_diff_signed <= CONFIG['env_config']['OPPONENT_INIT_DISTANCE_IDX']),
        #                         s_idx_diff_signed, 
        #                         torch.zeros_like(s_idx_diff_signed))  

        s_idx_diff_opp_ego[env_mask] = s_idx_diff_signed*CONFIG['env_config']['LEN_S_IDX']
        t_diff_opp_ego = s_idx_diff_opp_ego/(ego_map_vel_x+0.001)
        t_diff_opp_ego = torch.clamp(t_diff_opp_ego, -10.0, 10.0)  
        
        # d_diff_opp_ego[env_mask]     = torch.where(opp_detected[env_mask].bool(),
        #                                d_diff, 
        #                                torch.zeros(len(env_mask), device=env.device, dtype= torch.float32))  


        ego_map_heading_vec = torch.stack([torch.cos(ego_map_headings), torch.sin(ego_map_headings)], dim=1)
        dist = torch.norm(ego_map_positions - opp_map_positions, p=2, dim=1)
        ego_opp_vec = (opp_map_positions - ego_map_positions)
        ego_opp_positioning_vec = (ego_map_heading_vec[:, 0] * ego_opp_vec[:, 1] 
                                - ego_map_heading_vec[:, 1] * ego_opp_vec[:, 0]) / (dist + 1e-6)
        ego_opp_positioning_vec_norm = torch.clamp(ego_opp_positioning_vec, -1.0, 1.0)
        
        cross_pos_opp_ego[env_mask] = ego_opp_positioning_vec_norm
        vx_diff_opp_ego[env_mask] = opp_vel_x - ego_vel_x
        
    return torch.stack([t_diff_opp_ego, d_diff_opp_ego, cross_pos_opp_ego, vx_diff_opp_ego],dim=1)

def gaps_info(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    opponent_cfg: SceneEntityCfg = SceneEntityCfg("opponent")
) -> torch.Tensor:


    ego_pos_xy_world = mdp.root_pos_w(env=env, asset_cfg=asset_cfg)[..., :2]
    ego_heading_w = env.scene[asset_cfg.name].data.heading_w

    opp_pos_xy_world = mdp.root_pos_w(env=env, asset_cfg=opponent_cfg)[..., :2]

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
        
    # Get map levels for all environments
    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)

    num_envs              = ego_pos_xy_world.shape[0]

    gap_inner_heading_error     = torch.zeros(num_envs, device = env.device, dtype=torch.float32)
    gap_outer_heading_error     = torch.zeros(num_envs, device = env.device, dtype=torch.float32)

    opp_detected          = torch.zeros(num_envs, device = env.device, dtype=torch.long)

    # Process each map level separately

    # num_episodes = env.common_step_counter // env.max_episode_length
    # if num_episodes < CONFIG['env_config']['IGNORE_OPPONENT_UNTIL_EP']:
    #     return torch.stack([gap_inner_heading_error, gap_outer_heading_error],dim=1)

    for map_level in unique_map_levels:
        # Create mask for environments using this map
        env_mask = (map_levels == map_level)
        num_envs_in_map = env_mask.sum()
        
        if num_envs_in_map == 0:
            continue
            
        # Get positions for these environments
        ego_map_positions = ego_pos_xy_world[env_mask]
        ego_map_headings =  ego_heading_w[env_mask]

        opp_map_positions = opp_pos_xy_world[env_mask]

        # Get waypoints and track data for this map level
        waypoints_xy_world = torch.tensor(
            env.scene.terrain.cfg.waypoints_list[map_level],
            device=env.device,
            dtype=torch.float32
        )[:, :2]

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
        
        num_waypoints = len(waypoints_xy_world)
        ego_current_s_idx, ego_current_d = find_frenet_coord_along_waypoints(waypoints_xy_world, ego_map_positions)
        opp_current_s_idx, opp_current_d = find_frenet_coord_along_waypoints(waypoints_xy_world, opp_map_positions)

        # s_idx_diff_raw    = opp_current_s_idx - ego_current_s_idx
        # s_idx_diff_signed = (s_idx_diff_raw + num_waypoints // 2) % num_waypoints - (num_waypoints // 2)

        # opp_detected[env_mask]       = torch.where(abs(s_idx_diff_signed) <= CONFIG['env_config']['OPPONENT_INIT_DISTANCE_IDX'],
        #                                torch.ones(len(env_mask), device=env.device, dtype= torch.long), 
        #                                torch.zeros(len(env_mask), device=env.device, dtype= torch.long))  
    
        gap_inner = torch.norm(opp_map_positions-inner_xy_world[opp_current_s_idx, :], dim=1)
        gap_outer = torch.norm(opp_map_positions-outer_xy_world[opp_current_s_idx, :], dim=1)
        gap_inner_center =   (opp_map_positions + inner_xy_world[opp_current_s_idx, :])/2
        gap_outer_center =   (opp_map_positions + outer_xy_world[opp_current_s_idx, :])/2
        
        
        # Calculate desired heading vectors [num_envs_in_map, n_horizon]
        desired_headings_gap_inner = torch.atan2(
            gap_inner_center[:, 1] - ego_map_positions[:, 1],
            gap_inner_center[:, 0] - ego_map_positions[:, 0]
        )
        
        desired_headings_gap_outer = torch.atan2(
            gap_outer_center[:, 1] - ego_map_positions[:, 1],
            gap_outer_center[:, 0] - ego_map_positions[:, 0]
        )
        
        # Calculate smallest angle differences [num_envs_in_map, n_horizon]
        gap_inner_heading_error[env_mask] = torch.atan2(torch.sin(desired_headings_gap_inner - ego_map_headings),torch.cos(desired_headings_gap_inner - ego_map_headings))
        gap_outer_heading_error[env_mask] = torch.atan2(torch.sin(desired_headings_gap_outer - ego_map_headings),torch.cos(desired_headings_gap_outer - ego_map_headings))
        
        car_width = 0.30
        
    return torch.stack([gap_inner-car_width, gap_inner_heading_error, gap_outer-car_width, gap_outer_heading_error],dim=1)

def calculate_horizon_indices(
    current_idx: torch.Tensor,
    idx_horizon: torch.Tensor,
    waypoints_length: int,
    n_horizon: int,
    device: torch.device
) -> torch.Tensor:
    """
    Calculate n_horizon indices for waypoint following in a vectorized manner.
    
    Args:
        current_idx: Current waypoint indices [num_envs]
        idx_horizon: delta_s_idx distance in indices [num_envs]
        waypoints_length: Total number of waypoints
        n_horizon: Number of delta_s_idx points
        device: Target device for computations
        
    Returns:
        Tensor of waypoint indices [num_envs, n_horizon + 1]
    """
    # Create normalized steps [n_horizon + 1]
    normalized_steps = torch.linspace(0, 1, n_horizon + 1, device=device)
    
    # Vectorized step calculation [num_envs, n_horizon + 1]
    steps = idx_horizon.unsqueeze(-1) * normalized_steps.unsqueeze(0)
    steps = steps.long()
    
    # Create n_horizon indices with circular buffer
    horizon_indices = (current_idx.unsqueeze(-1) + steps) % waypoints_length
    return horizon_indices.long()