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
    norm = 8.0
    return base_lin_vel_x_history/norm

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
    norm = 2.0
    return base_ang_vel_z_history/norm

def base_lin_acc_x_history(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), mean_noise = 0, std_noise = 0) -> torch.Tensor:
    """Root linear acceleration in the asset's root frame. 2D, only x and y"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    noise = torch.empty(size=asset.data.root_ang_vel_b[:,2].unsqueeze(-1).shape, device=env.device).normal_(mean=mean_noise, std=std_noise)
    if not hasattr(env, '_base_lin_acc_x_history'):
        env._obs_history_length = CONFIG['env_config']['OBS_HISTORY_LENGTH']
        env._base_lin_acc_x_history = torch.zeros(
            (env.num_envs, env._obs_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
            )
    # shift the history to the right and insert the last angular velocity at the beginning
    env._base_lin_acc_x_history[:, 1:] = env._base_lin_acc_x_history[:, :-1].clone()
    env._base_lin_acc_x_history[:, 0]  = (env._base_lin_vel_x_history[:, 0] - env._base_lin_vel_x_history[:, 1])/(env.step_dt)
    norm = 3.0
    return env._base_lin_acc_x_history/norm

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
        env._target_steering_angle_history = torch.zeros(
            (env.num_envs, env._obs_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
            )    
    last_action = mdp.last_action(env)[..., 0]*CONFIG['env_config']['MAX_SPEED_INCREMENT']
    # # shift the history to the right and insert the last angular velocity at the beginning
    env._target_velocity_history[:, 1:] = env._target_velocity_history[:, :-1].clone()
    env._target_velocity_history[:, 0] = torch.clamp(env._target_velocity_history[:, 0] + last_action, min = 0.0)
    norm = 8.0
    return env._target_velocity_history/norm

def target_steering_angle_history(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), mean_noise = 0, std_noise = 0) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    noise = torch.empty(size=asset.data.root_ang_vel_b[:,2].unsqueeze(-1).shape, device=env.device).normal_(mean=mean_noise, std=std_noise)
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
    norm = 0.4
    return env._target_steering_angle_history/norm

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

def get_horizon_indices(
    env,
    map_level: int,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    delta_s_idx: int,
    n_horizon: int,
    t_horizon: int,
) -> torch.Tensor:
    """Return horizon waypoint indices for a batch of positions in a given map."""
    waypoints_xy_world = env._waypoints_list[map_level][:, :2]
    inner_xy_world = env._inner_list[map_level][:, :2]

    # Nearest waypoint index
    current_idx, _ = find_frenet_coord_along_waypoints(inner_xy_world, positions)

    if CONFIG['env_config']['DYNAMIC_LOOKAHEAD']:
        s_length  = CONFIG['env_config']['LEN_S_IDX']
        s_horizon = velocities * t_horizon
        idx_horizon = s_horizon / s_length
        horizon_indices = calculate_horizon_indices(
            current_idx=current_idx,
            idx_horizon=idx_horizon,
            waypoints_length=len(waypoints_xy_world),
            n_horizon=n_horizon,
            device=env.device
        )
    else:
        lookahead_steps = torch.linspace(0, delta_s_idx * n_horizon, n_horizon + 1, device=env.device)
        horizon_indices = (current_idx.unsqueeze(-1) + lookahead_steps.unsqueeze(0)) % len(waypoints_xy_world)
        horizon_indices = horizon_indices.long()

    return horizon_indices, waypoints_xy_world

def _init_env_waypoints(env):
    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, '_waypoints_list'):
        env._waypoints_list = [torch.tensor(wps, device=env.device, dtype=torch.float32)
                               for wps in env.scene.terrain.cfg.waypoints_list]
    if not hasattr(env, '_inner_list'):
        env._inner_list = [torch.tensor(inner, device=env.device, dtype=torch.float32)
                           for inner in env.scene.terrain.cfg.inner_list]
    if not hasattr(env, '_d_lat_list'):
        env._d_lat_list = [torch.tensor(d_lat, device=env.device, dtype=torch.float32)
                           for d_lat in env.scene.terrain.cfg.d_lat_list]
    if not hasattr(env, '_kappa_radpm_list'):
        env._kappa_radpm_list = [torch.tensor(kappa, device=env.device, dtype=torch.float32)
                                 for kappa in env.scene.terrain.cfg.kappa_radpm_list]

def track_info_horizon(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    delta_s_idx: int = 10,
    n_horizon: int = 5,
    t_horizon: int = 5,
    position_std_noise: float = 0.0
) -> torch.Tensor:
    """
    Compute multiple horizon-based observations in one pass:
      - deviation from centerline
      - heading error
      - lateral space (d_lat)
      - curvature (kappa)
    
    Returns:
        Tensor of shape [num_envs, stacked_dim], where stacked_dim =
            n_horizon (deviation)
          + n_horizon (heading)
          + n_horizon * 2 (d_lat left/right)
          + n_horizon (kappa)
    """
    # State
    pos_xy_world = mdp.root_pos_w(env)[..., :2]     
    # Add noise to each environment's position
    if position_std_noise > 0.0:
        # Create noise with the same shape as pos_xy_world: [num_envs, 2]
        noise = torch.normal(
            mean=0.0, 
            std=position_std_noise, 
            size=pos_xy_world.shape, 
            device=pos_xy_world.device
        )
        pos_xy_world = pos_xy_world + noise
        
    vel_x        = mdp.base_lin_vel(env)[..., 0]
    heading_w    = env.scene[asset_cfg.name].data.heading_w
    num_envs     = pos_xy_world.shape[0]

    # Init caches if needed
    _init_env_waypoints(env)

    # Output placeholders
    deviation_obs = torch.zeros(num_envs, n_horizon, device=env.device)
    heading_obs   = torch.zeros(num_envs, n_horizon, device=env.device)
    dlat_obs      = torch.zeros(num_envs, n_horizon*2, device=env.device)
    kappa_obs     = torch.zeros(num_envs, n_horizon, device=env.device)

    map_levels = env._map_levels
    for map_level in torch.unique(map_levels):
        mask = (map_levels == map_level)
        if mask.sum() == 0:
            continue

        # Positions, velocities, headings for this map batch
        map_pos   = pos_xy_world[mask]
        map_vel_x = vel_x[mask]
        map_head  = heading_w[mask]

        # Horizon indices
        horizon_indices, waypoints_xy_world = get_horizon_indices(
            env, map_level, map_pos, map_vel_x,
            delta_s_idx, n_horizon, t_horizon
        )

        # --- 1) Deviation from centerline ---
        seg_starts = waypoints_xy_world[horizon_indices[:, :-1]]
        seg_ends   = waypoints_xy_world[horizon_indices[:, 1:]]
        track_dirs = seg_ends - seg_starts
        car_offsets = map_pos.unsqueeze(1) - seg_starts
        cross_prods = (track_dirs[:,:,0]*car_offsets[:,:,1] - track_dirs[:,:,1]*car_offsets[:,:,0])
        signs = torch.sign(cross_prods)
        distances = torch.abs(cross_prods) / (torch.norm(track_dirs, dim=2) + 1e-6)
        deviation_obs[mask] = signs * distances/1.5

        # --- 2) Heading error ---
        lookahead_pts = waypoints_xy_world[horizon_indices[:, 1:]]
        desired_headings = torch.atan2(
            lookahead_pts[:,:,1] - map_pos[:,1].unsqueeze(-1),
            lookahead_pts[:,:,0] - map_pos[:,0].unsqueeze(-1),
        )
        heading_errors = torch.atan2(
            torch.sin(desired_headings - map_head.unsqueeze(-1)),
            torch.cos(desired_headings - map_head.unsqueeze(-1))
        )
        heading_obs[mask] = heading_errors/3.14

        # --- 3) Lateral space (d_lat) ---
        d_lat = env._d_lat_list[map_level][:, :2]
        next_dlat = d_lat[horizon_indices[:, 1:], :]
        dlat_obs[mask] = next_dlat.reshape(-1, n_horizon*2)/1.5

        # --- 4) Curvature (kappa) ---
        kappa_radpm = env._kappa_radpm_list[map_level][:]
        next_kappa = kappa_radpm[horizon_indices[:, 1:]].squeeze(-1)
        kappa_obs[mask] = next_kappa/3.0

    # Stack everything together
    return torch.cat([deviation_obs, heading_obs, dlat_obs, kappa_obs], dim=-1)

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
    if not hasattr(env, '_waypoints_list'):
        env._waypoints_list = [
            torch.tensor(wps, device=env.device, dtype=torch.float32)
            for wps in env.scene.terrain.cfg.waypoints_list
        ]
    if not hasattr(env, '_inner_list'):
        env._inner_list = [torch.tensor(inner, device=env.device, dtype=torch.float32)
                           for inner in env.scene.terrain.cfg.inner_list]    
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
        waypoints_xy_world = env._waypoints_list[map_level][:, :2]
        inner_xy_world = env._inner_list[map_level][:, :2]

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
        waypoints_xy_world = env._waypoints_list[map_level][:, :2]
        inner_xy_world = env._inner_list[map_level][:, :2]


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
    if not hasattr(env, '_d_lat_list'):
        env._d_lat_list = [
            torch.tensor(d_lat, device=env.device, dtype=torch.float32)
            for d_lat in env.scene.terrain.cfg.d_lat_list
        ]        
        
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
        waypoints_xy_world = env._waypoints_list[map_level][:, :2]

        
        inner_xy_world = env._inner_list[map_level][:, :2]


        d_lat = env._d_lat_list[map_level][:, :2]


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
    if not hasattr(env, '_kappa_radpm_list'):
        env._kappa_radpm_list = [
            torch.tensor(kappa, device=env.device, dtype=torch.float32)
            for kappa in env.scene.terrain.cfg.kappa_radpm_list
    ]
        
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
        waypoints_xy_world = env._waypoints_list[map_level][:, :2]
        inner_xy_world = env._inner_list[map_level][:, :2]
        kappa_radpm = env._kappa_radpm_list[map_level][:]


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
    if not hasattr(env, '_psi_rad_list'):
        env._psi_rad_list = [
            torch.tensor(psi, device=env.device, dtype=torch.float32)
            for psi in env.scene.terrain.cfg.psi_rad_list
    ]    
        
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
        waypoints_xy_world = env._waypoints_list[map_level][:, :2]
        inner_xy_world = env._inner_list[map_level][:, :2]

        psi_rad = env._psi_rad_list[map_level][:]


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

def opponent_relative_info_history(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    opponent_cfg: SceneEntityCfg = SceneEntityCfg("opponent"),
    position_std_noise: float = 0.0,
    velocity_std_noise: float = 0.0
) -> torch.Tensor:

    ego_pos_xy_world = mdp.root_pos_w(env=env, asset_cfg=asset_cfg)[..., :2]
    ego_heading_w = env.scene[asset_cfg.name].data.heading_w
    ego_vel_x = mdp.base_lin_vel(env=env, asset_cfg=asset_cfg)[..., 0]

    opp_pos_xy_world = mdp.root_pos_w(env=env, asset_cfg=opponent_cfg)[..., :2]
    # opp_vel_x = mdp.base_lin_vel(env=env, asset_cfg=opponent_cfg)[..., 0]

    if position_std_noise > 0.0:
        # Create noise with the same shape as pos_xy_world: [num_envs, 2]
        noise = torch.normal(
            mean=0.0, 
            std=position_std_noise, 
            size=opp_pos_xy_world.shape, 
            device=opp_pos_xy_world.device
        )
        opp_pos_xy_world = opp_pos_xy_world + noise
        
    if not hasattr(env, '_s_idx_diff_history'):
        env._s_idx_diff_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )
    if not hasattr(env, '_d_diff_history'):
        env._d_diff_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )
    if not hasattr(env, '_vx_diff_history'):
        env._vx_diff_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )
    if not hasattr(env, '_heading_diff_history'):
        env._heading_diff_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )
    if not hasattr(env, '_cross_pos_history'):
        env._cross_pos_history = torch.zeros(
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
    if not hasattr(env, '_opponent_vel_scaling_lvl'):
       env._opponent_vel_scaling_lvl = torch.zeros(
            env.num_envs,  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=env.device
        )       
    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
    if not hasattr(env, '_waypoints_list'):
        env._waypoints_list = [
            torch.tensor(wps, device=env.device, dtype=torch.float32)
            for wps in env.scene.terrain.cfg.waypoints_list
        ]
               
    opp_vel_x   = env._opponent_speed
    opp_heading = env._opponent_heading

    if velocity_std_noise > 0.0:
        # Create noise with the same shape as pos_xy_world: [num_envs, 2]
        noise = torch.normal(
            mean=0.0, 
            std=velocity_std_noise, 
            size=opp_vel_x.shape, 
            device=opp_vel_x.device
        )
        opp_vel_x = opp_vel_x + noise
        
    # Get map levels for all environments
    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)

    num_envs              = ego_pos_xy_world.shape[0]
    s_idx_diff_opp_ego    = torch.ones(num_envs, device=env.device, dtype= torch.float32)*CONFIG['env_config']['OPPONENT_INIT_DISTANCE_IDX_MAX']*CONFIG['env_config']['LEN_S_IDX']
    t_diff_opp_ego        = torch.zeros(num_envs, device=env.device, dtype= torch.float32)
    time_to_collision     = torch.zeros(num_envs, device=env.device, dtype= torch.float32)

    cross_pos_opp_ego     = torch.zeros(num_envs, device = env.device, dtype=torch.float32)
    d_diff_opp_ego        = torch.zeros(num_envs, device = env.device, dtype=torch.float32)
    is_behind             = torch.zeros(num_envs, device = env.device, dtype=torch.long)

    vx_diff_opp_ego       = torch.zeros(num_envs, device = env.device, dtype=torch.float32)
    opp_ego_heading_diff  = torch.zeros(num_envs, device = env.device, dtype=torch.float32)

    opp_d                 = torch.zeros(num_envs, device=env.device, dtype= torch.float32)

    # Process each map level separately

    for map_level in unique_map_levels:
        # Create mask for environments using this map
        env_mask = (map_levels == map_level)
        num_envs_in_map = env_mask.sum()
        
        if num_envs_in_map == 0:
            continue
            
        # Get positions for these environments
        ego_map_positions = ego_pos_xy_world[env_mask]
        ego_map_headings  = ego_heading_w[env_mask]
        ego_map_vel_x     = ego_vel_x[env_mask]
        opp_map_positions = opp_pos_xy_world[env_mask]
        opp_map_vel_x     = opp_vel_x[env_mask]
        opp_map_headings  = opp_heading[env_mask]
        
        # Get waypoints and track data for this map level
        waypoints_xy_world = env._waypoints_list[map_level][:, :2]
        
        # FRENET DISTANCES
        num_waypoints = len(waypoints_xy_world)
        ego_current_s_idx, ego_current_d = find_frenet_coord_along_waypoints(waypoints_xy_world, ego_map_positions)
        opp_current_s_idx, opp_current_d = find_frenet_coord_along_waypoints(waypoints_xy_world, opp_map_positions)
        
        opp_d[env_mask] = opp_current_d
        
        d_diff_opp_ego[env_mask]   = opp_current_d  - ego_current_d
        s_idx_diff_raw    = opp_current_s_idx - ego_current_s_idx
        s_idx_diff_signed = (s_idx_diff_raw + num_waypoints // 2) % num_waypoints - (num_waypoints // 2)
        s_idx_diff_opp_ego[env_mask] = s_idx_diff_signed*CONFIG['env_config']['LEN_S_IDX']

        # POSITIONING
        ego_map_heading_vec = torch.stack([torch.cos(ego_map_headings), torch.sin(ego_map_headings)], dim=1)
        dist = torch.norm(ego_map_positions - opp_map_positions, p=2, dim=1)
        ego_opp_vec = (opp_map_positions - ego_map_positions)
        ego_opp_positioning_vec = (ego_map_heading_vec[:, 0] * ego_opp_vec[:, 1] 
                                - ego_map_heading_vec[:, 1] * ego_opp_vec[:, 0]) / (dist + 1e-1)
        ego_opp_positioning_vec_norm = torch.clamp(ego_opp_positioning_vec, -1.0, 1.0)      
        cross_pos_opp_ego[env_mask] = ego_opp_positioning_vec_norm
  
        # RELATIVE VELOCITY
        vx_diff_opp_ego[env_mask] = opp_map_vel_x - ego_map_vel_x
        t_diff_opp_ego[env_mask] = s_idx_diff_opp_ego[env_mask]/(vx_diff_opp_ego[env_mask]+0.001)
        time_to_collision[env_mask] = torch.clamp(s_idx_diff_opp_ego[env_mask]/(-vx_diff_opp_ego[env_mask]+0.001), 0, 10)
        t_diff_opp_ego[env_mask] = torch.clamp(t_diff_opp_ego[env_mask], -10.0, 10.0)  

        # HEADING
        # Calculate smallest angle differences 
        opp_ego_heading_diff[env_mask] = torch.atan2(
            torch.sin(opp_map_headings - ego_map_headings),
            torch.cos(opp_map_headings - ego_map_headings)
        )

    env._s_idx_diff_history[:, 1:] = env._s_idx_diff_history[:, :-1].clone()
    env._s_idx_diff_history[:, 0] = s_idx_diff_opp_ego
 
    env._d_diff_history[:, 1:] = env._d_diff_history[:, :-1].clone()
    env._d_diff_history[:, 0] = d_diff_opp_ego

    env._vx_diff_history[:, 1:] = env._vx_diff_history[:, :-1].clone()
    env._vx_diff_history[:, 0] = vx_diff_opp_ego

    # env._heading_diff_history[:, 1:] = env._heading_diff_history[:, :-1].clone()
    # env._heading_diff_history[:, 0] = opp_ego_heading_diff                   

    env._cross_pos_history[:, 1:] = env._cross_pos_history[:, :-1].clone()
    env._cross_pos_history[:, 0] = cross_pos_opp_ego     
    ######
    
    env._opponent_d_history[:, 1:] = env._opponent_d_history[:, :-1].clone()
    env._opponent_d_history[:, 0] = opp_d
    
    return torch.cat([env._s_idx_diff_history.reshape(num_envs, -1), env._d_diff_history.reshape(num_envs, -1), env._vx_diff_history.reshape(num_envs, -1), env._cross_pos_history.reshape(num_envs, -1), env._opponent_d_history.reshape(num_envs, -1)],dim=1)

def gaps_info_history(
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
    if not hasattr(env, '_waypoints_list'):
        env._waypoints_list = [
            torch.tensor(wps, device=env.device, dtype=torch.float32)
            for wps in env.scene.terrain.cfg.waypoints_list
        ]
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

    if not hasattr(env, '_gap_inner_history'):
        env._gap_inner_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )
    if not hasattr(env, '_gap_outer_history'):
        env._gap_outer_history = torch.zeros(
            (env.num_envs, env._obs_history_length), 
            dtype=torch.float32,
            device=env.device
        )
             
    # Get map levels for all environments
    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)

    num_envs                    = ego_pos_xy_world.shape[0]
    gap_inner                   = torch.zeros(num_envs, device = env.device, dtype=torch.float32)
    gap_outer                   = torch.zeros(num_envs, device = env.device, dtype=torch.float32)
    # gap_inner_heading_error     = torch.zeros(num_envs, device = env.device, dtype=torch.float32)
    # gap_outer_heading_error     = torch.zeros(num_envs, device = env.device, dtype=torch.float32)

    # opp_detected          = torch.zeros(num_envs, device = env.device, dtype=torch.long)

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
        # ego_map_positions = ego_pos_xy_world[env_mask]
        # ego_map_headings =  ego_heading_w[env_mask]

        opp_map_positions = opp_pos_xy_world[env_mask]

        # Get waypoints and track data for this map level
        waypoints_xy_world = env._waypoints_list[map_level][:, :2]
        inner_xy_world = env._inner_list[map_level][:, :2]
        outer_xy_world = env._outer_list[map_level][:, :2]

        opp_current_inner_idx, inner_dist = find_frenet_coord_along_waypoints(inner_xy_world, opp_map_positions)
        opp_current_outer_idx, outer_dist = find_frenet_coord_along_waypoints(outer_xy_world, opp_map_positions)
    
        # gap_inner[env_mask] = torch.norm(opp_map_positions-inner_xy_world[opp_current_inner_idx, :], dim=1)
        # gap_outer[env_mask] = torch.norm(opp_map_positions-outer_xy_world[opp_current_outer_idx, :], dim=1)

        gap_inner[env_mask] = torch.abs(inner_dist)
        gap_outer[env_mask] = torch.abs(outer_dist)


    env._gap_inner_history[:, 1:] = env._gap_inner_history[:, :-1].clone()
    env._gap_inner_history[:, 0] = gap_inner                   
    
    env._gap_outer_history[:, 1:] = env._gap_outer_history[:, :-1].clone()
    env._gap_outer_history[:, 0] = gap_outer                   
    
    return torch.cat([env._gap_inner_history.reshape(num_envs,-1), env._gap_outer_history.reshape(num_envs,-1)],dim=1)

def opponent_frenet_coordinates_history(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    opponent_cfg: SceneEntityCfg = SceneEntityCfg("opponent")
) -> torch.Tensor:

    ego_pos_xy_world = mdp.root_pos_w(env=env, asset_cfg=asset_cfg)[..., :2]
    opp_pos_xy_world = mdp.root_pos_w(env=env, asset_cfg=opponent_cfg)[..., :2]
    # opp_vel_x = mdp.base_lin_vel(env=env, asset_cfg=opponent_cfg)[..., 0]
    
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
    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
    if not hasattr(env, '_waypoints_list'):
        env._waypoints_list = [
            torch.tensor(wps, device=env.device, dtype=torch.float32)
            for wps in env.scene.terrain.cfg.waypoints_list
        ]
        
    # Get map levels for all environments
    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)

    num_envs              = ego_pos_xy_world.shape[0]
    opp_current_s         = torch.zeros(num_envs, device=env.device, dtype= torch.float32)
    opp_current_d         = torch.zeros(num_envs, device=env.device, dtype= torch.float32)

    # Process each map level separately

    for map_level in unique_map_levels:
        # Create mask for environments using this map
        env_mask = (map_levels == map_level)
        num_envs_in_map = env_mask.sum()
        
        if num_envs_in_map == 0:
            continue

        opp_map_positions = opp_pos_xy_world[env_mask]
        
        # Get waypoints and track data for this map level
        waypoints_xy_world = env._waypoints_list[map_level][:, :2]
        
        # FRENET DISTANCES
        num_waypoints = len(waypoints_xy_world)
        opp_current_s_idx_map, opp_current_d_map = find_frenet_coord_along_waypoints(waypoints_xy_world, opp_map_positions)

        opp_current_s[env_mask] = (opp_current_s_idx_map+1)/num_waypoints
        opp_current_d[env_mask] = opp_current_d_map
        
    env._opponent_s_history[:, 1:] = env._opponent_s_history[:, :-1].clone()
    env._opponent_s_history[:, 0] = opp_current_s
 
    env._opponent_d_history[:, 1:] = env._opponent_d_history[:, :-1].clone()
    env._opponent_d_history[:, 0] = opp_current_d
    
    return torch.cat([env._opponent_s_history.reshape(num_envs, -1), env._opponent_d_history.reshape(num_envs, -1)],dim=1)


# def opponent_heading_error_horizon(
#     env: ManagerBasedEnv, 
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("opponent"),
#     delta_s_idx: int = 10,
#     n_horizon: int = 5,
#     t_horizon: int = 5
# ) -> torch.Tensor:
#     """
#     Calculate heading errors between car's current orientation and multiple delta_s_idx waypoints,
#     supporting multiple maps through env._map_levels.
    
#     Args:
#         env: The environment instance
#         asset_cfg: Configuration for the robot asset
#         delta_s_idx: Base number of waypoints to look ahead
#         n_horizon: Number of delta_s_idx points to return
        
#     Returns:
#         Tensor of heading errors in radians (range [-π, π]) for each n_horizon point.
#         Shape: [num_envs, n_horizon]
#     """
#     # Get current state
#     asset = env.scene[asset_cfg.name]
#     pos_xy_world = mdp.root_pos_w(env)[..., :2]
#     vel_x = mdp.base_lin_vel(env)[..., 0]

#     heading_w = asset.data.heading_w
#     num_envs = pos_xy_world.shape[0]

#     if not hasattr(env, '_map_levels'):
#         env._map_levels = torch.zeros(env.num_envs, 
#                                 dtype=torch.long,
#                                 device=env.device)
        
#     # Get map levels for all environments
#     map_levels = env._map_levels  # shape: [num_envs]
#     unique_map_levels = torch.unique(map_levels)
    
#     # Initialize output tensor
#     heading_errors = torch.zeros(num_envs, n_horizon, device=env.device)
    
#     # Process each map level separately
#     for map_level in unique_map_levels:
#         # Create mask for environments using this map
#         env_mask = (map_levels == map_level)
#         num_envs_in_map = env_mask.sum()
        
#         if num_envs_in_map == 0:
#             continue
            
#         # Get positions and headings for these environments
#         map_positions = pos_xy_world[env_mask]
#         map_vel_x    = vel_x[env_mask]
#         map_headings = heading_w[env_mask]
        
#         # Get waypoints for this map level
#         waypoints_xy_world = torch.tensor(
#             env.scene.terrain.cfg.waypoints_list[map_level],
#             device=env.device,
#             dtype=torch.float32
#         )[:, :2]
        
#         inner_xy_world = torch.tensor(
#             env.scene.terrain.cfg.inner_list[map_level],
#             device=env.device,
#             dtype=torch.float32
#         )[:, :2]

#         # Find nearest waypoint for these environments
#         current_idx, _ = find_frenet_coord_along_waypoints(inner_xy_world, map_positions)  # shape: [num_envs_in_map]
        
#         if CONFIG['env_config']['DYNAMIC_LOOKAHEAD']:
            
#             s_length  = CONFIG['env_config']['LEN_S_IDX'] # [m]
#             s_horizon = map_vel_x * t_horizon 
#             idx_horizon = s_horizon/s_length

#             horizon_indices = calculate_horizon_indices(
#                 current_idx=current_idx,
#                 idx_horizon=idx_horizon,
#                 waypoints_length=len(waypoints_xy_world),
#                 n_horizon=n_horizon,
#                 device=env.device
#             )
#         else:
#             lookahead_steps = torch.linspace(0, delta_s_idx*n_horizon, n_horizon+1, device=env.device)
#             horizon_indices = (current_idx.unsqueeze(-1) + lookahead_steps.unsqueeze(0)) % len(waypoints_xy_world)
#             horizon_indices = horizon_indices.long()


#         # Get delta_s_idx waypoints [num_envs_in_map, n_horizon, 2]
#         lookahead_points = waypoints_xy_world[horizon_indices[:, 1:]]
        
#         # Calculate desired heading vectors [num_envs_in_map, n_horizon]
#         desired_headings = torch.atan2(
#             lookahead_points[:, :, 1] - map_positions[:, 1].unsqueeze(-1),
#             lookahead_points[:, :, 0] - map_positions[:, 0].unsqueeze(-1)
#         )
        
#         # Calculate smallest angle differences [num_envs_in_map, n_horizon]
#         current_heading_errors = torch.atan2(
#             torch.sin(desired_headings - map_headings.unsqueeze(-1)),
#             torch.cos(desired_headings - map_headings.unsqueeze(-1))
#         )
        
#         # Store results in the output tensor
#         norm_heading_errors = 1
#         heading_errors[env_mask] = current_heading_errors / norm_heading_errors

#     return heading_errors


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