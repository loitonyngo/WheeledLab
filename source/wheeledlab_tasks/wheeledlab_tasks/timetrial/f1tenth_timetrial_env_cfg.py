from collections import Counter
import os
import time
from datetime import datetime

import torch
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, RigidObject, RigidObjectCfg, AssetBaseCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    CurriculumTermCfg as CurrTerm,
    CommandTermCfg as CmdTerm,
    SceneEntityCfg,
)
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs import ManagerBasedRLEnvCfg

from wheeledlab.envs.mdp import increase_reward_weight_over_time
from wheeledlab_assets import WHEELEDLAB_ASSETS_DATA_DIR
from wheeledlab_assets.mushr import MUSHR_SUS_CFG
from wheeledlab_assets.f1tenth import F1TENTH_CFG, OPPONENT_CFG, LB_CFG
from wheeledlab_tasks.common import Mushr4WDActionCfg
from wheeledlab_tasks.common import F1Tenth4WDActionCfg, LB4WDActionCfg
from .disable_lidar import disable_all_lidars

from .utils import create_maps_from_waypoints, generate_random_poses, generate_random_poses_from_list, generate_start_idx_poses_from_list, generate_random_poses_from_waypoints, TraversabilityHashmapUtil, find_frenet_coord_along_waypoints 
from . import mdp_sensors
from .mdp import reset_root_state_random, reset_root_state_random_opponent, reset_root_state_start_idx


import omni.usd

import yaml  # Add this import at the top of your file
from pathlib import Path
from typing import List  # For type hints
# Get the script's directory (/path/myscript/)
script_dir = Path(__file__).parent
# Navigate to the config file (go up one level, then into "config")
config_path = script_dir / "config" / "f1tenth_timetrial_config.yaml"
with open("/home/tongo/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/timetrial/config/f1tenth_timetrial_config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)


##############################
###### OBSERVATION #######
##############################

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

    opp_pos_xy_world = mdp.root_pos_w(env=env, asset_cfg=opponent_cfg)[..., :2]

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
        
    # Get map levels for all environments
    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)

    num_envs              = ego_pos_xy_world.shape[0]
    s_idx_diff_opp_ego    = -torch.ones(num_envs, device=env.device, dtype= torch.float32)*CONFIG['env_config']['OPPONENT_DETECTION_IDX']*CONFIG['env_config']['LEN_S_IDX']
    cross_pos_opp_ego     = -torch.ones(num_envs, device = env.device, dtype=torch.float32)
    opp_detected          = torch.zeros(num_envs, device = env.device, dtype=torch.long)

    # Process each map level separately

    num_episodes = env.common_step_counter // env.max_episode_length
    if num_episodes < CONFIG['env_config']['IGNORE_OPPONENT_EP']:
        return torch.stack([s_idx_diff_opp_ego, cross_pos_opp_ego, opp_detected],dim=1)

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
        
        num_waypoints = len(waypoints_xy_world)
        ego_current_s_idx, ego_current_d = find_frenet_coord_along_waypoints(waypoints_xy_world, ego_map_positions)
        opp_current_s_idx, opp_current_d = find_frenet_coord_along_waypoints(waypoints_xy_world, opp_map_positions)

        d_diff            = opp_current_d     - ego_current_d
        s_idx_diff_raw    = opp_current_s_idx - ego_current_s_idx
        s_idx_diff_signed = (s_idx_diff_raw + num_waypoints // 2) % num_waypoints - (num_waypoints // 2)

        # s_idx_diff_ot = torch.where((s_idx_diff_signed > 0) & (s_idx_diff_signed <= CONFIG['env_config']['OPPONENT_DETECTION_IDX']),
        #                         s_idx_diff_signed, 
        #                         torch.zeros_like(s_idx_diff_signed))  

        opp_detected[env_mask]       = torch.where(abs(s_idx_diff_signed) <= CONFIG['env_config']['OPPONENT_DETECTION_IDX'],
                                       torch.ones(len(env_mask), device=env.device, dtype= torch.long), 
                                       torch.zeros(len(env_mask), device=env.device, dtype= torch.long))  
        
        s_idx_diff_opp_ego[env_mask] = torch.where(opp_detected[env_mask].bool(),
                                       s_idx_diff_signed*CONFIG['env_config']['LEN_S_IDX'], 
                                       -torch.ones(len(env_mask), device=env.device, dtype= torch.float32)*CONFIG['env_config']['OPPONENT_DETECTION_IDX']*CONFIG['env_config']['LEN_S_IDX'])  
        # d_diff_opp_ego[env_mask]     = torch.where(opp_detected[env_mask].bool(),
        #                                d_diff, 
        #                                torch.zeros(len(env_mask), device=env.device, dtype= torch.float32))  


        ego_map_heading_vec = torch.stack([torch.cos(ego_map_headings), torch.sin(ego_map_headings)], dim=1)
        dist = torch.norm(ego_map_positions - opp_map_positions, p=2, dim=1)
        ego_opp_vec = (opp_map_positions - ego_map_positions)
        ego_opp_positioning_vec = (ego_map_heading_vec[:, 0] * ego_opp_vec[:, 1] 
                                - ego_map_heading_vec[:, 1] * ego_opp_vec[:, 0]) / (dist + 1e-6)
        ego_opp_positioning_vec_norm = torch.clamp(ego_opp_positioning_vec, -1.0, 1.0)
        
        cross_pos_opp_ego[env_mask] = torch.where(opp_detected[env_mask].bool(),
                                    ego_opp_positioning_vec_norm, 
                                    -torch.ones(len(env_mask), device=env.device, dtype= torch.float32)) 
        
    return torch.stack([s_idx_diff_opp_ego, cross_pos_opp_ego, opp_detected],dim=1)

def gap_overtake_info(
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

    gap_heading_error     = torch.zeros(num_envs, device = env.device, dtype=torch.float32)
    opp_detected          = torch.zeros(num_envs, device = env.device, dtype=torch.long)

    # Process each map level separately

    num_episodes = env.common_step_counter // env.max_episode_length
    if num_episodes < CONFIG['env_config']['IGNORE_OPPONENT_EP']:
        return torch.stack([gap_heading_error],dim=1)

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

        s_idx_diff_raw    = opp_current_s_idx - ego_current_s_idx
        s_idx_diff_signed = (s_idx_diff_raw + num_waypoints // 2) % num_waypoints - (num_waypoints // 2)

        opp_detected[env_mask]       = torch.where(abs(s_idx_diff_signed) <= CONFIG['env_config']['OPPONENT_DETECTION_IDX'],
                                       torch.ones(len(env_mask), device=env.device, dtype= torch.long), 
                                       torch.zeros(len(env_mask), device=env.device, dtype= torch.long))  
    
        gap_inner = torch.norm(opp_map_positions-inner_xy_world[opp_current_s_idx, :], dim=1)
        gap_outer = torch.norm(opp_map_positions-outer_xy_world[opp_current_s_idx, :], dim=1)
        gap_center = torch.where((gap_inner <= gap_outer).unsqueeze(-1),
                    (opp_map_positions + inner_xy_world[opp_current_s_idx, :])/2, 
                    (opp_map_positions + outer_xy_world[opp_current_s_idx, :])/2)
        # Calculate desired heading vectors [num_envs_in_map, n_horizon]
        desired_headings_gap = torch.atan2(
            gap_center[:, 1] - ego_map_positions[:, 1],
            gap_center[:, 0] - ego_map_positions[:, 0]
        )
        # Calculate smallest angle differences [num_envs_in_map, n_horizon]
        gap_heading_error[env_mask] = torch.where(opp_detected[env_mask].bool(),
                                                torch.atan2(torch.sin(desired_headings_gap - ego_map_headings),torch.cos(desired_headings_gap - ego_map_headings)),
                                                torch.zeros(len(env_mask), device=env.device, dtype= torch.float32))  
        
    return torch.stack([gap_heading_error],dim=1)

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

##########################
# Variables for observation space. Better way to implement it?

N_HORIZON = CONFIG['env_config']['N_HORIZON'] 
DELTA_S_IDX = CONFIG['env_config']['DELTA_S_IDX'] 

T_HORIZON = CONFIG['env_config']['T_HORIZON']
##########################

@configclass
class F1TenthTimeTrialObsCfg:
    """Observation specifications for the environment."""
    @configclass
    class PolicyCfg(ObsGroup):
        """
        [vx, wz, action1(vel), action2(steering), ...]
        """
        # lidar = ObsTerm(func=mdp_sensors.lidar_ranges, params={"sensor_cfg":SceneEntityCfg("lidar")})
        base_lin_vel_x_history = ObsTerm(
            func=base_lin_vel_x_history, 
            params={'mean_noise': 0,
                    'std_noise': 0}            
            )

        base_lin_vel_y_history = ObsTerm(
            func=base_lin_vel_y_history, 
            params={'mean_noise': 0,
                    'std_noise': 0}            
            )
                
        base_ang_vel_z_history = ObsTerm(
            func=base_ang_vel_z_history, 
            params={'mean_noise': 0,
                    'std_noise': 0}         
            )

        target_velocity_history = ObsTerm(
            func=target_velocity_history, 
            params={'mean_noise': 0,
                    'std_noise': 0}         
            )
             
        # last_action = ObsTerm(
        #     func=mdp.last_action,
        #     clip=(-1., 1.), # TODO: get from ClipAction wrapper
        #     noise=Unoise(n_min=-.0, n_max=.0, operation='add')
        # )
        
        action_history = ObsTerm(
            func=action_history,
        )

        heading_error = ObsTerm(
            func=heading_error_horizon,
            params={'delta_s_idx': DELTA_S_IDX,
                    'n_horizon': N_HORIZON,
                    't_horizon': T_HORIZON}
        )
        deviation_error = ObsTerm(
            func=deviation_centerline_horizon,
            params={'delta_s_idx': DELTA_S_IDX,
                    'n_horizon': N_HORIZON,
                    't_horizon': T_HORIZON}
        )
        d_lat_horizon = ObsTerm(
            func=d_lat_horizon,
            params={'delta_s_idx': DELTA_S_IDX,
                    'n_horizon': N_HORIZON,
                    't_horizon': T_HORIZON}
        )
        #only one env gives problem
        kappa_radpm_horizon = ObsTerm(
            func=kappa_radpm_horizon,
            params={'delta_s_idx': DELTA_S_IDX,
                    'n_horizon': N_HORIZON,
                    't_horizon': T_HORIZON}
        )

        # opponent_frenet_info = ObsTerm(
        #     func=opponent_frenet_info
        # )
        
        # gap_overtake_info = ObsTerm(
        #     func=gap_overtake_info
        # )
        # delta_psi_rad_horizon = ObsTerm(
        #     func=delta_psi_rad_horizon,
        #     params={'delta_s_idx': delta_s_idx,
        #             'n_horizon': N_STEP_LOOKAHEAD}
        # )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True


    policy: PolicyCfg = PolicyCfg()

@configclass
class InitialPoseCfg:
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot_euler_xyz_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)


##############################
###### TERRAIN / TRACK #######
##############################
DYNAMIC_FRICTION = CONFIG['env_config']['DYNAMIC_FRICTION']
STATIC_FRICTION = CONFIG['env_config']['STATIC_FRICTION']
RESTITUTION = CONFIG['env_config']['RESTITUTION']

@configclass
class F1TenthTimeTrialTerrainImporterCfg(TerrainImporterCfg):
    # Declare variables without initialization
    map_name_list: list = None
    origin_list: list = None

    traversability_hashmap_list: list = None
    waypoints_list: list = None
    outer_list: list = None
    inner_list: list = None
    d_lat_list: list = None
    psi_rad_list: list = None
    kappa_radpm_list: list = None
    vx_mps_list: list = None
    spacing_meters_list: list = None
    map_size_pixels_list: list = None
    row_spacing_list: list = None
    col_spacing_list: list = None
    num_cols_list: list = None
    num_rows_list: list = None
    width_list: list = None
    height_list: list = None

    # Other configurations that don't depend on runtime values
    env_spacing = 0
    prim_path = "/World/ground"
    terrain_type = "usd"
    usd_path = None  # Will be set in post_init
    collision_group = -1
    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="max",
        static_friction=STATIC_FRICTION,
        dynamic_friction=DYNAMIC_FRICTION,
        restitution=RESTITUTION
    )
    debug_vis = True
    
    def generate_random_poses_from_waypoints(self, env : ManagerBasedEnv, env_ids, num_poses, max_radius_offset=0.3):
        
        # generate random initial poses with margin
        env_origins = env.scene.env_origins
        map_levels = env._map_levels
        # add which map level

        init_poses, init_current_wps_idx = generate_random_poses_from_waypoints(env_ids, num_poses, map_levels, env_origins, self.waypoints_list, self.inner_list, max_radius_offset=0.3)
        # init_poses, init_current_wps_idx = generate_random_poses_from_list(env_ids, num_poses, map_levels, env_origins, self.origin_list, self.row_spacing_list, self.col_spacing_list, self.traversability_hashmap_list, self.waypoints_list, self.outer_list, self.inner_list, margin=0.1)
        max_radius_offset = 0.5
        valid_init_poses = [
            InitialPoseCfg(
                pos=(x + random.uniform(-1,1)*max_radius_offset, y + random.uniform(-1,1)*max_radius_offset, 0.02),
                rot_euler_xyz_deg=(0., 0., angle)
            ) for x, y, angle in init_poses
        ]
        return valid_init_poses, init_current_wps_idx

    def generate_start_idx_poses(self, env : ManagerBasedEnv, env_ids, num_poses):
        
        # generate random initial poses with margin
        env_origins = env.scene.env_origins
        map_levels = env._map_levels
        # add which map level

        init_poses, init_current_wps_idx = generate_start_idx_poses_from_list(env_ids, num_poses, map_levels, env_origins, self.row_spacing_list, self.col_spacing_list, self.traversability_hashmap_list, self.waypoints_list, self.outer_list, self.inner_list, margin=0.1)
        valid_init_poses = [
            InitialPoseCfg(
                pos=(x, y, 0.02),
                # rot_euler_xyz_deg=(0., 0., angle)
                rot_euler_xyz_deg=(0., 0., 0)
            ) for x, y, angle in init_poses
        ]
        return valid_init_poses, init_current_wps_idx
         
    """
    Get traversability value of an x, y coordinate
    """
    # def get_traversability(self, poses):
    #     traversability = []
    #     xs, ys = poses[:, 0], poses[:, 1]
    #     x_idx, y_idx = self.get_map_id(xs, ys)
    #     traversability = torch.tensor(self.traversability_hashmap).to(x_idx.device)[x_idx, y_idx]
    #     return traversability
    
    """
    Helper function to get the map id given x, y coordinates
    """
    # def get_map_id(self, x, y):
    #     x_idx = torch.floor((x + self.width/2 - self.row_spacing/2) / self.row_spacing).long()
    #     y_idx = torch.floor((y + self.height/2 - self.col_spacing/2) / self.col_spacing).long()
    #     x_idx = torch.clamp(x_idx, 0, self.num_rows-1)
    #     y_idx = torch.clamp(y_idx, 0, self.num_cols-1)
    #     return x_idx, y_idx


@configclass
class F1TenthTimeTrialSceneCfg(InteractiveSceneCfg):
    """Configuration for a Mushr car Scene with racetrack terrain and Sensors."""

    terrain = None
    MAP_NAME_LIST = None
    ground = AssetBaseCfg(
        prim_path="/World/base",
        spawn = sim_utils.GroundPlaneCfg(size=(1000, 1000),
                                         color=(0,0,0),
                                         physics_material=sim_utils.RigidBodyMaterialCfg(
                                            friction_combine_mode="multiply",
                                            restitution_combine_mode="multiply",
                                            static_friction=STATIC_FRICTION,
                                            dynamic_friction=DYNAMIC_FRICTION,
                                         ),
        )
    )

    # Add light configuration
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.5, 0.5, 0.5), intensity=1500.0),
    )

    robot: AssetBaseCfg = F1TENTH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # opponent: AssetBaseCfg = OPPONENT_CFG.replace(prim_path="{ENV_REGEX_NS}/Opponent")
    # robot: ArticulationCfg = MUSHR_SUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # robot: ArticulationCfg = LB_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
       # Add cuboid configuration
    # opponent = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Opponent",
    #     spawn=sim_utils.CuboidCfg(
    #         size = (0.6, 0.35, 0.3),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled= False, 
    #             rigid_body_enabled=True,
    #             solver_position_iteration_count=4,
    #             solver_velocity_iteration_count=1,
    #             max_angular_velocity=100.0,
    #             max_linear_velocity=100.0,
    #             max_depenetration_velocity=0.00001,
    #             disable_gravity=True,
    #         ),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=0.5,
    #             dynamic_friction=0.5,
    #             restitution=0.5,
    #         ),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),  # 10cm above ground
    # )

    ground.init_state.pos = (0.0, 0.0, -1e-4)

    def __post_init__(self):
        """Post intialization."""
        super().__post_init__()

        self.robot.init_state = self.robot.init_state.replace(
            pos=(0.0, 0.0, 0.0)
        )

        # self.opponent.init_state = self.opponent.init_state.replace(
        #     pos=(0.0, 0.0, 0.0)
        # )

#####################
###### EVENTS #######
#####################
def store_data(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: RigidObject = env.scene[asset_cfg.name]
    waypoints = torch.tensor(env.scene.terrain.cfg.waypoints_list[0], 
                        device=env.device)[:, :2]
    position_xy = mdp.root_pos_w(env)[..., :2]

    # Find nearest waypoint (vectorized)
    current_idx, _ = find_frenet_coord_along_waypoints(waypoints, position_xy)
    num_waypoints = len(waypoints)

    #there should be a more elegant way to store these...
    env.extras['v_x'] = asset.data.root_lin_vel_b[:,0]
    env.extras['s_idx'] = current_idx.clone()
    env.extras['time'] = torch.tensor(env.sim.current_time, device=env.device)
    env.extras['s_idx_max'] = torch.tensor(num_waypoints-1, device=env.device)


@configclass
class F1TenthTimeTrialEventsCfg:
    
    # on startup
    if CONFIG['env_config']['RESET_RANDOM']:
        reset_root_state_random = EventTerm(
            func=reset_root_state_random,
            mode="reset",
        )

        # reset_root_state_random_opponent = EventTerm(
        #     func=reset_root_state_random_opponent,
        #     mode="reset",
        # )
    else:
        reset_root_state_start_idx = EventTerm(
            func=reset_root_state_start_idx,
            mode="reset",
        )

    # if CONFIG['env_config']['VD_ENHANCED']:
    #     enhanced_braking = EventTerm(
    #         func=mdp.enhanced_braking,
    #         params={'k_p': 1,
    #                 'k_d': 0.2,
    #                 'min_speed_correction': -0.75},
    #         mode="interval",
    #         interval_range_s=(0.05, 0.05)
            
    #     )

    #     enhanced_tc = EventTerm(
    #         func=mdp.enhanced_tc,
    #         params={'k_p': 1,
    #                 'k_d': 0.1,
    #                 'k_i': 0.0,
    #                 'max_speed_correction': +0.40,
    #                 'tc_coefficient': 0.5},
    #         mode="interval",
    #         interval_range_s=(0.05, 0.05) 
    #     )

    #     enhanced_rotation = EventTerm(
    #         func=mdp.enhanced_rotation,
    #         params={'k_p': 0.1,
    #                 'k_d': 0.25,
    #                 'k_i': 0.3
    #                 },
    #         mode="interval",
    #         interval_range_s=(0.05, 0.05)
    #     )

    # enhanced_vy = EventTerm(
    #     func=mdp.enhanced_vy,
    #     params={'k_p': 0.5,
    #             'k_d': 0.5,
    #             'k_i': 0
    #             },
    #     mode="interval",
    #     interval_range_s=(0.025, 0.025)
    # )

    # store_data = EventTerm( 
    #     func= store_data,
    #     mode="interval",
    #     interval_range_s=(0.025, 0.025),
    #     params={
    #     },
    # )

    # def update_history_buffer(
    #     env: ManagerBasedEnv,
    #     env_ids: torch.Tensor,
    #     # valid_posns_and_rots: dict[str, tuple[float, float]],
    #     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # ):
        

@configclass
class F1TenthTimeTrialEventsRandomCfg(F1TenthTimeTrialEventsCfg):
    # change_wheel_friction = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "static_friction_range": (0.0, 0.0),
    #         "dynamic_friction_range": (0.0, 0.0),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 10,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_.*link"),
    #         "make_consistent": False,
    #     },
    # )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
    #         "mass_distribution_params": (0.0, 0.0),
    #         "operation": "abs",
    #     },
    # )

    # add_wheel_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_.*link"),
    #         "mass_distribution_params": (.0, 0.0),
    #         "operation": "abs",
    #     },
    # )

    # Override randomize_gains to target all four wheel motors (front and back)
    # randomize_gains = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_(back|front)_.*"]),
    #         "damping_distribution_params": (0.0, 0.0),
    #         "operation": "abs",
    #     },
    # )

    # change_wheel_friction = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "static_friction_range": (STATIC_FRICTION-0.1, STATIC_FRICTION+0.1),
    #         "dynamic_friction_range": (DYNAMIC_FRICTION-0.1, DYNAMIC_FRICTION+0.1),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 20,
    #         "asset_cfg": SceneEntityCfg("robot", body_names="wheel.*"),
    #         "make_consistent": True,
    #     },
    # )

    kill_lidar = EventTerm(
        func=disable_all_lidars,
        mode="startup",
        params={}          
    )


######################
###### REWARDS #######
######################

def wall_collision_penalty(env):
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
    num_envs = pos_xy_world.shape[0]

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
        
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

    return torch.where(collision_bool.bool(), -1, 0)
    # return torch.where(collision_bool.bool(), -1, 0)


def opponent_collision_penalty(env):
    num_episodes = env.common_step_counter // env.max_episode_length
    if num_episodes < CONFIG['env_config']['IGNORE_OPPONENT_EP']:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    
    ego_position_xy = env.scene["robot"].data.root_pos_w[:, :2]
    opp_position_xy = env.scene["opponent"].data.root_pos_w[:, :2]
    dist = torch.norm(ego_position_xy - opp_position_xy, p=2, dim=1)
    opp_collision = dist < CONFIG['env_config']['OPP_COLLISION_RADIUS']

    return torch.where(opp_collision.bool(), -1*forward_vel(env)**2, 0)
    # return torch.where(opp_collision.bool(), -1, 0)

def opponent_overtake_closing_reward(env):
    num_episodes = env.common_step_counter // env.max_episode_length
    if num_episodes < CONFIG['env_config']['IGNORE_OPPONENT_EP']:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    
    if not hasattr(env, '_prev_delta_s_opp_ego'):
        env._prev_delta_s_opp_ego = torch.ones(env.num_envs, 
                                dtype=torch.float32,
                                device=env.device)*CONFIG['env_config']['OPPONENT_DETECTION_IDX']
        
    ego_position_xy = env.scene["robot"].data.root_pos_w[:, :2]
    opp_position_xy = env.scene["opponent"].data.root_pos_w[:, :2]
    dist = torch.norm(ego_position_xy - opp_position_xy, p=2, dim=1)

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
    if not hasattr(env, '_progress_history_indices'):
        env._progress_history_length = CONFIG['env_config']['PROGRESS_HISTORY_LENGTH']  # Store last 10 waypoints

        env._progress_history_indices = torch.zeros(
            (env.num_envs, env._progress_history_length), 
            dtype=torch.long,
            device=env.device
        )
        env._reset_env_bool = torch.ones(  # Tracks where to insert the next index
            env.num_envs,
            dtype=torch.bool,
            device=env.device
        )
    if not hasattr(env, '_progress_history_checkpoint_idx'):
        env._progress_history_checkpoint_idx = CONFIG['env_config']['PROGRESS_HISTORY_CHECKPOINT_IDX']
    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)
    
    # Initialize outputs
    delta_s_opp_ego       = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    delta_delta_s_opp_ego = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    ego_behind_opp_bool  = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    detection_opp_bool = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    far_way_opp_bool   = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
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
        delta_delta_s_opp_ego[env_mask] = env._prev_delta_s_opp_ego[env_mask] - delta_s_opp_ego[env_mask]
        
        detection_opp_bool[env_mask] = abs(delta_s_opp_ego[env_mask]) < CONFIG['env_config']['OPPONENT_DETECTION_IDX']
        ego_behind_opp_bool[env_mask] = delta_s_opp_ego[env_mask] > 0
        far_way_opp_bool[env_mask]   = abs(delta_s_opp_ego[env_mask]) > CONFIG['env_config']['OPPONENT_FAR_AWAY_IDX']
        env._prev_delta_s_opp_ego[env_mask] = delta_s_opp_ego[env_mask]

    far_way_opp_bool_mask = far_way_opp_bool == 1
    env._prev_delta_s_opp_ego[far_way_opp_bool_mask] = torch.ones(far_way_opp_bool_mask.sum().item(),
                                dtype=torch.float32,
                                device=env.device)*CONFIG['env_config']['OPPONENT_DETECTION_IDX']

    if not hasattr(env, '_traversability_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._traversability_history = torch.ones(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
    no_off_track = env._traversability_history.min(dim=1).values == 1

    return torch.where(detection_opp_bool & no_off_track & ego_behind_opp_bool, delta_delta_s_opp_ego*CONFIG['env_config']['LEN_S_IDX'], 0)

def opponent_overtake_positioning_reward(env):
    num_episodes = env.common_step_counter // env.max_episode_length
    if num_episodes < CONFIG['env_config']['IGNORE_OPPONENT_EP']:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    
    if not hasattr(env, '_prev_delta_s_opp_ego'):
        env._prev_delta_s_opp_ego = torch.ones(env.num_envs, 
                                dtype=torch.float32,
                                device=env.device)*CONFIG['env_config']['OPPONENT_DETECTION_IDX']
        
    ego_position_xy = env.scene["robot"].data.root_pos_w[:, :2]
    ego_heading_w = env.scene["robot"].data.heading_w
    ego_heading_vec = torch.stack([torch.cos(ego_heading_w), torch.sin(ego_heading_w)], dim=1)

    opp_position_xy = env.scene["opponent"].data.root_pos_w[:, :2]
    dist = torch.norm(ego_position_xy - opp_position_xy, p=2, dim=1)

    ego_opp_vec = (opp_position_xy - ego_position_xy)
    
    ego_opp_positioning_vec = (ego_heading_vec[:, 0] * ego_opp_vec[:, 1] 
                            - ego_heading_vec[:, 1] * ego_opp_vec[:, 0]) / (dist + 1e-6)
    ego_opp_positioning_vec_norm = torch.clamp(ego_opp_positioning_vec, -1.0, 1.0)
    
    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
    if not hasattr(env, '_progress_history_indices'):
        env._progress_history_length = CONFIG['env_config']['PROGRESS_HISTORY_LENGTH']  # Store last 10 waypoints

        env._progress_history_indices = torch.zeros(
            (env.num_envs, env._progress_history_length), 
            dtype=torch.long,
            device=env.device
        )
        env._reset_env_bool = torch.ones(  # Tracks where to insert the next index
            env.num_envs,
            dtype=torch.bool,
            device=env.device
        )
    if not hasattr(env, '_progress_history_checkpoint_idx'):
        env._progress_history_checkpoint_idx = CONFIG['env_config']['PROGRESS_HISTORY_CHECKPOINT_IDX']
    map_levels = env._map_levels  # shape: [num_envs]
    unique_map_levels = torch.unique(map_levels)
    
    # Initialize outputs
    delta_s_opp_ego = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    delta_d_opp_ego = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    ego_behind_opp_bool  = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    detection_opp_bool = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    far_way_opp_bool   = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
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
        
        detection_opp_bool[env_mask]  = abs(delta_s_opp_ego[env_mask]) < CONFIG['env_config']['OPPONENT_DETECTION_IDX']
        ego_behind_opp_bool[env_mask] = delta_s_opp_ego[env_mask] > 0
        far_way_opp_bool[env_mask]    = abs(delta_s_opp_ego[env_mask]) > CONFIG['env_config']['OPPONENT_FAR_AWAY_IDX']
        env._prev_delta_s_opp_ego[env_mask] = delta_s_opp_ego[env_mask]

    far_way_opp_bool_mask = far_way_opp_bool == 1
    env._prev_delta_s_opp_ego[far_way_opp_bool_mask] = torch.ones(far_way_opp_bool_mask.sum().item(),
                                dtype=torch.float32,
                                device=env.device)*CONFIG['env_config']['OPPONENT_DETECTION_IDX']

    if not hasattr(env, '_traversability_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._traversability_history = torch.ones(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
    no_off_track = env._traversability_history.min(dim=1).values == 1

    positioning_reward = torch.abs(ego_opp_positioning_vec_norm)*torch.exp(-delta_s_opp_ego*0.05)
    
    return torch.where(detection_opp_bool & no_off_track & ego_behind_opp_bool, positioning_reward, 0)

# def traversable_reward(env):
#     poses =mdp.root_pos_w(env)[..., :2]
#     if not hasattr(env, '_map_levels'):
#         env._map_levels = torch.zeros(env.num_envs, 
#                                 dtype=torch.long,
#                                 device=env.device)
#     map_levels = env._map_levels
#     traversability = TraversabilityHashmapUtil().get_traversability(poses, map_levels)
#     return torch.where(traversability, 1, 0.)

# def out_of_track_penalty(env):
#     poses =mdp.root_pos_w(env)[..., :2]
#     if not hasattr(env, '_map_levels'):
#         env._map_levels = torch.zeros(env.num_envs, 
#                                 dtype=torch.long,
#                                 device=env.device)
#     map_levels = env._map_levels
#     traversability = TraversabilityHashmapUtil().get_traversability(poses, map_levels)

#     return torch.where(traversability, 0., -1.)

def upright_penalty(env, thresh_deg):
    rot_mat = math_utils.matrix_from_quat(mdp.root_quat_w(env))
    up_dot = rot_mat[:, 2, 2]
    up_dot = torch.rad2deg(torch.arccos(up_dot))
    penalty = torch.where(up_dot > thresh_deg, up_dot - thresh_deg, 0.)
    return penalty

def off_track(env, straight, corner_out_radius):
    poses = mdp.root_pos_w(env)
    penalty = torch.where(torch.abs(poses[...,1]) < straight,
                torch.where(torch.abs(poses[...,0]) > corner_out_radius, 1, 0),
                torch.where(poses[...,1] > 0,
                    torch.where((poses[...,1] - straight)**2 + poses[...,0]**2 > corner_out_radius**2, 1, 0),
                    torch.where((poses[...,1] + straight)**2 + poses[...,0]**2 > corner_out_radius**2, 1, 0)))
    return 

def low_speed_penalty(env, low_speed_thresh: float=0.3):
    lin_speed = torch.norm(mdp.base_lin_vel(env), dim=-1)
    pen = torch.where(lin_speed < low_speed_thresh, 1., 0.)
    return pen

def forward_vel(env):
    return mdp.base_lin_vel(env)[:, 0]

def progress_rew(env):
    """Reward for passing each new waypoint, handling lap transitions."""
    progress_bool, progress = progress_waypoint_bool(env)

    if not hasattr(env, '_traversability_history'):
        env._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']
        env._traversability_history = torch.ones(
            (env.num_envs, env._rew_history_length), 
            dtype=torch.long,
            device=env.device
        )
    no_off_track = env._traversability_history.min(dim=1).values == 1

    return torch.where(progress_bool & no_off_track, progress*0.1, 0.0)

def progress_waypoint_bool(env):
    # Initialize buffer if first run
    if not hasattr(env, '_progress_history_indices'):
        env._progress_history_length = CONFIG['env_config']['PROGRESS_HISTORY_LENGTH']  # Store last 10 waypoints

        env._progress_history_indices = torch.zeros(
            (env.num_envs, env._progress_history_length), 
            dtype=torch.long,
            device=env.device
        )
        env._reset_env_bool = torch.ones(  # Tracks where to insert the next index
            env.num_envs,
            dtype=torch.bool,
            device=env.device
        )
    if not hasattr(env, '_progress_history_checkpoint_idx'):
        env._progress_history_checkpoint_idx = CONFIG['env_config']['PROGRESS_HISTORY_CHECKPOINT_IDX']

    # Get current positions and map levels
    position_xy_world = mdp.root_pos_w(env)[..., :2]
    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                dtype=torch.long,
                                device=env.device)
    
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
        waypoints_world = torch.tensor(
            env.scene.terrain.cfg.waypoints_list[map_level], 
            device=env.device
        )[:, :2]
        
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

    env.extras['yaw_rate'] = asset.data.root_ang_vel_b[:,2]
    env.extras['s_idx'] = current_idx.clone()
    env.extras['time'] = torch.tensor(env.sim.current_time, device=env.device)
    env.extras['s_idx_max'] = torch.tensor(num_waypoints, device=env.device)
    env.extras['throttle_joints_applied_effort'] = asset.actuators['throttle_joints'].applied_effort
    ###########################
    
    env.extras['vel_y_calc'] = env._vel_y_calc


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
    return torch.where(steering_diff > 0., steering_diff, 0.) # speed target


####### F1TenthTimeTrial Environment #######
@configclass
class F1TenthTimeTrialRewardsCfg:
    # """Reward terms for the MDP."""
    # Set "weight" to 0 to deactivate a reward term

    # Penalty if the car goes off-track (it would be crashing on the walls), weight=1
    # out_of_track = RewTerm(
    #     func=out_of_track_penalty,
    #     weight=1,
    # )

    # Standard reward for progressing along centerline, weight=1
    progress_rew = RewTerm(
        func=progress_rew,
        weight=1.0,
    )
    
    wall_collision_penalty = RewTerm(
        func=wall_collision_penalty,
        weight=1,
    )

    var_throttle_penalty =  RewTerm(
        func=var_throttle_penalty,
        weight=0.03,
    )

    var_throttle_rate_penalty =  RewTerm(
        func=var_throttle_rate_penalty,
        weight=0.00,
    )
    # delta_throttle_l2_penalty =  RewTerm(
    #     func=delta_throttle_l2_penalty,
    #     weight=0.0,
    # )

    var_steering_penalty =  RewTerm(
        func=var_steering_penalty,
        weight=0.5,
    )

    effort_throttle_penalty =  RewTerm(
        func=effort_throttle_penalty,
        weight=0.01,
    )
    
    effort_steering_penalty =  RewTerm(
        func=effort_steering_penalty,
        weight=0.05,
    )
    # delta_steering_l2_penalty =  RewTerm(
    #     func=delta_steering_l2_penalty,
    #     weight=0.0,
    # )
    
    # delta_speed_cmd_penalty =  RewTerm(
    #     func=delta_speed_cmd_penalty,
    #     weight=0.000,
    # )
    
    # opponent_overtake_closing_reward = RewTerm(
    #     func=opponent_overtake_closing_reward,
    #     weight=1,
    # )

    # opponent_overtake_positioning_reward = RewTerm(
    #     func=opponent_overtake_positioning_reward,
    #     weight=0.01,
    # )
    
    # opponent_collision_penalty = RewTerm(
    #     func=opponent_collision_penalty,
    #     weight=20,
    # )


    if CONFIG['env_config']['CONSTANT_SPEED']:
        # # # Reward terms to test various frictions, simple task (constant velocity and steering, drive in circle)
        speed_target_rew = RewTerm(
            func=speed_target_rew,
            params={
                "speed_target": CONFIG['env_config']['CONSTANT_SPEED_TARGET']
            },
            weight= 1.,
        )

########################
###### CURRICULUM ######
########################

@configclass
class TimeTrialCurriculumCfg:

    wall_collision_penalty = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "wall_collision_penalty",
            "weight_increase": 0,
            "first_episode_increase": 50,
            "episodes_per_increase": 50,
            "max_num_increases": 0,
        }
    )

    var_throttle_penalty = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "var_throttle_penalty",
            "weight_increase": 0.01,
            "first_episode_increase": 4,
            "episodes_per_increase": 4,
            "max_num_increases": 0,
        }
    )
    
    # delta_throttle_l2_penalty = CurrTerm(
    #     func=increase_reward_weight_over_time,
    #     params={
    #         "reward_term_name": "delta_throttle_l2_penalty",
    #         "weight_increase": 0.2,
    #         "first_episode_increase": 3,
    #         "episodes_per_increase": 3,
    #         "max_num_increases": 0,
    #     }
    # )

    var_steering_penalty = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "var_steering_penalty",
            "weight_increase": 0.01,
            "first_episode_increase": 4,
            "episodes_per_increase": 4,
            "max_num_increases": 0,
        }
    )
    
    # effort_steering_penalty = CurrTerm(
    #     func=increase_reward_weight_over_time,
    #     params={
    #         "reward_term_name": "effort_steering_penalty",
    #         "weight_increase": 0.001,
    #         "first_episode_increase": 4,
    #         "episodes_per_increase": 5,
    #         "max_num_increases": 1,
    #     }
    # )
        
    # delta_steering_l2_penalty = CurrTerm(
    #     func=increase_reward_weight_over_time,
    #     params={
    #         "reward_term_name": "delta_steering_l2_penalty",
    #         "weight_increase": 0.2,
    #         "first_episode_increase": 3,
    #         "episodes_per_increase": 3,
    #         "max_num_increases": 0,
    #     }
    # )

    # delta_speed_cmd_penalty = CurrTerm(
    #     func=increase_reward_weight_over_time,
    #     params={
    #         "reward_term_name": "delta_speed_cmd_penalty",
    #         "weight_increase": 0.01,
    #         "first_episode_increase": 16,
    #         "episodes_per_increase": 4,
    #         "max_num_increases": 10,
    #     }
    # )
    
    # less_traversability = CurrTerm(
    #     func=increase_reward_weight_over_time,
    #     params={
    #         "reward_term_name": "traversablility",
    #         "increase": -0.25,
    #         "first_episode_increase": 25,
    #         "episodes_per_increase": 25,
    #         "max_num_increases": 2,
    #     }
    # )

##########################
###### TERMINATION #######
##########################

# def out_of_map(env):
#     poses = mdp.root_pos_w(env)
#     poses = poses[..., :2]
#     terrain = env.scene[SceneEntityCfg("terrain").name]
#     width = terrain.cfg.width_list[0]
#     height = terrain.cfg.height_list[0]
#     x_out_range = torch.logical_or(poses[..., 0] > width / 2, poses[..., 0] < -width / 2)
#     y_out_range = torch.logical_or(poses[..., 1] > height / 2, poses[..., 1] < -height / 2)
#     return torch.logical_or(x_out_range, y_out_range)


def upright_bool(env, thresh_deg):
    return upright_penalty(env, thresh_deg) > 0.0


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

    return collision_bool.bool()

def opponent_collision(env):
    num_episodes = env.common_step_counter // env.max_episode_length
    if num_episodes <  CONFIG['env_config']['IGNORE_OPPONENT_EP']:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    
    if not hasattr(env, '_prev_delta_s_opp_ego'):
        env._prev_delta_s_opp_ego = torch.ones(env.num_envs, 
                                dtype=torch.float32,
                                device=env.device)*CONFIG['env_config']['OPPONENT_DETECTION_IDX']*CONFIG['env_config']['LEN_S_IDX']
        
        
    ego_position_xy = env.scene["robot"].data.root_pos_w[:, :2]
    opp_position_xy = env.scene["opponent"].data.root_pos_w[:, :2]
    
    dist = torch.norm(ego_position_xy - opp_position_xy, p=2, dim=1)


    opp_collision = dist < CONFIG['env_config']['OPP_COLLISION_RADIUS']

    
    return opp_collision.bool()

def is_reverse(env):
    reverse = reverse_waypoint_bool(env)
    return reverse

def reverse_waypoint_bool(env):
    # Safe access to buffer with fallback
    if not hasattr(env, '_progress_history_indices'):
        env._progress_history_indices = torch.zeros(env.num_envs, 
                                               dtype=torch.long,
                                               device=env.device)
        
    num_episodes = env.common_step_counter // env.max_episode_length
    if num_episodes < 10:
        return torch.zeros(env.num_envs, device=env.device) == 0
    
    # Get current positions and waypoints
    position_xy = env.scene["robot"].data.root_pos_w[:, :2]
    # waypoints = torch.tensor(env.scene.terrain.cfg.waypoints_list[0], 
    #                        device=env.device)[:, :2]

    waypoints = torch.tensor(env.scene.terrain.cfg.inner_list[0], 
                           device=env.device)[:, :2]
    
    # Find nearest waypoint
    current_idx, _ = find_frenet_coord_along_waypoints(waypoints, position_xy)
    

    if current_idx == 0:
        return torch.zeros(env.num_envs, device=env.device) == 1
    
    # Handle lap transitions by checking modulo distance
    num_waypoints = len(waypoints)

    progress = current_idx - env._progress_history_indices
    
    # Consider progress if moved forward (even across lap boundary)
    reverse_bool = progress < 0
    
    # Update stored indices
    env._progress_history_indices = current_idx.clone()
    
    return reverse_bool

@configclass
class F1TenthTimeTrialTerminationsCfg:
    # Time Out terms, i.e. conditions to terminate episode
    
    # Max episode time reached
    time_out = DoneTerm(
        func=mdp.time_out, 
        time_out=True)

    # Car rolls over
    rollover = DoneTerm(
        func=upright_bool,
        params={"thresh_deg": 90.},
    )

    # Car goes out of track
    if CONFIG['env_config']['NON_TRAVERSABLE_TERMINATION']:
        # non_traversable = DoneTerm(
        #     func=is_not_traversable
        # )

        wall_collision = DoneTerm(
            func=wall_collision
        )

        # opponent_collision = DoneTerm(
        #     func=opponent_collision
        # )

    # out_range = DoneTerm(
    #     func=out_of_map,
    # )


@configclass
class F1TenthTimeTrialRLEnvCfg(ManagerBasedRLEnvCfg):

    # These will be overwritten by the rss_cfgs
    seed: int = 42
    num_envs: int = 1
    env_spacing: int = 0

    ######################
    # MAP_NAME_LIST, to be manually changed here (it would be nice to pass it via hydra cfg)
    # THE INITIALIZATION TIME EXPONENTIALLY INCREASE WITH THE 
    MAP_NAME_LIST: List[str] = CONFIG['env_config']['MAP_NAME_LIST']

    ######################

    # Reset config
    events: F1TenthTimeTrialEventsCfg = F1TenthTimeTrialEventsCfg()

    # actions: Mushr4WDActionCfg = Mushr4WDActionCfg()
    actions: F1Tenth4WDActionCfg = F1Tenth4WDActionCfg()
    # actions: LB4WDActionCfg = LB4WDActionCfg()

    # MDP settings
    observations: F1TenthTimeTrialObsCfg = F1TenthTimeTrialObsCfg()
    rewards: F1TenthTimeTrialRewardsCfg = F1TenthTimeTrialRewardsCfg()
    terminations: F1TenthTimeTrialTerminationsCfg = F1TenthTimeTrialTerminationsCfg()
    curriculum: TimeTrialCurriculumCfg = TimeTrialCurriculumCfg()


    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        print('[INFO]: F1TenthTimeTrialRLEnvCfg class post init START')

        # viewer settings
        self.viewer.eye = [0., 0.0, 35.0] 
        self.viewer.lookat = [0.0, 0.0, -3.]
        self.sim.dt = CONFIG['env_config']['SIM_DT']
        self.decimation = CONFIG['env_config']['SIM_DECIMATION']
        # self.sim.dt = 0.025/2
        # self.decimation = 2
        # self.sim.render_interval = self.decimation
        self.sim.render_interval = self.decimation

        # Terminations config
        self.episode_length_s = CONFIG['env_config']['EPISODE_LENGTH_S']
        self.actions.throttle_steer.scale = (CONFIG['env_config']['MAX_SPEED_SCALING'], CONFIG['env_config']['MAX_STEERING_SCALING'])
        self.actions.throttle_steer.offset = (CONFIG['env_config']['SPEED_OFFSET'], CONFIG['env_config']['STEERING_OFFSET'])


        # Terrain variables
        MAP_NAME_LIST = self.MAP_NAME_LIST
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Folder where you save the usd files, name can be optimized, now it is possible it creates the same files with different names
        stage_path = os.path.join(WHEELEDLAB_ASSETS_DATA_DIR, 'maps', timestamp + '_test.usd')
        ORIGIN_LIST = CONFIG['env_config']['ORIGIN_LIST']

        
        # Folder where you have the maps (race stack format)
        maps_folder_path = '/home/tongo/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/timetrial/utils/maps'    

        ############################
        # IT IS IMPORTANT THE ORDER; 
        # first create the maps and initialize the lists, 
        # and secondly pass the lists to F1TenthTimeTrialTerrainImporterCfg

        # traversability_hashmap_list, 
        waypoints_list, outer_list, inner_list, d_lat_list, psi_rad_list, kappa_radpm_list, vx_mps_list, spacing_meters_list, map_size_pixels_list  = create_maps_from_waypoints(maps_folder_path, MAP_NAME_LIST, ORIGIN_LIST, stage_path, resolution=0.1)
        traversability_hashmap_list = []
        
        # Calculate derived values
        row_spacing_list = np.array(spacing_meters_list)[:, 0].tolist()
        col_spacing_list = np.array(spacing_meters_list)[:, 1].tolist()
        num_cols_list = np.array(map_size_pixels_list)[:, 0].tolist()
        num_rows_list = np.array(map_size_pixels_list)[:, 1].tolist()
        width_list = (np.array(num_rows_list) * np.array(row_spacing_list)).tolist()
        height_list = (np.array(num_cols_list) * np.array(col_spacing_list)).tolist()

        # Create terrain config
        self.terrain = F1TenthTimeTrialTerrainImporterCfg(
            prim_path="/World/envs/env_.*",
            env_spacing=self.env_spacing,
            usd_path=stage_path,
            map_name_list=MAP_NAME_LIST,
            traversability_hashmap_list=traversability_hashmap_list,
            waypoints_list=waypoints_list,
            outer_list=outer_list,
            inner_list=inner_list,
            d_lat_list=d_lat_list,
            psi_rad_list=psi_rad_list,
            kappa_radpm_list=kappa_radpm_list,
            vx_mps_list=vx_mps_list,
            spacing_meters_list=spacing_meters_list,
            map_size_pixels_list=map_size_pixels_list,
            row_spacing_list=row_spacing_list,
            col_spacing_list=col_spacing_list,
            num_cols_list=num_cols_list,
            num_rows_list=num_rows_list,
            width_list=width_list,
            height_list=height_list,
            # origin_list=ORIGIN_LIST,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="max",
                static_friction=STATIC_FRICTION,
                dynamic_friction=DYNAMIC_FRICTION,
                restitution=RESTITUTION
            ),
            debug_vis=True,
        )
        ############################

        self.scene = F1TenthTimeTrialSceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing, terrain = self.terrain
        )

        # Set the environment class
        self.env_class = F1TenthTimeTrialEnv
        print('[INFO]: F1TenthTimeTrialRLEnvCfg class post init END')


class F1TenthTimeTrialEnv(ManagerBasedEnv):
    def __init__(self, cfg: F1TenthTimeTrialRLEnvCfg, **kwargs):
        # Initialize parent class first
        super().__init__(cfg, **kwargs)
        
        # Initialize buffers needed in observations, rewards, etc

        # Save the initial waypoint idx (as soon as the car respawns)
        self._initial_waypoint_indices = torch.zeros(self.num_envs, 
                                                dtype=torch.long,
                                                device=self.device)

        self._prev_delta_s_opp_ego = torch.zeros(self.num_envs,
                                                   dtype=torch.float32,
                                                   device=self.device)
        
        # Save history of last #history_length waypoints idx
        self._progress_history_length = CONFIG['env_config']['PROGRESS_HISTORY_LENGTH']
        self._progress_history_checkpoint_idx = CONFIG['env_config']['PROGRESS_HISTORY_CHECKPOINT_IDX']

        self._progress_history_indices = torch.zeros(
            (self.num_envs, self._progress_history_length),  # Shape: (num_envs, history_length)
            dtype=torch.long,
            device=self.device
        )

        self._action_history_length = CONFIG['env_config']['ACTION_HISTORY_LENGTH']
        self._obs_history_length = CONFIG['env_config']['OBS_HISTORY_LENGTH']
        self._rew_history_length = CONFIG['env_config']['REW_HISTORY_LENGTH']

        self._action_history = torch.zeros(
            (self.num_envs, self._action_history_length, 2),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=self.device
        )

        self._base_lin_vel_x_history = torch.zeros(
            (self.num_envs, self._obs_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=self.device
        )
        self._base_lin_vel_y_history = torch.zeros(
            (self.num_envs, self._obs_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=self.device
        )
        self._base_ang_vel_z_history = torch.zeros(
            (self.num_envs, self._obs_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=self.device
        )

        self._traversability_history = torch.ones(
            (self.num_envs, self._rew_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.long,
            device=self.device
        )

        # Bool to determine if the car has just reset; it is set to True when a new pose is generated, and afterwards immediately set to false 
        self._reset_env_bool = torch.zeros(  # Tracks where to insert the next index
            self.num_envs,
            dtype=torch.bool,
            device=self.device
        )

        # Save how many waypoints idx the car has progressed
        self._current_progress = torch.zeros(self.num_envs,
                                           device=self.device)

        # Defines at which map the car is assigned to (see list order)
        self._map_levels = torch.zeros(self.num_envs, 
                                    dtype=torch.long,
                                    device=self.device)

        self._last_velocity_adjustment = torch.zeros(self.num_envs, 
                                    dtype=torch.float32,
                                    device=self.device)
        
        self._vel_y_calc = torch.zeros(self.num_envs, 
                                    dtype=torch.float32,
                                    device=self.device)

        self._target_steering_angle = torch.zeros(self.num_envs, 
                                    dtype=torch.float32,
                                    device=self.device)

        self._target_velocity = torch.zeros(self.num_envs, 
                                    dtype=torch.float32,
                                    device=self.device)

        self._target_velocity_history = torch.zeros(
            (self.num_envs, self._obs_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=self.device
        )
        
@configclass
class F1TenthTimeTrialRLRandomEnvCfg(F1TenthTimeTrialRLEnvCfg):
    events: F1TenthTimeTrialEventsRandomCfg = F1TenthTimeTrialEventsRandomCfg()

######################
###### PLAY ENV ######
######################

@configclass
class F1TenthTimeTrialPlayEnvCfg(F1TenthTimeTrialRLEnvCfg):
    """no terminations"""
  
    # on startup
    if CONFIG['env_config']['RESET_RANDOM']:
        events: F1TenthTimeTrialEventsCfg = F1TenthTimeTrialEventsRandomCfg(
            reset_root_state_random = EventTerm(
                func=reset_root_state_random,
                mode="reset",
            )
        )
    else:
        events: F1TenthTimeTrialEventsCfg = F1TenthTimeTrialEventsRandomCfg(
            reset_root_state_start_idx = EventTerm(
                func=reset_root_state_start_idx,
                mode="reset",
            )
        )

    rewards: F1TenthTimeTrialRewardsCfg = None
    terminations: F1TenthTimeTrialTerminationsCfg = None

    def __post_init__(self):
        super().__post_init__()