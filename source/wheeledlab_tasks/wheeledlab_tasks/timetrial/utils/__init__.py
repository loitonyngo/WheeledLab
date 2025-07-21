
import os
import numpy as np
from .traversability_utils import *
from .maps_utils import *

### hard coded for now

import time

def create_maps_from_waypoints(maps_folder_path, map_name_list, origin_list, stage_path, resolution):
    """
    Create a USD file and traversability hashmap from a PNG + YAML pair.
    
    Args:
        png_path: Path to the PNG image (black=obstacle, white=drivable).
        yaml_path: Path to the YAML file with metadata (resolution, origin).
        output_usd_path: Output USD file path (e.g., "/path/to/track.usd").
    """
    # Set stage first as it's needed throughout
    stage_name = stage_path
    stage = set_stage_usd(stage_name)
    
    # First pass: collect all needed data and calculate min_common_square_size
    map_data = []
    min_common_square_size = 0
    
    for i, map_name in enumerate(map_name_list):
        map_path = os.path.join(maps_folder_path, map_name)
        waypoints_path = os.path.join(map_path, 'global_waypoints.json')
        
        # Load and process waypoints data
        waypoints, trackbounds, d_lat, psi_rad, kappa_radpm, vx_mps = load_waypoints(waypoints_path)
        outer, inner = separate_and_align_bounds(waypoints, trackbounds)
        outer, inner = match_by_projection(waypoints, outer, inner)
        
        # Store intermediate results
        map_data.append({
            'waypoints': waypoints,
            'trackbounds': trackbounds,
            'd_lat': d_lat,
            'psi_rad': psi_rad,
            'kappa_radpm': kappa_radpm,
            'vx_mps': vx_mps,
            'outer': outer,
            'inner': inner
        })
        
        # Update min size
        min_map_size = get_min_map_size(outer, resolution)
        if min_map_size > min_common_square_size:
            min_common_square_size = min_map_size

    # Second pass: process stored data and generate outputs
    hashmap_list = []
    waypoints_list = []
    outer_list = []
    inner_list = []
    d_lat_list = []
    psi_rad_list = []
    kappa_radpm_list = []
    vx_mps_list = []
    spacing_meters_list = []
    map_size_pixels_list = []

    for i, (map_name, data) in enumerate(zip(map_name_list, map_data)):
        # Create drivable map
        hashmap, map_size_meters, map_size_pixels, (x_min, x_max), (y_min, y_max), res = create_square_drivable_map_v2(
            min_common_square_size, data['outer'], data['inner'], resolution
        )
        
        # Set USD elements
        set_hashmap_usd(
            map_name, hashmap, origin_list[i], map_size_pixels, 
            map_size_meters, stage, x_min, x_max, y_min, y_max, res
        )
        
        # Convert points to USD coordinates
        waypoints_usd = set_points_usd(
            data['waypoints'], map_name, 'waypoints', origin_list[i], 
            map_size_meters, stage, [(1.0, 0.0, 0.0)], x_min, y_min
        )
        outer_usd = set_points_usd(
            data['outer'], map_name, 'outer', origin_list[i], 
            map_size_meters, stage, [(0.0, 0.0, 1.0)], x_min, y_min
        )
        inner_usd = set_points_usd(
            data['inner'], map_name, 'inner', origin_list[i], 
            map_size_meters, stage, [(0.0, 0.0, 1.0)], x_min, y_min
        )
        
        # Store results (convert to lists at the end)
        hashmap_list.append(hashmap)
        waypoints_list.append([[p[0], p[1], p[2]] for p in waypoints_usd])
        outer_list.append([[p[0], p[1], p[2]] for p in outer_usd])
        inner_list.append([[p[0], p[1], p[2]] for p in inner_usd])
        d_lat_list.append(data['d_lat'].tolist())
        psi_rad_list.append(data['psi_rad'].tolist())
        kappa_radpm_list.append(data['kappa_radpm'].tolist())
        vx_mps_list.append(data['vx_mps'].tolist())
        spacing_meters_list.append([res, res])
        map_size_pixels_list.append(map_size_pixels)
        
        # Add to traversability hashmap
        TraversabilityHashmapUtil().add_traversability_hashmap(
            i, hashmap.tolist(), map_size_pixels, (res, res), origin_list[i]
        )

    # Save stage
    stage.GetRootLayer().Save()

    return (
        [h.tolist() for h in hashmap_list],  # Convert all numpy arrays to lists at the end
        waypoints_list,
        outer_list,
        inner_list,
        d_lat_list,
        psi_rad_list,
        kappa_radpm_list,
        vx_mps_list,
        spacing_meters_list,
        map_size_pixels_list
    )


def generate_random_poses_from_list(env_ids, num_poses, map_levels, env_origins, map_origin_list, row_spacing_list, col_spacing_list, traversability_hashmap_list, waypoints_usd_list, outer_usd_list, inner_usd_list, margin=0.1):
    """
    Generate random poses with vectorized operations, supporting multiple hashmaps based on map_level.
    Only generates poses for environments specified in env_ids.
    """
    # Convert inputs to numpy/torch as needed
    env_ids_np = env_ids.cpu().numpy() if torch.is_tensor(env_ids) else np.array(env_ids)
    map_levels_np = map_levels.cpu().numpy() if torch.is_tensor(map_levels) else np.array(map_levels)
    
    # Get map levels only for the requested environments
    requested_map_levels = map_levels_np[env_ids_np]
    
    # Initialize output containers
    all_xs_shifted = np.zeros(len(env_ids))
    all_ys_shifted = np.zeros(len(env_ids))
    current_wps_idx = np.zeros(len(env_ids))
    all_angles = np.zeros(len(env_ids))
    
    # Process each unique map_level separately among the requested environments
    unique_map_levels = np.unique(requested_map_levels)
    
    for map_level in unique_map_levels:
        # Get indices (within env_ids) of environments with this map_level
        env_mask = (requested_map_levels == map_level)
        current_env_ids = env_ids_np[env_mask]
        
        # Skip if no environments use this map_level (shouldn't happen due to unique)
        if len(current_env_ids) == 0:
            continue
            
        # Get the hashmap and parameters for this map_level
        traversability_array = np.array(traversability_hashmap_list[map_level])
        H, W = traversability_array.shape
        
        # Get all valid positions in one operation
        valid_y, valid_x = traversability_array.nonzero()
        
        # Sample indices without replacement (one per environment)
        idxs = np.random.choice(len(valid_x), size=len(current_env_ids), replace=len(current_env_ids) > len(valid_x))
        
        # Convert all positions at once (vectorized)
        xs = (valid_x[idxs] - W // 2) * row_spacing_list[map_level]
        ys = (valid_y[idxs] - H // 2) * col_spacing_list[map_level]
        
        # Pre-compute waypoints tensor shifted by the origins
        waypoints_xy = torch.tensor(waypoints_usd_list[map_level])[:, :2].to(torch.float32) 
        inner_xy = torch.tensor(inner_usd_list[map_level])[:, :2].to(torch.float32)

        # Vectorized angle computation
        positions = torch.stack([torch.tensor(xs), torch.tensor(ys)], dim=1) + torch.tensor(map_origin_list[map_level][:2])
        current_indices, _ = find_nearest_waypoint(inner_xy, positions)
        current_wps_idx[env_mask] = current_indices

        # Compute angles for all positions at once
        lookahead = 5
        next_indices = (current_indices + lookahead) % len(inner_xy)
        
        current_wps = inner_xy[current_indices] 
        next_wps = inner_xy[next_indices]
        
        deltas = next_wps - current_wps
        angles = torch.rad2deg(torch.atan2(deltas[:, 1], deltas[:, 0])) + np.random.uniform(-15, 15, size=len(current_env_ids))
        
        # Shift the coordinates according to env_origins for the reset
        xs_shifted = xs + env_origins[current_env_ids, 0].cpu().numpy() + np.array(map_origin_list)[map_level, 0]
        ys_shifted = ys + env_origins[current_env_ids, 1].cpu().numpy() + np.array(map_origin_list)[map_level, 1]
        
        # Store results in the output arrays at the correct positions
        all_xs_shifted[env_mask] = xs_shifted
        all_ys_shifted[env_mask] = ys_shifted
        all_angles[env_mask] = angles.numpy() if torch.is_tensor(angles) else angles
    
    # Combine results while maintaining original order
    poses = list(zip(all_xs_shifted.tolist(), all_ys_shifted.tolist(), all_angles.tolist()))
    
    return poses, current_wps_idx

def generate_start_idx_poses_from_list(env_ids, num_poses, map_levels, env_origins, map_origin_list, row_spacing_list, col_spacing_list, traversability_hashmap_list, waypoints_usd_list, outer_usd_list, inner_usd_list, margin=0.1):
    """
    Generate random poses with vectorized operations, supporting multiple hashmaps based on map_level.
    Only generates poses for environments specified in env_ids.
    """
    # Convert inputs to numpy/torch as needed
    env_ids_np = env_ids.cpu().numpy() if torch.is_tensor(env_ids) else np.array(env_ids)
    map_levels_np = map_levels.cpu().numpy() if torch.is_tensor(map_levels) else np.array(map_levels)
    
    # Get map levels only for the requested environments
    requested_map_levels = map_levels_np[env_ids_np]
    
    # Initialize output containers
    all_xs_shifted = np.zeros(len(env_ids))
    all_ys_shifted = np.zeros(len(env_ids))
    current_wps_idx = np.zeros(len(env_ids))
    all_angles = np.zeros(len(env_ids))
    
    # Process each unique map_level separately among the requested environments
    unique_map_levels = np.unique(requested_map_levels)
    
    for map_level in unique_map_levels:
        # Get indices (within env_ids) of environments with this map_level
        env_mask = (requested_map_levels == map_level)
        current_env_ids = env_ids_np[env_mask]
        
        # Skip if no environments use this map_level (shouldn't happen due to unique)
        if len(current_env_ids) == 0:
            continue
            
        # # Get the hashmap and parameters for this map_level
        # traversability_array = np.array(traversability_hashmap_list[map_level])
        # H, W = traversability_array.shape
        
        # # Get all valid positions in one operation
        # valid_y, valid_x = traversability_array.nonzero()
        
        # # Sample indices without replacement (one per environment)
        # idxs = np.random.choice(len(valid_x), size=len(current_env_ids), replace=len(current_env_ids) > len(valid_x))
        
        # # Convert all positions at once (vectorized)
        # xs = (valid_x[idxs] - W // 2) * row_spacing_list[map_level]
        # ys = (valid_y[idxs] - H // 2) * col_spacing_list[map_level]
        
        # Pre-compute waypoints tensor shifted by the origins
        waypoints_xy = torch.tensor(waypoints_usd_list[map_level])[:, :2].to(torch.float32) 
        xs = waypoints_xy[0, 0].cpu().numpy()
        ys = waypoints_xy[0, 1].cpu().numpy()
        inner_xy = torch.tensor(inner_usd_list[map_level])[:, :2].to(torch.float32)

        # Vectorized angle computation
        positions = torch.stack([torch.tensor(xs), torch.tensor(ys)], dim=-1) + torch.tensor(map_origin_list[map_level][:2])
        current_indices = 0
        current_wps_idx[env_mask] = current_indices

        # Compute angles for all positions at once
        lookahead = 5
        next_indices = (current_indices + lookahead) % len(inner_xy)
        
        current_wps = inner_xy[current_indices] 
        next_wps = inner_xy[next_indices]
        
        deltas = next_wps - current_wps
        angles = torch.rad2deg(torch.atan2(deltas[1], deltas[0])) 
        
        # Shift the coordinates according to env_origins for the reset
        xs_shifted = xs + env_origins[current_env_ids, 0].cpu().numpy() + np.array(map_origin_list)[map_level, 0]
        ys_shifted = ys + env_origins[current_env_ids, 1].cpu().numpy() + np.array(map_origin_list)[map_level, 1]
        
        # Store results in the output arrays at the correct positions
        all_xs_shifted[env_mask] = xs_shifted
        all_ys_shifted[env_mask] = ys_shifted
        all_angles[env_mask] = angles.numpy() if torch.is_tensor(angles) else angles
    
    # Combine results while maintaining original order
    poses = list(zip(all_xs_shifted.tolist(), all_ys_shifted.tolist(), all_angles.tolist()))
    
    return poses, current_wps_idx

def generate_random_poses(env_origins, env_ids, num_poses, row_spacing, col_spacing, traversability_hashmap, waypoints_usd, outer_usd, inner_usd, margin=0.1):
    """
    Generate random poses with vectorized operations.
    """
    # Convert to numpy array once
    traversability_array = np.array(traversability_hashmap)
    H, W = traversability_array.shape
    
    # Get all valid positions in one operation
    valid_y, valid_x = traversability_array.nonzero()
    
    # Sample indices without replacement
    idxs = np.random.choice(len(valid_x), size=num_poses, replace=False)
    
    # Convert all positions at once (vectorized)
    xs = (valid_x[idxs] - W // 2) * row_spacing
    ys = (valid_y[idxs] - H // 2) * col_spacing 
    
    # Pre-compute waypoints tensor shifted by the origins
    waypoints_xy = torch.tensor(waypoints_usd)[:, :2].to(torch.float32) 
    inner_xy = torch.tensor(inner_usd)[:, :2].to(torch.float32)

    # Vectorized angle computation
    positions = torch.stack([torch.tensor(xs), torch.tensor(ys)], dim=1)
    current_indices, _ = find_nearest_waypoint(inner_xy, positions)
    
    # Compute angles for all positions at once
    lookahead = 5
    next_indices = (current_indices + lookahead) % len(inner_xy)
    
    # we take the inner bound instead of the closest waypoint to avoid case when the centerline from the other side of the wall is closer
    current_wps = inner_xy[current_indices] 
    next_wps = inner_xy[next_indices]
    
    deltas = next_wps - current_wps
    angles = torch.rad2deg(torch.atan2(deltas[:, 1], deltas[:, 0])) + np.random.uniform(-15,15)
    
    # Now shift the coordinates according to env_origins for the reset
    xs_shifted = (valid_x[idxs] - W // 2) * row_spacing + env_origins[env_ids, 0].cpu().numpy()
    ys_shifted = (valid_y[idxs] - H // 2) * col_spacing + env_origins[env_ids, 1].cpu().numpy()
    poses = list(zip(xs_shifted.tolist(), ys_shifted.tolist(), angles.tolist()))

    # Combine results
    # poses = list(zip(xs.tolist(), ys.tolist(), angles.tolist()))
    
    return poses

def generate_random_poses_from_waypoints(env_ids, num_poses, map_levels, env_origins, map_origin_list, waypoints_usd_list, inner_usd_list, margin=0.1):
    """
    Generate random poses by selecting from waypoints, supporting multiple maps based on map_level.
    Only generates poses for environments specified in env_ids.
    """
    # Convert inputs to numpy/torch as needed
    env_ids_np = env_ids.cpu().numpy() if torch.is_tensor(env_ids) else np.array(env_ids)
    map_levels_np = map_levels.cpu().numpy() if torch.is_tensor(map_levels) else np.array(map_levels)
    
    # Get map levels only for the requested environments
    requested_map_levels = map_levels_np[env_ids_np]
    
    # Initialize output containers
    all_xs_shifted = np.zeros(len(env_ids))
    all_ys_shifted = np.zeros(len(env_ids))
    current_wps_idx = np.zeros(len(env_ids))
    all_angles = np.zeros(len(env_ids))
    
    # Process each unique map_level separately among the requested environments
    unique_map_levels = np.unique(requested_map_levels)
    
    for map_level in unique_map_levels:
        # Get indices (within env_ids) of environments with this map_level
        env_mask = (requested_map_levels == map_level)
        current_env_ids = env_ids_np[env_mask]
        
        # Skip if no environments use this map_level (shouldn't happen due to unique)
        if len(current_env_ids) == 0:
            continue
            
        # Get the waypoints for this map_level
        waypoints_xy = torch.tensor(waypoints_usd_list[map_level])[:, :2].to(torch.float32)
        inner_xy = torch.tensor(inner_usd_list[map_level])[:, :2].to(torch.float32)
        num_waypoints = len(waypoints_xy)
        
        # Randomly select waypoints for each environment
        selected_indices = np.random.choice(num_waypoints, size=len(current_env_ids), replace=True)
        
        # Get the positions of the selected waypoints
        selected_waypoints = waypoints_xy[selected_indices]
        xs = selected_waypoints[:, 0].numpy()
        ys = selected_waypoints[:, 1].numpy()
        
        # Compute angles (looking at next waypoint)
        lookahead = 5
        next_indices = (selected_indices + lookahead) % num_waypoints
        next_waypoints = waypoints_xy[next_indices]
        
        deltas = next_waypoints - selected_waypoints
        angles = torch.rad2deg(torch.atan2(deltas[:, 1], deltas[:, 0])) + np.random.uniform(-15, 15, size=len(current_env_ids))
        
        # Shift the coordinates according to env_origins for the reset
        xs_shifted = xs + env_origins[current_env_ids, 0].cpu().numpy() + np.array(map_origin_list)[map_level, 0] + np.random.rand()*0.3 - np.random.rand()*0.3
        ys_shifted = ys + env_origins[current_env_ids, 1].cpu().numpy() + np.array(map_origin_list)[map_level, 1] + np.random.rand()*0.3 - np.random.rand()*0.3
        
        # Store results in the output arrays at the correct positions
        all_xs_shifted[env_mask] = xs_shifted
        all_ys_shifted[env_mask] = ys_shifted
        all_angles[env_mask] = angles.numpy() if torch.is_tensor(angles) else angles
        current_wps_idx[env_mask] = selected_indices
    
    # Combine results while maintaining original order
    poses = list(zip(all_xs_shifted.tolist(), all_ys_shifted.tolist(), all_angles.tolist()))
    
    return poses, current_wps_idx

def find_nearest_waypoint(waypoints: torch.Tensor,  # Shape: [M, 2] - M waypoints
                         positions: torch.Tensor, # [N,2] - N environements
                         ) -> tuple[int, torch.Tensor]:
    """
    Finds closest waypoints for all cars, handling circular track wrapping.
    If lookahead is None, searches all waypoints (accurate but slower).
    With lookahead, only checks next K waypoints from current closest (faster).
    """
    M = waypoints.shape[0]
    N = positions.shape[0]
    
    # First find rough closest without wrapping [N]
    diffs = positions.unsqueeze(1) - waypoints.unsqueeze(0)  # [N,M,2]
    dists = torch.norm(diffs, p=2, dim=2)  # [N,M]
    closest_idx = torch.argmin(dists, dim=1)  # [N]
    
    return closest_idx, dists[torch.arange(N), closest_idx]

if __name__ == "__main__":
    create_maps_from_png('test.usd', 100, 100, 0.3, 0.3, 0.3)

