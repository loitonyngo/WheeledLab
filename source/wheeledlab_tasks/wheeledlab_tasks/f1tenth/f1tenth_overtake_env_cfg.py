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

from .utils import create_maps_from_waypoints, generate_random_poses_from_waypoints_with_opponent, find_frenet_coord_along_waypoints 
from . import mdp_sensors
from .mdp import reset_root_state_random, reset_root_state_random_with_opponent, reset_root_state_start_idx

from .mdp.observations import *
from .mdp.rewards import *
from .mdp.terminations import *
from .mdp.events import move_opponent_s_based


import omni.usd

import yaml  # Add this import at the top of your file
from pathlib import Path
from typing import List  # For type hints

with open("/home/tongo/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/f1tenth/config/f1tenth_config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)


##############################
###### OBSERVATION #######
##############################
##########################
# Variables for observation space. Better way to implement it?

N_HORIZON = CONFIG['env_config']['N_HORIZON'] 
DELTA_S_IDX = CONFIG['env_config']['DELTA_S_IDX'] 

T_HORIZON = CONFIG['env_config']['T_HORIZON']
##########################

@configclass
class F1TenthOvertakeObsCfg:
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
                    'std_noise': 0.5}            
            )

        base_lin_acc_x_history = ObsTerm(
            func=base_lin_acc_x_history, 
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

        target_steering_angle_history = ObsTerm(
            func=target_steering_angle_history, 
            params={'mean_noise': 0,
                    'std_noise': 0}         
            )
                
        action_history = ObsTerm(
            func=action_history,
        )

        track_info_horizon = ObsTerm(
            func=track_info_horizon,
            params={'delta_s_idx': DELTA_S_IDX,
                    'n_horizon': N_HORIZON,
                    't_horizon': T_HORIZON,
                    'position_std_noise': 0.08}
        )
        
        opponent_relative_info_history = ObsTerm(
            func=opponent_relative_info_history,
            params={'position_std_noise': 0.08,
                    'velocity_std_noise': 0.45}
        )
        
        gaps_info = ObsTerm(
            func=gaps_info_history
        ) 

        # opponent_frenet_coordinates_history = ObsTerm(
        #     func=opponent_frenet_coordinates_history
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
class F1TenthOvertakeTerrainImporterCfg(TerrainImporterCfg):
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

    opp_traj_center_list= None
    opp_traj_iqp_list= None
    opp_traj_sp_list= None
    
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
    
    def generate_random_poses_from_waypoints_with_opponent(self, env: ManagerBasedEnv, env_ids, num_poses, max_radius_offset=0.3):
        # generate random initial poses with margin
        env_origins = env.scene.env_origins
        map_levels = env._map_levels
        
        # Get poses and velocities from the helper function
        init_poses, init_current_wps_idx, ego_velocities, opp_init_poses, opp_init_current_wps_idx, opp_velocities = generate_random_poses_from_waypoints_with_opponent(
            env_ids, num_poses, map_levels, env_origins, 
            self.waypoints_list, self.inner_list, self.vx_mps_list, 
            max_radius_offset=0.3
        )
        
        max_radius_offset = 0.5
        valid_init_poses = [
            InitialPoseCfg(
                pos=(x + random.uniform(-1,1)*max_radius_offset, 
                y + random.uniform(-1,1)*max_radius_offset, 
                0.02),
                rot_euler_xyz_deg=(0., 0., angle),
                lin_vel=(random.uniform(0,2), 0, 0.0),  # Add linear velocity
                ang_vel=(0.0, 0.0, 0.0)  # Angular velocity is zero
            ) for (x, y, angle), (vx, vy) in zip(init_poses, ego_velocities)
        ]
        
        opp_valid_init_poses = [
            InitialPoseCfg(
                pos=(x + random.uniform(-1,1)*max_radius_offset, 
                    y + random.uniform(-1,1)*max_radius_offset, 
                    0.02),
                rot_euler_xyz_deg=(0., 0., angle),
                lin_vel=(0, 0, 0.0),  # Add linear velocity for opponent
                ang_vel=(0.0, 0.0, 0.0)  # Angular velocity is zero
            ) for (x, y, angle), (vx, vy) in zip(opp_init_poses, opp_velocities)
        ]
                
        return valid_init_poses, init_current_wps_idx, opp_valid_init_poses, opp_init_current_wps_idx
    # def generate_start_idx_poses(self, env : ManagerBasedEnv, env_ids, num_poses):
        
    #     # generate random initial poses with margin
    #     env_origins = env.scene.env_origins
    #     map_levels = env._map_levels
    #     # add which map level

    #     init_poses, init_current_wps_idx = generate_start_idx_poses_from_list(env_ids, num_poses, map_levels, env_origins, self.row_spacing_list, self.col_spacing_list, self.traversability_hashmap_list, self.waypoints_list, self.outer_list, self.inner_list, margin=0.1)
    #     valid_init_poses = [
    #         InitialPoseCfg(
    #             pos=(x, y, 0.02),
    #             # rot_euler_xyz_deg=(0., 0., angle)
    #             rot_euler_xyz_deg=(0., 0., 0)
    #         ) for x, y, angle in init_poses
    #     ]
    #     return valid_init_poses, init_current_wps_idx

@configclass
class F1TenthOvertakeSceneCfg(InteractiveSceneCfg):
    """Configuration for a Mushr car Scene with racetrack terrain and Sensors."""

    terrain = None
    MAP_NAME_LIST = None
    ground = AssetBaseCfg(
        prim_path="/World/base",
        spawn = sim_utils.GroundPlaneCfg(size=(1500 , 1500),
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

    
       # Add cuboid configuration
    opponent = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Opponent",
        spawn=sim_utils.CylinderCfg(
            radius=CONFIG['env_config']['OPP_SIZE_RADIUS'],
            height=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled= False, 
                rigid_body_enabled=True,
                solver_position_iteration_count=1,
                solver_velocity_iteration_count=1,
                max_angular_velocity=100.0,
                max_linear_velocity=100.0,
                max_depenetration_velocity=1,
                disable_gravity=True,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),  # 10cm above ground
    )

    ground.init_state.pos = (0.0, 0.0, -1e-4)

    def __post_init__(self):
        """Post intialization."""
        super().__post_init__()

        self.robot.init_state = self.robot.init_state.replace(
            pos=(0.0, 0.0, 0.0)
        )

        self.opponent.init_state = self.opponent.init_state.replace(
            pos=(0.0, 0.0, 0.0)
        )

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
class F1TenthOvertakeEventsCfg:

    # on startup
    if CONFIG['env_config']['RESET_RANDOM']:
        # reset_root_state_random = EventTerm(
        #     func=reset_root_state_random,
        #     mode="reset",
        # )
            
        reset_root_state_random = EventTerm(
            func=reset_root_state_random_with_opponent,
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

    move_opponent_s_based = EventTerm(
        func=move_opponent_s_based,
        mode="interval",
        interval_range_s=(CONFIG['env_config']['OPP_MOVE_DT'] , CONFIG['env_config']['OPP_MOVE_DT'] )
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
class F1TenthOvertakeEventsRandomCfg(F1TenthOvertakeEventsCfg):
    
    if CONFIG['env_config']['RANDOMIZE_FRICTION']:
        change_wheel_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "static_friction_range": (CONFIG['env_config']['STATIC_FRICTION']-CONFIG['env_config']['STD_FRICTION'], CONFIG['env_config']['STATIC_FRICTION']+CONFIG['env_config']['STD_FRICTION']),
                "dynamic_friction_range": (CONFIG['env_config']['DYNAMIC_FRICTION']-CONFIG['env_config']['STD_FRICTION'], CONFIG['env_config']['DYNAMIC_FRICTION']+CONFIG['env_config']['STD_FRICTION']),
                "restitution_range": (0.0, 0.0),
                "num_buckets": CONFIG['env_config']['NUM_BUCKETS_FRICTION'],
                "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_.*link"),
                "make_consistent": True,
            },
        )

    # standard mass is 3.17 kg
    if CONFIG['env_config']['RANDOMIZE_BODY_MASS']:
        add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
                "mass_distribution_params": (0.99, 1.01),
                "operation": "scale",
            },
        )
        
    if CONFIG['env_config']['RANDOMIZE_ACTUATOR_STEERING_GAIN']:
        # Randomize steering actuator gains with scaling
        randomize_steering_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",  # apply once at environment reset
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["rotator_(left|right)"]),
                "stiffness_distribution_params": (0.9, 1.1),  # scale between 80% and 120%
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        
    if CONFIG['env_config']['RANDOMIZE_ACTUATOR_THROTTLE_GAIN']:
        # Randomize throttle actuator damping with scaling
        randomize_throttle_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*wheel_(back|front)_.*"]),
                "damping_distribution_params": (0.9, 1.1),  # 80%–120% of default damping
                "operation": "scale",
                "distribution": "uniform",
            },
        )


    kill_lidar = EventTerm(
        func=disable_all_lidars,
        mode="startup",
        params={}          
    )

######################
###### REWARDS #######
######################

####### F1TenthOvertake Environment #######
@configclass
class F1TenthOvertakeRewardsCfg:
    # """Reward terms for the MDP."""
    # Set "weight" to 0 to deactivate a reward term

    # Standard reward for progressing along centerline, weight=1
    progress_rew = RewTerm(
        func=progress_rew,
        weight=1.0,
    )
    average_vel = RewTerm(
        func=average_vel,
        weight=0.08,
    )
        
    wall_collision_penalty = RewTerm(
        func=wall_collision_penalty,
        weight=0.5,
    )

    soft_wall_collision_penalty = RewTerm(
        func=soft_wall_collision_penalty,
        weight=0.05,
    )
    
    opponent_collision_penalty = RewTerm(
        func=opponent_collision_penalty,
        weight=0.5,
    )

    opponent_overtake_completed_reward = RewTerm(
        func=opponent_overtake_completed_reward,
        weight=500,
    )
    
    opponent_overtake_distance_reward = RewTerm(
        func=opponent_overtake_distance_reward,
        weight=0.0,
    )

    # opponent_overtake_delta_distance_reward = RewTerm(
    #     func=opponent_overtake_delta_distance_reward,
    #     weight=0.5,
    # )

    # opponent_mean_delta_speed = RewTerm(
    #     func=opponent_mean_delta_speed,
    #     weight=0.01,
    # )

    # opponent_overtake_positioning_reward = RewTerm(
    #     func=opponent_overtake_positioning_reward,
    #     weight=0.1,
    # )

    # var_throttle_rate_penalty =  RewTerm(
    #     func=var_throttle_rate_penalty,
    #     weight=0.00,
    # )
    
    var_throttle_penalty =  RewTerm(
        func=var_throttle_penalty,
        weight=0.01,
    )
    
    var_steering_penalty =  RewTerm(
        func=var_steering_penalty,
        weight=0.001,
    )

    effort_throttle_penalty =  RewTerm(
        func=effort_throttle_penalty,
        weight=0.0,
    )
    
    effort_steering_penalty =  RewTerm(
        func=effort_steering_penalty,
        weight=0.001,
    )

    effort_steering_penalty =  RewTerm(
        func=effort_target_steering_angle_penalty,
        weight=0.1,
    )
    # low_speed_penalty =  RewTerm(
    #     func=low_speed_penalty,
    #     weight=0.0,
    # )


########################
###### CURRICULUM ######
########################

MAX_NUM_INC = 0
@configclass
class OvertakeCurriculumCfg:

    progress_rew = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "progress_rew",
            "weight_increase": -0.20,
            "min_weight": 0.25,
            "first_episode_increase": 25,
            "episodes_per_increase": 25,
            "max_num_increases": MAX_NUM_INC,
        }
    )
    
    wall_collision_penalty = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "wall_collision_penalty",
            "weight_increase": 0.1,
            "max_weight": 5,
            "first_episode_increase":20,
            "episodes_per_increase": 20,
            "max_num_increases": MAX_NUM_INC,
        }
    )

    opponent_collision_penalty = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "opponent_collision_penalty",
            "weight_increase": 0.1,
            "max_weight": 5,
            "first_episode_increase": 20,
            "episodes_per_increase": 20,
            "max_num_increases": MAX_NUM_INC,
        }
    )

    opponent_overtake_delta_distance_reward = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "opponent_overtake_delta_distance_reward",
            "weight_increase": 2,
            "max_weight": 20,
            "first_episode_increase": 50,
            "episodes_per_increase": 50,
            "max_num_increases": MAX_NUM_INC,
        }
    )

    opponent_overtake_distance_reward = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "opponent_overtake_distance_reward",
            "weight_increase": 0.05,
            "max_weight": 1,
            "first_episode_increase": 50,
            "episodes_per_increase": 50,
            "max_num_increases": MAX_NUM_INC,
        }
    )
    
    # opponent_overtake_positioning_reward = CurrTerm(
    #     func=increase_reward_weight_over_time,
    #     params={
    #         "reward_term_name": "opponent_overtake_positioning_reward",
    #         "weight_increase": 0.5,
    #         "first_episode_increase": 50,
    #         "episodes_per_increase": 50,
    #         "max_num_increases": 3,
    #     }
    # )
    
    var_throttle_penalty = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "var_throttle_penalty",
            "weight_increase": 0.5,
            "max_weight": 1.5,
            "first_episode_increase": 50,
            "episodes_per_increase": 50,
            "max_num_increases": MAX_NUM_INC,
        }
    )

    var_steering_penalty = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "var_steering_penalty",
            "weight_increase": 0.5,
            "max_weight": 0.5,
            "first_episode_increase": 25,
            "episodes_per_increase": 25,
            "max_num_increases": MAX_NUM_INC,
        }
    )


##########################
###### TERMINATION #######
##########################

@configclass
class F1TenthOvertakeTerminationsCfg:
    # Time Out terms, i.e. conditions to terminate episode
    
    # Max episode time reached
    time_out = DoneTerm(
        func=time_out, 
        time_out=True
    )

    opponent_overtaken = DoneTerm(
            func=opponent_overtaken,
    )
    
    far_from_opponent = DoneTerm(
        func=far_from_opponent,
    )
    
    opponent_collision = DoneTerm(
            func=opponent_collision
    )

    # Car goes out of track
    if CONFIG['env_config']['NON_TRAVERSABLE_TERMINATION']:

        wall_collision = DoneTerm(
            func=wall_collision
        )


@configclass
class F1TenthOvertakeRLEnvCfg(ManagerBasedRLEnvCfg):

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
    events: F1TenthOvertakeEventsCfg = F1TenthOvertakeEventsCfg()

    actions: F1Tenth4WDActionCfg = F1Tenth4WDActionCfg()

    # MDP settings
    observations: F1TenthOvertakeObsCfg = F1TenthOvertakeObsCfg()
    rewards: F1TenthOvertakeRewardsCfg = F1TenthOvertakeRewardsCfg()
    terminations: F1TenthOvertakeTerminationsCfg = F1TenthOvertakeTerminationsCfg()
    curriculum: OvertakeCurriculumCfg = OvertakeCurriculumCfg()


    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        print('[INFO]: F1TenthOvertakeRLEnvCfg class post init START')

        # viewer settings
        self.viewer.eye = [0., 0.0, 40.0] 
        self.viewer.lookat = [0.0, 0.0, -3.]
        self.sim.dt = CONFIG['env_config']['SIM_DT']
        self.decimation = CONFIG['env_config']['SIM_DECIMATION']
        # self.sim.dt = 0.025/2
        # self.decimation = 2
        # self.sim.render_interval = self.decimation
        self.sim.render_interval = self.decimation

        # Terminations config
        self.episode_length_s = CONFIG['env_config']['EPISODE_LENGTH_S_OVERTAKE']
        self.actions.throttle_steer.scale = (CONFIG['env_config']['MAX_SPEED_SCALING'], CONFIG['env_config']['MAX_STEERING_SCALING'])
        self.actions.throttle_steer.offset = (CONFIG['env_config']['SPEED_OFFSET'], CONFIG['env_config']['STEERING_OFFSET'])


        # Terrain variables
        MAP_NAME_LIST = self.MAP_NAME_LIST
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Folder where you save the usd files, name can be optimized, now it is possible it creates the same files with different names
        stage_path = os.path.join(WHEELEDLAB_ASSETS_DATA_DIR, 'maps', timestamp + '_test.usd')
        ORIGIN_LIST = CONFIG['env_config']['ORIGIN_LIST']

        
        # Folder where you have the maps (race stack format)
        maps_folder_path = '/home/tongo/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/f1tenth/utils/maps'    

        ############################
        # IT IS IMPORTANT THE ORDER; 
        # first create the maps and initialize the lists, 
        # and secondly pass the lists to F1TenthOvertakeTerrainImporterCfg

        # traversability_hashmap_list, 
        waypoints_list, outer_list, inner_list, d_lat_list, psi_rad_list, kappa_radpm_list, vx_mps_list, opp_traj_center_list, opp_traj_iqp_list, opp_traj_sp_list, spacing_meters_list, map_size_pixels_list  = create_maps_from_waypoints(maps_folder_path, MAP_NAME_LIST, ORIGIN_LIST, stage_path, resolution=0.1)
        traversability_hashmap_list = []
        
        # Calculate derived values
        row_spacing_list = np.array(spacing_meters_list)[:, 0].tolist()
        col_spacing_list = np.array(spacing_meters_list)[:, 1].tolist()
        num_cols_list = np.array(map_size_pixels_list)[:, 0].tolist()
        num_rows_list = np.array(map_size_pixels_list)[:, 1].tolist()
        width_list = (np.array(num_rows_list) * np.array(row_spacing_list)).tolist()
        height_list = (np.array(num_cols_list) * np.array(col_spacing_list)).tolist()

        # Create terrain config
        self.terrain = F1TenthOvertakeTerrainImporterCfg(
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
            opp_traj_center_list=opp_traj_center_list,
            opp_traj_iqp_list=opp_traj_iqp_list,
            opp_traj_sp_list=opp_traj_sp_list,
            debug_vis=True,
        )
        ############################

        self.scene = F1TenthOvertakeSceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing, terrain = self.terrain
        )

        # Set the environment class
        self.env_class = F1TenthOvertakeEnv
        print('[INFO]: F1TenthOvertakeRLEnvCfg class post init END')


class F1TenthOvertakeEnv(ManagerBasedEnv):
    def __init__(self, cfg: F1TenthOvertakeRLEnvCfg, **kwargs):
        # Initialize parent class first
        super().__init__(cfg, **kwargs)
        
        # Initialize buffers needed in observations, rewards, etc
        ########### CACHE MAP WAYPOINTS IN ENV AS TENSOR ##########

        # Save the initial waypoint idx (as soon as the car respawns)
        self._initial_waypoint_indices = torch.zeros(self.num_envs, 
                                                dtype=torch.long,
                                                device=self.device)

        self._prev_delta_s_opp_ego = torch.zeros(self.num_envs,
                                                   dtype=torch.float32,
                                                   device=self.device)
        
        # Save history of last #history_length waypoints idx
        self._progress_history_checkpoint_idx = CONFIG['env_config']['PROGRESS_HISTORY_CHECK_IDX']

        self._progress_history_indices = torch.zeros(
            (self.num_envs, self._reward_history_length),  # Shape: (num_envs, history_length)
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
        
        self._wall_collision_history = torch.zeros(
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
        
        self._target_steering_angle_history = torch.zeros(
            (self.num_envs, self._obs_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=self.device
        )
        self._opponent_type = torch.zeros(
            self.num_envs,  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.long,
            device=self.device
        )
        
        self._opponent_vel_scaling = torch.ones(
            self.num_envs,  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float16,
            device=self.device
        )

        self._opponent_trajectory_alpha = torch.ones(
            self.num_envs,  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float16,
            device=self.device
        )

        self._opponent_overtaken_bool = torch.zeros(
            self.num_envs,  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.bool,
            device=self.device
        )

        self._opponent_overtaken_history = torch.zeros(
            (self.num_envs, self._rew_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.long,
            device=self.device
        )
        
        self._opponent_speed = torch.zeros(
            self.num_envs,  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=self.device
        )

        self._opponent_d_dot = torch.zeros(
            self.num_envs,  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=self.device
        )
        
        self._opponent_heading = torch.zeros(
            self.num_envs,  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.float32,
            device=self.device
        )
        
        self._opponent_always_ahead = torch.zeros(
            self.num_envs,  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.bool,
            device=self.device
        )

        self._opponent_collision_history = torch.zeros(
            (self.num_envs, self._rew_history_length),  # Shape: (num_envs, history_length, n_actions)
            dtype=torch.long,
            device=self.device
        )

        self._opponent_s_history = torch.zeros(
            (self.num_envs, self._obs_history_length), 
            dtype=torch.float32,
            device=self.device
        )

        self._opponent_d_history = torch.zeros(
            (self.num_envs, self._obs_history_length), 
            dtype=torch.float32,
            device=self.device
        )
                
        self._s_idx_diff_history = torch.zeros(
            (self.num_envs, self._obs_history_length), 
            dtype=torch.float32,
            device=self.device
        )

        self._d_diff_history = torch.zeros(
            (self.num_envs, self._obs_history_length), 
            dtype=torch.float32,
            device=self.device
        )

        self._vx_diff_history = torch.zeros(
            (self.num_envs, self._obs_history_length), 
            dtype=torch.float32,
            device=self.device
        )

        self._heading_diff_history = torch.zeros(
            (self.num_envs, self._obs_history_length), 
            dtype=torch.float32,
            device=self.device
        )

        self._cross_pos_history = torch.zeros(
            (self.num_envs, self._obs_history_length), 
            dtype=torch.float32,
            device=self.device
        )
        
        self._gap_inner_history = torch.zeros(
            (self.num_envs, self._obs_history_length), 
            dtype=torch.float32,
            device=self.device
        )
        self._gap_outer_history = torch.zeros(
            (self.num_envs, self._obs_history_length), 
            dtype=torch.float32,
            device=self.device
        )        
        
        self._opponent_overtaken_counter = 0
        self._opponent_collision_counter = 0
        self._wall_collision_counter = 0

        self._opponent_min_vel_scaling = CONFIG['env_config']['OPP_MIN_VEL_SCALING']

        self._opponent_traj_x_shift = torch.zeros(
            self.num_envs, 
            dtype=torch.float32,
            device=self.device
        )        
        self._opponent_traj_y_shift = torch.zeros(
            self.num_envs, 
            dtype=torch.float32,
            device=self.device
        )                            
                
        ######################################################################
        ################# TRACK INFO STORED AS TENSOR IN LISTS ###############
        self._waypoints_list = [
            torch.tensor(wps, device=self.device, dtype=torch.float32)
            for wps in cfg.waypoints_list
        ]

        self._outer_list = [
            torch.tensor(outer, device=self.device, dtype=torch.float32)
            for outer in cfg.outer_list
        ]

        self._inner_list = [
            torch.tensor(inner, device=self.device, dtype=torch.float32)
            for inner in cfg.inner_list
        ]

        self._d_lat_list = [
            torch.tensor(d_lat, device=self.device, dtype=torch.float32)
            for d_lat in cfg.d_lat_list
        ]

        self._psi_rad_list = [
            torch.tensor(psi, device=self.device, dtype=torch.float32)
            for psi in cfg.psi_rad_list
        ]

        self._kappa_radpm_list = [
            torch.tensor(kappa, device=self.device, dtype=torch.float32)
            for kappa in cfg.kappa_radpm_list
        ]

        self._vx_mps_list = [
            torch.tensor(vx, device=self.device, dtype=torch.float32)
            for vx in cfg.vx_mps_list
        ]

        self._opp_traj_center_list = [
            torch.tensor(center, device=self.device, dtype=torch.float32)
            for center in cfg.opp_traj_center_list
        ]

        self._opp_traj_iqp_list = [
            torch.tensor(iqp, device=self.device, dtype=torch.float32)
            for iqp in cfg.opp_traj_iqp_list
        ]

        self._opp_traj_sp_list = [
            torch.tensor(sp, device=self.device, dtype=torch.float32)
            for sp in cfg.opp_traj_sp_list
        ]

        
@configclass
class F1TenthOvertakeRLRandomEnvCfg(F1TenthOvertakeRLEnvCfg):
    events: F1TenthOvertakeEventsRandomCfg = F1TenthOvertakeEventsRandomCfg()

######################
###### PLAY ENV ######
######################

@configclass
class F1TenthOvertakePlayEnvCfg(F1TenthOvertakeRLEnvCfg):
    """no terminations"""
  
    # on startup
    if CONFIG['env_config']['RESET_RANDOM']:
        events: F1TenthOvertakeEventsCfg = F1TenthOvertakeEventsRandomCfg(
            reset_root_state_random = EventTerm(
                func=reset_root_state_random_with_opponent,
                mode="reset",
            )
        )
    else:
        events: F1TenthOvertakeEventsCfg = F1TenthOvertakeEventsRandomCfg(
            reset_root_state_start_idx = EventTerm(
                func=reset_root_state_start_idx,
                mode="reset",
            )
        )

    rewards: F1TenthOvertakeRewardsCfg = None
    terminations: F1TenthOvertakeTerminationsCfg = None

    def __post_init__(self):
        super().__post_init__()