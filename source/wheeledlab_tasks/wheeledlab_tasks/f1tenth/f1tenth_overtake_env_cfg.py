"""
F1TENTH Overtake RL Environment (Isaac Lab)
==========================================

This module defines a manager-based RL environment for *overtaking* on
F1TENTH tracks. It extends the time-trial setup by introducing an opponent
(actor) that moves along a parameterized trajectory. The agent learns to
safely and efficiently overtake while staying on track.

**Highlights**
- Thorough docstrings and comments for hand-off readiness.
- Trimmed imports and consistent typing.
- Clear separation of concerns (observations, terrain, scene, events,
  rewards, curriculum, terminations, runtime env buffers).
- Opponent object is a lightweight `RigidObject` (cylinder) with kinematic
  control via an event hook (`move_opponent_s_based`).

**Key classes**
- `F1TenthOvertakeObsCfg` – policy-facing observation terms (ego + opponent).
- `F1TenthOvertakeTerrainImporterCfg` – track data & spawn utilities including
  opponent trajectory lists.
- `F1TenthOvertakeSceneCfg` – ground, light, robot and the opponent.
- `F1TenthOvertakeEventsCfg` – reset logic (ego + opponent) and the periodic
  opponent-motion event.
- `F1TenthOvertakeRewardsCfg` – progress + safety + overtake-specific rewards.
- `OvertakeCurriculumCfg` – curriculum to emphasize overtake progress while
  increasing collision penalties and shaping terms over time.
- `F1TenthOvertakeTerminationsCfg` – timeout, success/failure and track exits.
- `F1TenthOvertakeRLEnvCfg` – top-level RL env wiring & terrain build.
- `F1TenthOvertakeEnv` – runtime tensors used by MDP terms.

"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from wheeledlab.envs.mdp import increase_reward_weight_over_time
from wheeledlab_assets import WHEELEDLAB_ASSETS_DATA_DIR
from wheeledlab_assets.f1tenth import F1TENTH_CFG
from wheeledlab_tasks.common import F1Tenth4WDActionCfg
from wheeledlab_tasks.config_loader import load_config

# Local helpers and MDP callables
from .disable_lidar import disable_all_lidars
from .mdp import (
    reset_root_state_random,
    reset_root_state_random_with_opponent,
    reset_root_state_start_idx,
)
from .mdp.events import move_opponent_s_based
from .utils import (
    create_maps_from_waypoints,
    generate_random_poses_from_waypoints_with_opponent,
)

# Observation / reward / termination callables
from .mdp.observations import (
    action_history,
    base_ang_vel_z_history,
    base_lin_vel_x_history,
    gaps_info_history,
    opponent_relative_info_history,
    target_steering_angle_history,
    target_velocity_history,
    track_info_horizon,
)
from .mdp.rewards import (
    delta_steering_l2_penalty,
    delta_target_velocity_penalty,
    effort_steering_penalty,
    effort_target_steering_angle_penalty,
    opponent_collision_penalty,
    opponent_overtake_completed_reward,
    opponent_overtake_delta_distance_reward,
    opponent_overtake_distance_reward,
    progress_rew,
    side_slip_penalty,
    soft_wall_collision_penalty,
    var_steering_penalty,
    var_throttle_penalty,
    wall_collision_penalty,
)
from .mdp.terminations import (
    far_from_opponent,
    opponent_collision,
    opponent_overtaken,
    wall_collision,
)

# -----------------------------------------------------------------------------
# Global configuration & constants
# -----------------------------------------------------------------------------
CONFIG = load_config()
ENV_CFG = CONFIG["env_config"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE_USD_PATH = PROJECT_ROOT / "wheeledlab_tasks" / "f1tenth" / "utils" / "stage_usd"
MAPS_FOLDER_PATH = PROJECT_ROOT / "wheeledlab_tasks" / "f1tenth" / "utils" / "maps"

# Observation horizon parameters
N_HORIZON: int = ENV_CFG["N_HORIZON"]
DELTA_S_IDX: int = ENV_CFG["DELTA_S_IDX"]
T_HORIZON: int = ENV_CFG["T_HORIZON"]

# Physics material defaults
STATIC_FRICTION: float = ENV_CFG["STATIC_FRICTION"]
DYNAMIC_FRICTION: float = ENV_CFG["DYNAMIC_FRICTION"]
RESTITUTION: float = ENV_CFG["RESTITUTION"]


# -----------------------------------------------------------------------------
# Observation configuration
# -----------------------------------------------------------------------------
@configclass
class F1TenthOvertakeObsCfg:
    """Observation specs for overtake: ego history + opponent & gap features."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Concatenated policy observation vector (proprio + track + opponent)."""

        base_lin_vel_x_history = ObsTerm(
            func=base_lin_vel_x_history, params={"mean_noise": 0.0, "std_noise": 0.05}
        )
        base_ang_vel_z_history = ObsTerm(
            func=base_ang_vel_z_history, params={"mean_noise": 0.0, "std_noise": 0.01}
        )
        if CONFIG['env_config']['INCREMENTAL_MODE']:
            target_velocity_history = ObsTerm(
                func=target_velocity_history, params={"mean_noise": 0.0, "std_noise": 0.0}
            )
            target_steering_angle_history = ObsTerm(
                func=target_steering_angle_history, params={"mean_noise": 0.0, "std_noise": 0.0}
            )
            
        action_history = ObsTerm(func=action_history)

        track_info_horizon = ObsTerm(
            func=track_info_horizon,
            params={
                "delta_s_idx": DELTA_S_IDX,
                "n_horizon": N_HORIZON,
                "t_horizon": T_HORIZON,
                "position_std_noise": 0.0,
            },
        )

        opponent_relative_info_history = ObsTerm(
            func=opponent_relative_info_history,
            params={"position_std_noise": 0.00, "velocity_std_noise": 0.0},
        )

        gaps_info = ObsTerm(func=gaps_info_history)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# -----------------------------------------------------------------------------
# Initial pose container
# -----------------------------------------------------------------------------
@configclass
class InitialPoseCfg:
    """Spawn pose and initial velocities for ego/opponent."""

    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot_euler_xyz_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    lin_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)


# -----------------------------------------------------------------------------
# Terrain / Track configuration (includes opponent trajectories)
# -----------------------------------------------------------------------------
@configclass
class F1TenthOvertakeTerrainImporterCfg(TerrainImporterCfg):
    """Track geometry & helpers including opponent reference trajectories."""

    # Populated at runtime
    map_name_list: list | None = None
    origin_list: list | None = None

    traversability_hashmap_list: list | None = None
    waypoints_list: list | None = None
    outer_list: list | None = None
    inner_list: list | None = None
    d_lat_list: list | None = None
    psi_rad_list: list | None = None
    kappa_radpm_list: list | None = None
    vx_mps_list: list | None = None
    spacing_meters_list: list | None = None
    map_size_pixels_list: list | None = None
    row_spacing_list: list | None = None
    col_spacing_list: list | None = None
    num_cols_list: list | None = None
    num_rows_list: list | None = None
    width_list: list | None = None
    height_list: list | None = None

    # Opponent trajectory references (per map)
    opp_traj_center_list: list | None = None
    opp_traj_iqp_list: list | None = None
    opp_traj_sp_list: list | None = None

    # Static defaults
    env_spacing: float = 0.0
    prim_path: str = "/World/ground"
    terrain_type: str = "usd"
    usd_path: str | None = None
    collision_group: int = -1
    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="max",
        static_friction=STATIC_FRICTION,
        dynamic_friction=DYNAMIC_FRICTION,
        restitution=RESTITUTION,
    )
    debug_vis: bool = True

    # --- Utilities ---------------------------------------------------------
    def generate_random_poses_from_waypoints_with_opponent(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        num_poses: int,
        max_radius_offset: float = 0.3,
    ) -> tuple[list[InitialPoseCfg], torch.Tensor, list[InitialPoseCfg], torch.Tensor]:
        """Sample ego and opponent poses near track waypoints with mild jitter."""
        env_origins = env.scene.env_origins
        map_levels = env._map_levels

        (
            ego_init_poses,
            ego_init_idx,
            _ego_velocities,
            opp_init_poses,
            opp_init_idx,
            _opp_velocities,
        ) = generate_random_poses_from_waypoints_with_opponent(
            env_ids,
            num_poses,
            map_levels,
            env_origins,
            self.waypoints_list,
            self.inner_list,
            self.vx_mps_list,
            max_radius_offset=max_radius_offset,
        )

        jitter = 0.5
        ego_valid = [
            InitialPoseCfg(
                pos=(x + float(torch.rand(()).item() * 2 - 1) * jitter,
                     y + float(torch.rand(()).item() * 2 - 1) * jitter,
                     0.02),
                rot_euler_xyz_deg=(0.0, 0.0, angle),
                lin_vel=(float(torch.rand(()).item() * 2), 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
            )
            for x, y, angle in ego_init_poses
        ]

        opp_valid = [
            InitialPoseCfg(
                pos=(x + float(torch.rand(()).item() * 2 - 1) * jitter,
                     y + float(torch.rand(()).item() * 2 - 1) * jitter,
                     0.02),
                rot_euler_xyz_deg=(0.0, 0.0, angle),
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
            )
            for x, y, angle in opp_init_poses
        ]
        return ego_valid, ego_init_idx, opp_valid, opp_init_idx


# -----------------------------------------------------------------------------
# Scene configuration (adds an opponent object)
# -----------------------------------------------------------------------------
@configclass
class F1TenthOvertakeSceneCfg(InteractiveSceneCfg):
    """Scene with ground plane, lighting, ego robot and a simple opponent."""

    terrain = None

    ground = AssetBaseCfg(
        prim_path="/World/base",
        spawn=sim_utils.GroundPlaneCfg(
            size=(1500, 1500),
            color=(0.0, 0.0, 0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=STATIC_FRICTION,
                dynamic_friction=DYNAMIC_FRICTION,
            ),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.5, 0.5, 0.5), intensity=1500.0),
    )

    robot: AssetBaseCfg = F1TENTH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    opponent = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Opponent",
        spawn=sim_utils.CylinderCfg(
            radius=ENV_CFG["OPP_SIZE_RADIUS"],
            height=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                rigid_body_enabled=True,
                solver_position_iteration_count=1,
                solver_velocity_iteration_count=1,
                max_angular_velocity=100.0,
                max_linear_velocity=100.0,
                max_depenetration_velocity=1.0,
                disable_gravity=True,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    ground.init_state.pos = (0.0, 0.0, -1e-4)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.robot.init_state = self.robot.init_state.replace(pos=(0.0, 0.0, 0.0))
        self.opponent.init_state = self.opponent.init_state.replace(pos=(0.0, 0.0, 0.0))


# -----------------------------------------------------------------------------
# Events: resets, domain randomization, and opponent motion
# -----------------------------------------------------------------------------
@configclass
class F1TenthOvertakeEventsCfg:
    """Resets for ego/opponent and periodic opponent motion."""

    if ENV_CFG["RESET_RANDOM"]:
        reset_root_state_random = EventTerm(
            func=reset_root_state_random_with_opponent, mode="reset"
        )
    else:
        reset_root_state_start_idx = EventTerm(
            func=reset_root_state_start_idx, mode="reset"
        )

    move_opponent_s_based = EventTerm(
        func=move_opponent_s_based,
        mode="interval",
        interval_range_s=(ENV_CFG["OPP_MOVE_DT"], ENV_CFG["OPP_MOVE_DT"]),
    )


@configclass
class F1TenthOvertakeEventsRandomCfg(F1TenthOvertakeEventsCfg):
    """Optional domain randomization of friction, mass, and actuators."""

    if ENV_CFG["RANDOMIZE_FRICTION"]:
        change_wheel_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "static_friction_range": (
                    ENV_CFG["STATIC_FRICTION"] - ENV_CFG["STD_FRICTION"],
                    ENV_CFG["STATIC_FRICTION"] + ENV_CFG["STD_FRICTION"],
                ),
                "dynamic_friction_range": (
                    ENV_CFG["DYNAMIC_FRICTION"] - ENV_CFG["STD_FRICTION"],
                    ENV_CFG["DYNAMIC_FRICTION"] + ENV_CFG["STD_FRICTION"],
                ),
                "restitution_range": (0.0, 0.0),
                "num_buckets": ENV_CFG["NUM_BUCKETS_FRICTION"],
                "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_.*link"),
                "make_consistent": True,
            },
        )

    if ENV_CFG["RANDOMIZE_BODY_MASS"]:
        add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
                "mass_distribution_params": (0.99, 1.01),
                "operation": "scale",
            },
        )

    if ENV_CFG.get("RANDOMIZE_ACTUATOR_STEERING_GAIN", False):
        randomize_steering_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["rotator_(left|right)"]),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

    if ENV_CFG.get("RANDOMIZE_ACTUATOR_THROTTLE_GAIN", False):
        randomize_throttle_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*wheel_(back|front)_.*"]),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

    kill_lidar = EventTerm(func=disable_all_lidars, mode="startup", params={})


# -----------------------------------------------------------------------------
# Rewards & curriculum
# -----------------------------------------------------------------------------
@configclass
class F1TenthOvertakeRewardsCfg:
    """Reward terms for overtaking (set weight=0 to disable a term)."""

    # Progress + safety
    progress_rew = RewTerm(func=progress_rew, 
                           weight=1.0)
    wall_collision_penalty = RewTerm(func=wall_collision_penalty, 
                                     weight=1.5)
    soft_wall_collision_penalty = RewTerm(func=soft_wall_collision_penalty, 
                                          weight=0.5)
    side_slip_penalty = RewTerm(func=side_slip_penalty, 
                                weight=0.5, 
                                params={"slip_thresh": 0.18})

    # Overtake-specific
    opponent_collision_penalty = RewTerm(func=opponent_collision_penalty, 
                                         weight=10.0)
    
    opponent_overtake_completed_reward = RewTerm(
        func=opponent_overtake_completed_reward, 
        weight=500.0
    )
    opponent_overtake_distance_reward = RewTerm(
        func=opponent_overtake_distance_reward, 
        weight=0.01
    )
    opponent_overtake_delta_distance_reward = RewTerm(
        func=opponent_overtake_delta_distance_reward, 
        weight=0.01
    )

    # Action smoothing / effort
    delta_target_velocity_penalty = RewTerm(
        func=delta_target_velocity_penalty, weight=0.5
    )
    var_throttle_penalty = RewTerm(func=var_throttle_penalty, weight=0.001)
    var_steering_penalty = RewTerm(func=var_steering_penalty, weight=0.04)
    effort_steering_penalty = RewTerm(func=effort_steering_penalty, weight=0.1)
    effort_abs_steering_penalty = RewTerm(
        func=effort_target_steering_angle_penalty, weight=0.25
    )
    delta_steering_l2_penalty = RewTerm(
        func=delta_steering_l2_penalty, weight=0.25
    )


MAX_NUM_INC = 0


@configclass
class OvertakeCurriculumCfg:
    """Curriculum focusing on safe, decisive overtakes over episodes."""

    progress_rew = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "progress_rew",
            "weight_increase": -0.20,
            "min_weight": 0.25,
            "first_episode_increase": 25,
            "episodes_per_increase": 25,
            "max_num_increases": MAX_NUM_INC,
        },
    )

    wall_collision_penalty = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "wall_collision_penalty",
            "weight_increase": 0.1,
            "max_weight": 5.0,
            "first_episode_increase": 20,
            "episodes_per_increase": 20,
            "max_num_increases": MAX_NUM_INC,
        },
    )

    opponent_collision_penalty = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "opponent_collision_penalty",
            "weight_increase": 0.1,
            "max_weight": 5.0,
            "first_episode_increase": 20,
            "episodes_per_increase": 20,
            "max_num_increases": MAX_NUM_INC,
        },
    )

    opponent_overtake_delta_distance_reward = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "opponent_overtake_delta_distance_reward",
            "weight_increase": 2.0,
            "max_weight": 20.0,
            "first_episode_increase": 50,
            "episodes_per_increase": 50,
            "max_num_increases": MAX_NUM_INC,
        },
    )

    opponent_overtake_distance_reward = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "opponent_overtake_distance_reward",
            "weight_increase": 0.05,
            "max_weight": 1.0,
            "first_episode_increase": 50,
            "episodes_per_increase": 50,
            "max_num_increases": MAX_NUM_INC,
        },
    )

    var_throttle_penalty = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "var_throttle_penalty",
            "weight_increase": 0.5,
            "max_weight": 1.5,
            "first_episode_increase": 50,
            "episodes_per_increase": 50,
            "max_num_increases": MAX_NUM_INC,
        },
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
        },
    )


# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------
@configclass
class F1TenthOvertakeTerminationsCfg:
    """Overtake episode termination conditions."""

    # Always cap episode duration
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Success/failure signals
    opponent_overtaken = DoneTerm(func=opponent_overtaken)
    far_from_opponent = DoneTerm(func=far_from_opponent)
    opponent_collision = DoneTerm(func=opponent_collision)

    # Optional: track exit/end
    if ENV_CFG["NON_TRAVERSABLE_TERMINATION"]:
        wall_collision = DoneTerm(func=wall_collision)


# -----------------------------------------------------------------------------
# RL Env Config (terrain build, scene assembly, wiring)
# -----------------------------------------------------------------------------
@configclass
class F1TenthOvertakeRLEnvCfg(ManagerBasedRLEnvCfg):
    """Top-level RL config for the overtake task."""

    seed: int = ENV_CFG["RANDOM_SEED"]
    num_envs: int = 1
    env_spacing: float = 0.0

    MAP_NAME_LIST: List[str] = ENV_CFG["MAP_NAME_LIST"]

    events: F1TenthOvertakeEventsCfg = F1TenthOvertakeEventsCfg()
    actions: F1Tenth4WDActionCfg = F1Tenth4WDActionCfg()
    observations: F1TenthOvertakeObsCfg = F1TenthOvertakeObsCfg()
    rewards: F1TenthOvertakeRewardsCfg = F1TenthOvertakeRewardsCfg()
    terminations: F1TenthOvertakeTerminationsCfg = F1TenthOvertakeTerminationsCfg()
    curriculum: OvertakeCurriculumCfg = OvertakeCurriculumCfg()

    # Lists copied out for the runtime env
    waypoints_list: list | None = None
    outer_list: list | None = None
    inner_list: list | None = None
    d_lat_list: list | None = None
    psi_rad_list: list | None = None
    kappa_radpm_list: list | None = None
    vx_mps_list: list | None = None
    opp_traj_center_list: list | None = None
    opp_traj_iqp_list: list | None = None
    opp_traj_sp_list: list | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        print("[INFO]: F1TenthOvertakeRLEnvCfg.__post_init__ START")

        # Viewer & sim
        self.viewer.eye = [0.0, 0.0, 40.0]
        self.viewer.lookat = [0.0, 0.0, -3.0]
        self.sim.dt = ENV_CFG["SIM_DT"]
        self.decimation = ENV_CFG["SIM_DECIMATION"]
        self.sim.render_interval = self.decimation

        # Episode & action scaling
        self.episode_length_s = ENV_CFG["EPISODE_LENGTH_S_OVERTAKE"]
        self.actions.throttle_steer.scale = (
            ENV_CFG["MAX_SPEED_SCALING"],
            ENV_CFG["MAX_STEERING_ANGLE"],
        )
        self.actions.throttle_steer.offset = (
            ENV_CFG["SPEED_OFFSET"],
            ENV_CFG["STEERING_OFFSET"],
        )

        # Build a consolidated USD stage for all requested maps
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stage_path = (
            Path(STAGE_USD_PATH) / f"{timestamp}_stage.usd"
        ).as_posix()
        origin_list = ENV_CFG["ORIGIN_LIST"]

        (
            waypoints_list,
            outer_list,
            inner_list,
            d_lat_list,
            psi_rad_list,
            kappa_radpm_list,
            vx_mps_list,
            opp_traj_center_list,
            opp_traj_iqp_list,
            opp_traj_sp_list,
            spacing_meters_list,
            map_size_pixels_list,
        ) = create_maps_from_waypoints(
            maps_folder_path=MAPS_FOLDER_PATH,
            map_name_list=self.MAP_NAME_LIST,
            origin_list=origin_list,
            stage_path=stage_path,
            resolution=0.1,
        )

        # Derived geometry
        row_spacing_list = np.array(spacing_meters_list)[:, 0].tolist()
        col_spacing_list = np.array(spacing_meters_list)[:, 1].tolist()
        num_cols_list = np.array(map_size_pixels_list)[:, 0].tolist()
        num_rows_list = np.array(map_size_pixels_list)[:, 1].tolist()
        width_list = (np.array(num_rows_list) * np.array(row_spacing_list)).tolist()
        height_list = (np.array(num_cols_list) * np.array(col_spacing_list)).tolist()

        # Terrain config with opponent traj lists
        self.terrain = F1TenthOvertakeTerrainImporterCfg(
            prim_path="/World/envs/env_.*",
            env_spacing=self.env_spacing,
            usd_path=stage_path,
            map_name_list=self.MAP_NAME_LIST,
            traversability_hashmap_list=[],
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
            opp_traj_center_list=opp_traj_center_list,
            opp_traj_iqp_list=opp_traj_iqp_list,
            opp_traj_sp_list=opp_traj_sp_list,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="max",
                static_friction=STATIC_FRICTION,
                dynamic_friction=DYNAMIC_FRICTION,
                restitution=RESTITUTION,
            ),
            debug_vis=True,
        )

        # Copy lists for runtime env access
        self.waypoints_list = waypoints_list
        self.outer_list = outer_list
        self.inner_list = inner_list
        self.d_lat_list = d_lat_list
        self.psi_rad_list = psi_rad_list
        self.kappa_radpm_list = kappa_radpm_list
        self.vx_mps_list = vx_mps_list
        self.opp_traj_center_list = opp_traj_center_list
        self.opp_traj_iqp_list = opp_traj_iqp_list
        self.opp_traj_sp_list = opp_traj_sp_list

        # Assemble scene & bind env class
        self.scene = F1TenthOvertakeSceneCfg(num_envs=self.num_envs, env_spacing=self.env_spacing, terrain=self.terrain)
        self.env_class = F1TenthOvertakeEnv
        print("[INFO]: F1TenthOvertakeRLEnvCfg.__post_init__ END")


# -----------------------------------------------------------------------------
# Runtime env (buffers for observations/rewards/terminations)
# -----------------------------------------------------------------------------
class F1TenthOvertakeEnv(ManagerBasedEnv):
    """Runtime state & buffers consumed by MDP terms during training."""

    def __init__(self, cfg: F1TenthOvertakeRLEnvCfg, **kwargs) -> None:
        super().__init__(cfg, **kwargs)

        # --- Waypoint / progress bookkeeping --------------------------------
        self._initial_waypoint_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_delta_s_opp_ego = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # History sizes (with backward-compatible key for the checkpoint idx)
        self._action_history_length: int = ENV_CFG["ACTION_HISTORY_LENGTH"]
        self._obs_history_length: int = ENV_CFG["OBS_HISTORY_LENGTH"]
        self._rew_history_length: int = ENV_CFG["REW_HISTORY_LENGTH"]
        self._progress_history_checkpoint_idx: int = ENV_CFG.get(
            "PROGRESS_HISTORY_CHECK_IDX", ENV_CFG.get("PROGRESS_HISTORY_CHECKPOINT_IDX", 0)
        )

        self._progress_history_indices = torch.zeros(
            (self.num_envs, self._rew_history_length), dtype=torch.long, device=self.device
        )

        # --- Ego action/state histories -------------------------------------
        self._action_history = torch.zeros((self.num_envs, self._action_history_length, 2), dtype=torch.float32, device=self.device)
        self._base_lin_vel_x_history = torch.zeros((self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device)
        self._base_lin_vel_y_history = torch.zeros((self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device)
        self._base_ang_vel_z_history = torch.zeros((self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device)

        self._traversability_history = torch.ones((self.num_envs, self._rew_history_length), dtype=torch.long, device=self.device)
        self._wall_collision_history = torch.zeros((self.num_envs, self._rew_history_length), dtype=torch.long, device=self.device)

        self._reset_env_bool = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._current_progress = torch.zeros(self.num_envs, device=self.device)
        self._map_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Targets & auxiliaries
        self._last_velocity_adjustment = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._vel_y_calc = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._target_steering_angle = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._target_velocity = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._target_velocity_history = torch.zeros((self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device)
        self._target_steering_angle_history = torch.zeros((self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device)

        # --- Opponent-related buffers ----------------------------------------
        self._opponent_type = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._opponent_vel_scaling = torch.ones(self.num_envs, dtype=torch.float16, device=self.device)
        self._opponent_trajectory_alpha = torch.ones(self.num_envs, dtype=torch.float16, device=self.device)
        self._opponent_overtaken_bool = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._opponent_overtaken_history = torch.zeros((self.num_envs, self._rew_history_length), dtype=torch.long, device=self.device)
        self._opponent_speed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._opponent_d_dot = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._opponent_heading = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._opponent_always_ahead = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._opponent_collision_history = torch.zeros((self.num_envs, self._rew_history_length), dtype=torch.long, device=self.device)

        self._opponent_s_history = torch.zeros((self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device)
        self._opponent_d_history = torch.zeros((self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device)
        self._opponent_xy_position = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        self._s_idx_diff_history = torch.zeros((self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device)
        self._d_diff_history = torch.zeros((self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device)
        self._vx_diff_history = torch.zeros((self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device)
        self._heading_diff_history = torch.zeros((self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device)
        self._cross_pos_history = torch.zeros((self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device)
        self._gap_inner_history = torch.zeros((self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device)
        self._gap_outer_history = torch.zeros((self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device)

        self._opponent_overtaken_counter = 0
        self._opponent_collision_counter = 0
        self._wall_collision_counter = 0

        self._opponent_min_vel_scaling = ENV_CFG["OPP_MIN_VEL_SCALING"]
        self._opponent_traj_x_shift = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._opponent_traj_y_shift = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # --- Map geometry (copied from cfg) ----------------------------------
        self._waypoints_list = [torch.tensor(wps, device=self.device, dtype=torch.float32) for wps in (cfg.waypoints_list or [])]
        self._outer_list = [torch.tensor(o, device=self.device, dtype=torch.float32) for o in (cfg.outer_list or [])]
        self._inner_list = [torch.tensor(i, device=self.device, dtype=torch.float32) for i in (cfg.inner_list or [])]
        self._d_lat_list = [torch.tensor(d, device=self.device, dtype=torch.float32) for d in (cfg.d_lat_list or [])]
        self._psi_rad_list = [torch.tensor(p, device=self.device, dtype=torch.float32) for p in (cfg.psi_rad_list or [])]
        self._kappa_radpm_list = [torch.tensor(k, device=self.device, dtype=torch.float32) for k in (cfg.kappa_radpm_list or [])]
        self._vx_mps_list = [torch.tensor(v, device=self.device, dtype=torch.float32) for v in (cfg.vx_mps_list or [])]
        self._opp_traj_center_list = [torch.tensor(c, device=self.device, dtype=torch.float32) for c in (cfg.opp_traj_center_list or [])]
        self._opp_traj_iqp_list = [torch.tensor(q, device=self.device, dtype=torch.float32) for q in (cfg.opp_traj_iqp_list or [])]
        self._opp_traj_sp_list = [torch.tensor(s, device=self.device, dtype=torch.float32) for s in (cfg.opp_traj_sp_list or [])]


# -----------------------------------------------------------------------------
# Variants: randomized training & play
# -----------------------------------------------------------------------------
@configclass
class F1TenthOvertakeRLRandomEnvCfg(F1TenthOvertakeRLEnvCfg):
    events: F1TenthOvertakeEventsRandomCfg = F1TenthOvertakeEventsRandomCfg()


@configclass
class F1TenthOvertakePlayEnvCfg(F1TenthOvertakeRLEnvCfg):
    """Inference-only env (no rewards/terminations), keeps opponent motion."""

    if ENV_CFG["RESET_RANDOM"]:
        events: F1TenthOvertakeEventsCfg = F1TenthOvertakeEventsRandomCfg(
            reset_root_state_random=EventTerm(func=reset_root_state_random_with_opponent, mode="reset")
        )
    else:
        events: F1TenthOvertakeEventsCfg = F1TenthOvertakeEventsRandomCfg(
            reset_root_state_start_idx=EventTerm(func=reset_root_state_start_idx, mode="reset")
        )

    rewards: F1TenthOvertakeRewardsCfg | None = None
    terminations: F1TenthOvertakeTerminationsCfg | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
