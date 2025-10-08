"""
F1TENTH Time-Trial RL Environment (Isaac Lab)
=============================================

This module defines a *manager-based* RL environment that trains an F1TENTH
RC car to minimize lap time on various race-track maps using Isaac Lab.

**What you'll find here**
- Clear, high-level docstrings on every public class and method.
- Tight imports (unused imports removed) and consistent typing.
- Grouped constants and CONFIG usage in one place.
- Comments explaining *why*, not just *what*.
- Reduced redundancy and sharper naming.

**Key Objects**
- `F1TenthTimeTrialObsCfg`: Observation terms exposed to the policy.
- `F1TenthTimeTrialTerrainImporterCfg`: Track/terrain assets & helpers for initial poses.
- `F1TenthTimeTrialSceneCfg`: Scene composition (ground plane, lights, robot).
- `F1TenthTimeTrialEventsCfg`: Episodic reset and domain-randomization hooks.
- `F1TenthTimeTrialRewardsCfg`: Reward terms and default weights.
- `TimeTrialCurriculumCfg`: Curriculum scheduling for reward weights.
- `F1TenthTimeTrialTerminationsCfg`: Episode termination conditions.
- `F1TenthTimeTrialRLEnvCfg`: Top-level RL env config (ties everything together).
- `F1TenthTimeTrialEnv`: The runtime env that holds tensors/buffers used by
  observations/rewards/terminations.

**Notes**
- The env uses an external YAML/Dict config via `wheeledlab_tasks.config_loader.load_config()`.
- Track assets are generated once at startup into a USD stage.
- You can disable/enable many knobs in `CONFIG['env_config']`.

"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils  # noqa: F401 (often imported by downstream code)
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
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

from wheeledlab.envs.mdp import (
    increase_reward_weight_over_time,
    increase_reward_weight_over_time_every_n_steps,  # noqa: F401 (kept for optional curriculum)
)
from wheeledlab_assets import WHEELEDLAB_ASSETS_DATA_DIR
from wheeledlab_assets.f1tenth import F1TENTH_CFG
from wheeledlab_tasks.common import F1Tenth4WDActionCfg
from wheeledlab_tasks.config_loader import load_config

# Local helpers (kept explicit to make import surface obvious)
from .disable_lidar import disable_all_lidars
from .mdp import reset_root_state_random, reset_root_state_start_idx
from .utils import (
    create_maps_from_waypoints,
    generate_random_poses_from_waypoints,
    generate_start_idx_poses_from_list,
)

# Observation / reward / termination callables (referenced by name in cfgs)
from .mdp.observations import (
    action_history,
    base_ang_vel_z_history,
    base_lin_vel_x_history,
    target_steering_angle_history,
    target_velocity_history,
    track_info_horizon,
)
from .mdp.rewards import (
    delta_steering_l2_penalty,
    delta_target_velocity_penalty,
    effort_target_steering_angle_penalty,
    effort_steering_penalty,
    progress_rew,
    side_slip_penalty,
    soft_wall_collision_penalty,
    var_steering_penalty,
    var_throttle_penalty,
    wall_collision_penalty,
)
from .mdp.terminations import wall_collision

# -----------------------------------------------------------------------------
# Global configuration & constants
# -----------------------------------------------------------------------------
CONFIG = load_config()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE_USD_PATH = PROJECT_ROOT / "wheeledlab_tasks" / "f1tenth" / "utils" / "stage_usd"
MAPS_FOLDER_PATH = PROJECT_ROOT / "wheeledlab_tasks" / "f1tenth" / "utils" / "maps"

# Sub-config shortcuts (kept close to top so changing values is obvious)
ENV_CFG = CONFIG["env_config"]
N_HORIZON: int = ENV_CFG["N_HORIZON"]
DELTA_S_IDX: int = ENV_CFG["DELTA_S_IDX"]
T_HORIZON: int = ENV_CFG["T_HORIZON"]

STATIC_FRICTION: float = ENV_CFG["STATIC_FRICTION"]
DYNAMIC_FRICTION: float = ENV_CFG["DYNAMIC_FRICTION"]
RESTITUTION: float = ENV_CFG["RESTITUTION"]

# -----------------------------------------------------------------------------
# Observation configuration
# -----------------------------------------------------------------------------
@configclass
class F1TenthTimeTrialObsCfg:
    """Observation specifications.

    The *Policy* observation group concatenates its terms into a single tensor
    by default (see `__post_init__`). This is the vector the policy consumes.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Low-latency proprioceptive and map-centric features for control.

        Each `ObsTerm` references a callable (imported above) and optional
        parameters (e.g., observation noise). Keep noise small for sim2real
        stability but non-zero to avoid overfitting.
        """

        base_lin_vel_x_history = ObsTerm(
            func=base_lin_vel_x_history,
            params={"mean_noise": 0.0, "std_noise": 0.05},
        )

        base_ang_vel_z_history = ObsTerm(
            func=base_ang_vel_z_history,
            params={"mean_noise": 0.0, "std_noise": 0.01},
        )

        if CONFIG['env_config']['INCREMENTAL_MODE']:
            target_velocity_history = ObsTerm(
                func=target_velocity_history,
                params={"mean_noise": 0.0, "std_noise": 0.0},
            )

            target_steering_angle_history = ObsTerm(
                func=target_steering_angle_history,
                params={"mean_noise": 0.0, "std_noise": 0.0},
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

        def __post_init__(self) -> None:  # noqa: D401
            """Finalize defaults (kept here to avoid repetition elsewhere)."""
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# -----------------------------------------------------------------------------
# Initial pose data structure
# -----------------------------------------------------------------------------
@configclass
class InitialPoseCfg:
    """Simple structured container for spawn pose & initial velocities."""

    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot_euler_xyz_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    lin_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)


# -----------------------------------------------------------------------------
# Terrain / Track configuration
# -----------------------------------------------------------------------------
@configclass
class F1TenthTimeTrialTerrainImporterCfg(TerrainImporterCfg):
    """Holds track geometry and utilities to sample valid spawn poses.

    The lists below are populated at runtime from `create_maps_from_waypoints`.
    We keep them on the terrain cfg because they conceptually describe the
    *world*, and the env copies references it needs later (see RLEnvCfg).
    """

    # Data populated at runtime (default None keeps dataclass compact)
    map_name_list: List[str] | None = None
    origin_list: List[Tuple[float, float]] | None = None

    traversability_hashmap_list: List[np.ndarray] | None = None
    waypoints_list: List[np.ndarray] | None = None
    outer_list: List[np.ndarray] | None = None
    inner_list: List[np.ndarray] | None = None
    d_lat_list: List[np.ndarray] | None = None
    psi_rad_list: List[np.ndarray] | None = None
    kappa_radpm_list: List[np.ndarray] | None = None
    vx_mps_list: List[np.ndarray] | None = None
    spacing_meters_list: List[Tuple[float, float]] | None = None
    map_size_pixels_list: List[Tuple[int, int]] | None = None
    row_spacing_list: List[float] | None = None
    col_spacing_list: List[float] | None = None
    num_cols_list: List[int] | None = None
    num_rows_list: List[int] | None = None
    width_list: List[float] | None = None
    height_list: List[float] | None = None

    # Static defaults (physics & visualization)
    env_spacing: float = 0.0
    prim_path: str = "/World/ground"
    terrain_type: str = "usd"
    usd_path: str | None = None
    collision_group: int = -1
    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="average",
        restitution_combine_mode="max",
        static_friction=STATIC_FRICTION,
        dynamic_friction=DYNAMIC_FRICTION,
        restitution=RESTITUTION,
    )
    debug_vis: bool = True

    # --- Utilities ---------------------------------------------------------
    def generate_random_poses_from_waypoints(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        num_poses: int,
        max_radius_offset: float = 0.3,
    ) -> Tuple[List[InitialPoseCfg], torch.Tensor]:
        """Sample *num_poses* valid positions near track waypoints.

        Adds a small random offset in XY and forward linear velocity to induce
        variation. Returns both the list of `InitialPoseCfg` and the starting
        waypoint indices per environment (as a tensor on env.device).
        """
        env_origins = env.scene.env_origins
        map_levels = env._map_levels

        init_poses, init_wps_idx = generate_random_poses_from_waypoints(
            env_ids,
            num_poses,
            map_levels,
            env_origins,
            self.waypoints_list,
            self.inner_list,
            max_radius_offset=max_radius_offset,
        )

        jitter = 0.5
        valid_init_poses = [
            InitialPoseCfg(
                pos=(x + float(torch.rand(()).item() * 2 - 1) * jitter,
                     y + float(torch.rand(()).item() * 2 - 1) * jitter,
                     0.02),
                rot_euler_xyz_deg=(0.0, 0.0, angle),
                lin_vel=(float(torch.rand(()).item() * 2), 0.0, 0.0),
            )
            for x, y, angle in init_poses
        ]
        return valid_init_poses, init_wps_idx

    def generate_start_idx_poses(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        num_poses: int,
    ) -> Tuple[List[InitialPoseCfg], torch.Tensor]:
        """Spawn poses at deterministic start indices (e.g., on start grid)."""
        env_origins = env.scene.env_origins
        map_levels = env._map_levels

        init_poses, init_wps_idx = generate_start_idx_poses_from_list(
            env_ids,
            num_poses,
            map_levels,
            env_origins,
            self.row_spacing_list,
            self.col_spacing_list,
            self.traversability_hashmap_list,
            self.waypoints_list,
            self.outer_list,
            self.inner_list,
            margin=0.1,
        )
        valid_init_poses = [
            InitialPoseCfg(pos=(x, y, 0.02), rot_euler_xyz_deg=(0.0, 0.0, 0.0))
            for x, y, _angle in init_poses
        ]
        return valid_init_poses, init_wps_idx


# -----------------------------------------------------------------------------
# Scene configuration
# -----------------------------------------------------------------------------
@configclass
class F1TenthTimeTrialSceneCfg(InteractiveSceneCfg):
    """Scene with ground plane, lighting, and a single F1TENTH robot per env."""

    terrain = None  # set by RLEnvCfg.__post_init__

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
        spawn=sim_utils.DistantLightCfg(color=(0.5, 0.5, 0.5), intensity=500.0),
    )

    robot: AssetBaseCfg = F1TENTH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ground.init_state.pos = (0.0, 0.0, -1e-4)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Ensure robots spawn at ground level origin; reset per-env later via events
        self.robot.init_state = self.robot.init_state.replace(pos=(0.0, 0.0, 0.0))


# -----------------------------------------------------------------------------
# Events (resets & domain randomization)
# -----------------------------------------------------------------------------
@configclass
class F1TenthTimeTrialEventsCfg:
    """Reset behavior (randomized spawn or fixed start indices)."""

    if ENV_CFG["RESET_RANDOM"]:
        reset_root_state_random = EventTerm(func=reset_root_state_random, mode="reset")
    else:
        reset_root_state_start_idx = EventTerm(
            func=reset_root_state_start_idx, mode="reset"
        )


@configclass
class F1TenthTimeTrialEventsRandomCfg(F1TenthTimeTrialEventsCfg):
    """Optional domain randomization hooks (mass, friction, sensors...)."""

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
                "mass_distribution_params": (["MIN_MASS_SCALE"], ["MAX_MASS_SCALE"]),
                "operation": "scale",
            },
        )

    # Disable LiDARs by default to focus on proprioception + map embedding
    kill_lidar = EventTerm(func=disable_all_lidars, mode="startup", params={})


# -----------------------------------------------------------------------------
# Rewards & curriculum
# -----------------------------------------------------------------------------
@configclass
class F1TenthTimeTrialRewardsCfg:
    """Reward terms (set weight to 0.0 to disable a term)."""

    progress_rew = RewTerm(func=progress_rew, weight=1.0)
    wall_collision_penalty = RewTerm(func=wall_collision_penalty, weight=0.5)
    soft_wall_collision_penalty = RewTerm(func=soft_wall_collision_penalty, weight=1.0)

    side_slip_penalty = RewTerm(
        func=side_slip_penalty, weight=0.5, params={"slip_thresh": 0.14}
    )

    delta_target_velocity_penalty = RewTerm(
        func=delta_target_velocity_penalty, weight=1.0
    )

    var_throttle_penalty = RewTerm(func=var_throttle_penalty, weight=0.001)
    var_steering_penalty = RewTerm(func=var_steering_penalty, weight=0.01)

    effort_steering_penalty = RewTerm(func=effort_steering_penalty, weight=0.08)
    effort_abs_steering_penalty = RewTerm(
        func=effort_target_steering_angle_penalty, weight=0.01
    )
    delta_steering_l2_penalty = RewTerm(
        func=delta_steering_l2_penalty, weight=0.025
    )


@configclass
class TimeTrialCurriculumCfg:
    """Curriculum for gradually enabling/strengthening regularization terms."""

    average_vel_reward = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "average_vel",
            "weight_increase": 0.25,
            "max_weight": 0.5,
            "first_episode_increase": 750,
            "episodes_per_increase": 750,
            "max_num_increases": 0,
        },
    )

    wall_collision_penalty = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "wall_collision_penalty",
            "weight_increase": 0.0,
            "first_episode_increase": 25,
            "episodes_per_increase": 25,
            "max_num_increases": 0,
        },
    )

    var_throttle_penalty = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "var_throttle_penalty",
            "weight_increase": 0.5,
            "first_episode_increase": 500,
            "episodes_per_increase": 500,
            "max_num_increases": 0,
        },
    )

    var_steering_penalty = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "var_steering_penalty",
            "weight_increase": 0.1,
            "max_weight": 1.0,
            "first_episode_increase": 25,
            "episodes_per_increase": 25,
            "max_num_increases": 0,
        },
    )

    effort_abs_steering_penalty = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "effort_abs_steering_penalty",
            "weight_increase": 0.01,
            "max_weight": 0.1,
            "first_episode_increase": 25,
            "episodes_per_increase": 25,
            "max_num_increases": 0,
        },
    )


# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------
@configclass
class F1TenthTimeTrialTerminationsCfg:
    """Episode termination conditions (timeout + map-specific)."""

    # Always enforce a maximum episode length
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Optional: terminate on leaving the drivable area (via wall collision)
    if ENV_CFG["NON_TRAVERSABLE_TERMINATION"]:
        wall_collision = DoneTerm(func=wall_collision)


# -----------------------------------------------------------------------------
# RL Env Config (ties everything together)
# -----------------------------------------------------------------------------
@configclass
class F1TenthTimeTrialRLEnvCfg(ManagerBasedRLEnvCfg):
    """Top-level RL environment configuration for time-trial training."""

    # Overridden by hydra/launcher often
    seed: int = 42
    num_envs: int = 1
    env_spacing: float = 0.0

    # Tracks to load (order matters; used to shard envs across maps)
    MAP_NAME_LIST: List[str] = ENV_CFG["MAP_NAME_LIST"]

    # Manager groups
    events: F1TenthTimeTrialEventsCfg = F1TenthTimeTrialEventsCfg()
    actions: F1Tenth4WDActionCfg = F1Tenth4WDActionCfg()
    observations: F1TenthTimeTrialObsCfg = F1TenthTimeTrialObsCfg()
    rewards: F1TenthTimeTrialRewardsCfg = F1TenthTimeTrialRewardsCfg()
    terminations: F1TenthTimeTrialTerminationsCfg = F1TenthTimeTrialTerminationsCfg()
    curriculum: TimeTrialCurriculumCfg = TimeTrialCurriculumCfg()

    # The following fields are populated in __post_init__ to be consumed by
    # `F1TenthTimeTrialEnv` at runtime
    waypoints_list: List[np.ndarray] | None = None
    outer_list: List[np.ndarray] | None = None
    inner_list: List[np.ndarray] | None = None
    d_lat_list: List[np.ndarray] | None = None
    psi_rad_list: List[np.ndarray] | None = None
    kappa_radpm_list: List[np.ndarray] | None = None
    vx_mps_list: List[np.ndarray] | None = None
    opp_traj_center_list: List[np.ndarray] | None = None
    opp_traj_iqp_list: List[np.ndarray] | None = None
    opp_traj_sp_list: List[np.ndarray] | None = None

    def __post_init__(self) -> None:
        """Assemble the scene, terrain, and runtime lists from MAP_NAME_LIST."""
        super().__post_init__()
        print("[INFO]: F1TenthTimeTrialRLEnvCfg.__post_init__ START")

        # Viewer & sim timing (render decoupled from sim)
        self.viewer.eye = [0.0, 0.0, 70.0]
        self.viewer.lookat = [0.0, 0.0, -3.0]
        self.sim.dt = ENV_CFG["SIM_DT"]
        self.decimation = ENV_CFG["SIM_DECIMATION"]
        self.sim.render_interval = 10

        # Action scaling and episode length
        self.episode_length_s = ENV_CFG["EPISODE_LENGTH_S_TIMETRIAL"]
        self.actions.throttle_steer.scale = (
            ENV_CFG["MAX_SPEED_SCALING"],
            ENV_CFG["MAX_STEERING_ANGLE"],
        )
        self.actions.throttle_steer.offset = (
            ENV_CFG["SPEED_OFFSET"],
            ENV_CFG["STEERING_OFFSET"],
        )

        # --- Build USD stage for all maps (one stage to speed up startup) ---
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

        # Terrain cfg (keeps world/asset info)
        self.terrain = F1TenthTimeTrialTerrainImporterCfg(
            prim_path="/World/envs/env_.*",
            env_spacing=self.env_spacing,
            usd_path=stage_path,
            map_name_list=self.MAP_NAME_LIST,
            traversability_hashmap_list=[],  # unused for these tracks but kept for API
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
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="max",
                static_friction=STATIC_FRICTION,
                dynamic_friction=DYNAMIC_FRICTION,
                restitution=RESTITUTION,
            ),
            debug_vis=True,
        )

        # Copy lists to cfg so the runtime env can access them without digging
        # through scene/terrain. This mirrors patterns in Isaac Lab examples.
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

        # Assemble the scene
        self.scene = F1TenthTimeTrialSceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing, terrain=self.terrain
        )

        # Bind the runtime env class
        self.env_class = F1TenthTimeTrialEnv
        print("[INFO]: F1TenthTimeTrialRLEnvCfg.__post_init__ END")


# -----------------------------------------------------------------------------
# Runtime environment (buffers & per-step state)
# -----------------------------------------------------------------------------
class F1TenthTimeTrialEnv(ManagerBasedEnv):
    """Runtime env holding all buffers accessed by MDP terms.

    This class does not implement stepping itself; stepping is managed by Isaac
    Lab's manager stack. We mainly declare and size all tensors used by the MDP
    components (observations, rewards, terminations, etc.).
    """

    def __init__(self, cfg: F1TenthTimeTrialRLEnvCfg, **kwargs) -> None:
        super().__init__(cfg, **kwargs)

        # --- Indices & episode bookkeeping --------------------------------
        self._initial_waypoint_indices = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._prev_delta_s_opp_ego = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        # History sizes (from CONFIG) — centralized for consistency
        self._action_history_length: int = ENV_CFG["ACTION_HISTORY_LENGTH"]
        self._obs_history_length: int = ENV_CFG["OBS_HISTORY_LENGTH"]
        self._rew_history_length: int = ENV_CFG["REW_HISTORY_LENGTH"]
        self._progress_history_checkpoint_idx: int = ENV_CFG[
            "PROGRESS_HISTORY_CHECKPOINT_IDX"
        ]

        # Rolling indices and histories
        self._total_progress_indices = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._progress_history_indices = torch.zeros(
            (self.num_envs, self._rew_history_length), dtype=torch.long, device=self.device
        )

        # --- Action & state histories --------------------------------------
        self._action_history = torch.zeros(
            (self.num_envs, self._action_history_length, 2),
            dtype=torch.float32,
            device=self.device,
        )
        self._base_lin_vel_x_history = torch.zeros(
            (self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device
        )
        self._base_lin_vel_y_history = torch.zeros(
            (self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device
        )
        self._base_ang_vel_z_history = torch.zeros(
            (self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device
        )
        self._base_lin_acc_x_history = torch.zeros(
            (self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device
        )

        # Reward relevant histories
        self._traversability_history = torch.ones(
            (self.num_envs, self._rew_history_length), dtype=torch.long, device=self.device
        )
        self._wall_collision_history = torch.zeros(
            (self.num_envs, self._rew_history_length), dtype=torch.long, device=self.device
        )

        # Reset flag toggled on spawn, then cleared after first step
        self._reset_env_bool = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Progress and map assignment
        self._current_progress = torch.zeros(self.num_envs, device=self.device)
        self._map_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Targets & helper buffers used by observation terms
        self._last_velocity_adjustment = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._vel_y_calc = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._target_steering_angle = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._target_velocity = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._target_velocity_history = torch.zeros(
            (self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device
        )
        self._target_steering_angle_history = torch.zeros(
            (self.num_envs, self._obs_history_length), dtype=torch.float32, device=self.device
        )

        # Opponent (kept even in time-trial to make multi-car reuse easy)
        self._opponent_type = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._opponent_vel_scaling = torch.ones(self.num_envs, dtype=torch.float16, device=self.device)
        self._opponent_trajectory_alpha = torch.ones(self.num_envs, dtype=torch.float16, device=self.device)
        self._opponent_overtaken_bool = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._opponent_speed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._opponent_d_dot = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._opponent_heading = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._opponent_always_ahead = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._opponent_collision_history = torch.zeros(
            (self.num_envs, self._rew_history_length), dtype=torch.long, device=self.device
        )

        # --- Map geometry (copied from cfg for fast access on device) ------
        # These lists remain on CPU; individual terms decide what to move to GPU.
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
# Variants for domain-randomized training and play/testing
# -----------------------------------------------------------------------------
@configclass
class F1TenthTimeTrialRLRandomEnvCfg(F1TenthTimeTrialRLEnvCfg):
    """Same as `F1TenthTimeTrialRLEnvCfg` but with domain-randomization events."""

    events: F1TenthTimeTrialEventsRandomCfg = F1TenthTimeTrialEventsRandomCfg()


@configclass
class F1TenthTimeTrialPlayEnvCfg(F1TenthTimeTrialRLEnvCfg):
    """Inference-only environment (no rewards/terminations)."""

    # Resets (mirror training choice, but use the randomization subclass to keep hooks)
    if ENV_CFG["RESET_RANDOM"]:
        events: F1TenthTimeTrialEventsCfg = F1TenthTimeTrialEventsRandomCfg(
            reset_root_state_random=EventTerm(func=reset_root_state_random, mode="reset")
        )
    else:
        events: F1TenthTimeTrialEventsCfg = F1TenthTimeTrialEventsRandomCfg(
            reset_root_state_start_idx=EventTerm(func=reset_root_state_start_idx, mode="reset")
        )

    rewards: F1TenthTimeTrialRewardsCfg | None = None
    terminations: F1TenthTimeTrialTerminationsCfg | None = None

    def __post_init__(self) -> None:  # noqa: D401
        """Just call parent; defined for symmetry and future tweaks."""
        super().__post_init__()
