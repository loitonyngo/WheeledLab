import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg, DelayedImplicitActuatorCfg, DelayedPDActuatorCfg, DCMotorModCfg, DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg

from . import WHEELEDLAB_ASSETS_DATA_DIR

from wheeledlab_tasks.config_loader import load_config
CONFIG = load_config()

# F1Tenth 4WD actuator configuration.
# For 4WD, all throttle joints (front and back) are active.
F1TENTH_4WD_ACTUATOR_CFG = {
    "steering_joints": DelayedImplicitActuatorCfg(
        joint_names_expr=["rotator_(left|right)"],
        velocity_limit=0.005,    # F1Tenth steering is slightly slower than Hound
        effort_limit=1.0,
        stiffness=15.0,
        damping=0.05,
        friction=0.0,
        min_delay=CONFIG['env_config']['MIN_DELAY_STEERING'],  # Delays depends on physics step size! e.g. if physics step is 0.025s and max_delay is 2, then the delay is 0.2s.
        max_delay=CONFIG['env_config']['MAX_DELAY_STEERING']), #60-68 for dt= 0.025/4, 

    "throttle_joints": DCMotorModCfg(
        joint_names_expr=[".*wheel_(back|front)_.*"],  # Matches all throttle joints (e.g. wheel_back_left, wheel_front_right, etc.)
        saturation_effort=1,
        effort_limit=0.45, # Adjusted for the 3s VXL-3s motor/ESC
        velocity_limit=400.0,  # Reduced speed compared to a 4s system
        stiffness=0.0,
        damping=1100.0,
        friction=0.00,
        min_delay=CONFIG['env_config']['MIN_DELAY_THROTTLE'],  # Delays depends on physics step size! e.g. if physics step is 0.025s and max_delay is 2, then the delay is 0.2s.
        max_delay=CONFIG['env_config']['MAX_DELAY_THROTTLE'], # 8-12  for dt= 0.025/4, 
        low_velocity_threshold= 35.0,
        low_velocity_effort_limit= 0.13, 
    ),
}


# Initial state configuration for F1Tenth.
_ZERO_INIT_STATES = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "rotator_left": 0.0,
        "rotator_right": 0.0,
        "wheel_back_left": 0.0,
        "wheel_back_right": 0.0,
        "wheel_front_left": 0.0,
        "wheel_front_right": 0.0,
    },
)

# Overall configuration tying together the asset, physics, and initial state.
F1TENTH_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{WHEELEDLAB_ASSETS_DATA_DIR}/Robots/F1TENTH/f1tenth_red_mod.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            linear_damping=None,
            angular_damping=None,
            max_linear_velocity=1000.0,
            max_angular_velocity=100000.0,
            max_depenetration_velocity=100.0,
            max_contact_impulse=0.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=_ZERO_INIT_STATES,
    actuators=F1TENTH_4WD_ACTUATOR_CFG,
)

OPPONENT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{WHEELEDLAB_ASSETS_DATA_DIR}/Robots/F1TENTH/f1tenth_opp.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            linear_damping=None,
            angular_damping=None,
            max_linear_velocity=1000.0,
            max_angular_velocity=100000.0,
            max_depenetration_velocity=100.0,
            max_contact_impulse=0.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=_ZERO_INIT_STATES,
    actuators=F1TENTH_4WD_ACTUATOR_CFG,
)

LB_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{WHEELEDLAB_ASSETS_DATA_DIR}/Robots/F1TENTH/f1tenth_alt.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
        joint_pos={
            "Wheel__Knuckle__Front_Left": 0.0,
            "Wheel__Knuckle__Front_Right": 0.0,
            "Wheel__Upright__Rear_Right": 0.0,
            "Wheel__Upright__Rear_Left": 0.0,
            "Knuckle__Upright__Front_Right": 0.0,
            "Knuckle__Upright__Front_Left": 0.0,
        },
    ),
    actuators={
        "throttle": ImplicitActuatorCfg(
            joint_names_expr=["Wheel.*"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=0.0,
            # damping=100000.0,
            damping=100.0,
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["Knuckle__Upright__Front.*"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=1000.0,
            damping=0.0,
        ),
    },
)