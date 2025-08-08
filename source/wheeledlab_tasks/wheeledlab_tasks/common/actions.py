from isaaclab.utils import configclass

from wheeledlab.envs.mdp import RCCar4WDActionCfg, RCCarRWDActionCfg
import yaml
with open("/home/tongo/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/timetrial/config/f1tenth_timetrial_config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)


@configclass
class F1Tenth4WDActionCfg:
    """Action configuration for F1Tenth 4WD, using RCCar4WDActionCfg with F1Tenth's joint names."""
    throttle_steer = RCCar4WDActionCfg(
        wheel_joint_names=[
            "wheel_back_left",
            "wheel_back_right",
            "wheel_front_left",
            "wheel_front_right",
        ],
        steering_joint_names=[
            "rotator_left",
            "rotator_right",
        ],
        base_length=0.365, # 32
        base_width=0.284, # half wheel 25, including full wheel 29
        wheel_radius=0.055, # wheelwidth = 0.045
        scale=(10.0, 0.40),
        no_reverse=True,
        bounding_strategy="clip",
        asset_name="robot",
    )

@configclass
class Mushr4WDActionCfg:

    throttle_steer = RCCar4WDActionCfg(
        wheel_joint_names=[
            "back_left_wheel_throttle",
            "back_right_wheel_throttle",
            "front_left_wheel_throttle",
            "front_right_wheel_throttle",
        ],
        steering_joint_names=[
            "front_left_wheel_steer",
            "front_right_wheel_steer",
        ],
        base_length=0.325,
        base_width=0.2,
        wheel_radius=0.05,
        scale=(10.0, 0.488),
        no_reverse=True,
        bounding_strategy="clip",
        asset_name="robot",
    )

@configclass
class LB4WDActionCfg:

    throttle_steer = RCCar4WDActionCfg(
        wheel_joint_names=[
            "Wheel__Knuckle__Front_Left",
            "Wheel__Knuckle__Front_Right",
            "Wheel__Upright__Rear_Right",
            "Wheel__Upright__Rear_Left"
        ],
        steering_joint_names=[
            "Knuckle__Upright__Front_Right",
            "Knuckle__Upright__Front_Left",
        ],
        base_length=0.34,
        base_width=0.28,
        wheel_radius=0.05,
        scale=(10.0, 0.488),
        no_reverse=True,
        bounding_strategy="clip",
        asset_name="robot",
    )

# # @configclass
# class MushrRWDActionCfg:

#     throttle_steer = RCCarRWDActionCfg(
#         wheel_joint_names=[
#             "back_left_wheel_throttle",
#             "back_right_wheel_throttle",
#         ],
#         steering_joint_names=[
#             "front_left_wheel_steer",
#             "front_right_wheel_steer",
#         ],
#         base_length=0.325,
#         base_width=0.2,
#         wheel_radius=0.05,
#         scale=(10.0, 0.488),
#         no_reverse=True,
#         bounding_strategy="Clip",
#         asset_name="robot",
#     )
