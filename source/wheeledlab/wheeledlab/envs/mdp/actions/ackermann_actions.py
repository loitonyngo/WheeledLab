# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg
    
from wheeledlab_tasks.config_loader import load_config
CONFIG = load_config()
    
class AckermannAction(ActionTerm):
    r"""
    Action term for controlling Ackermann steering vehicles.

    This class encapsulates the logic required to process and apply actions for an Ackermann steering vehicle
    in a simulation environment. It handles the transformation of raw actions, including clipping or applying
    a tanh function to bound the actions within a specified range, and computes the appropriate steering angles
    and wheel velocities.

    Attributes:
    ----------
    cfg : actions_cfg.AckermannActionCfg
        The configuration of the action term, including parameters like scaling, offset, and bounding strategy.
    _asset : Articulation
        The articulation asset on which the action term is applied.
    _scale : torch.Tensor
        The scaling factor applied to the input action. Shape is (1, 2).
    _offset : torch.Tensor
        The offset applied to the input action. Shape is (1, 2).
    _bounding_strategy : str | None
        The strategy used to bound the actions. Can be 'clip' or 'tanh'. If None, no bounding is applied.
    _raw_actions : torch.Tensor
        Tensor to store raw actions before processing.
    _processed_actions : torch.Tensor
        Tensor to store processed actions after applying the bounding strategy, scaling, and offset.
    base_length : torch.Tensor
        The length of the vehicle base.
    base_width : torch.Tensor
        The width of the vehicle base.
    wheel_rad : torch.Tensor
        The radius of the vehicle wheels.

    Methods:
    -------
    process_actions(actions):
        Processes the raw actions based on the bounding strategy, scaling, and offset.

    apply_actions():
        Applies the processed actions to the articulation asset.

    calculate_ackermann_angles_and_velocities(target_steering_angle_rad, target_velocity):
        Calculates the steering angles for the left and right front wheels and the wheel velocities based on the
        Ackermann steering geometry.
    """

    cfg: actions_cfg.AckermannActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, 2)."""
    _offset: torch.Tensor
    """The offset applied to the input action. Shape is (1, 2)."""
    _bounding_strategy: str | None
    """The strategy used to bound the actions. Can be 'clip' or 'tanh'. If None, no bounding is applied."""

    def __init__(self, cfg: actions_cfg.AckermannActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        wheel_ids, wheel_names = self._asset.find_joints(cfg.wheel_joint_names)
        self._wheel_ids = wheel_ids
        self._wheel_names = wheel_names

        steering_ids, steering_names = self._asset.find_joints(cfg.steering_joint_names)
        self._steering_ids = steering_ids
        self._steering_names = steering_names

        # Action scaling and offset
        self._scale = torch.tensor(cfg.scale, device=self.device, dtype=torch.float32)
        self._offset = torch.tensor(cfg.offset, device=self.device, dtype=torch.float32)
        self._bounding_strategy = cfg.bounding_strategy

        # Initialize tensors for actions
        self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=self.device)  # Placeholder for [velocity, steering_angle]

        self.base_length = torch.tensor(cfg.base_length, device=self.device)
        self.base_width = torch.tensor(cfg.base_width, device=self.device)
        self.wheel_rad = torch.tensor(cfg.wheel_radius, device=self.device)


    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        # store the raw actions
        self._raw_actions[:] = actions

        if self._bounding_strategy == 'clip':
            # useful for testing
            self._processed_actions = torch.clip(actions, min=-1.0, max=1.0) * self._scale + self._offset
        
        elif self._bounding_strategy == 'tanh':
            self._processed_actions = torch.tanh(actions) * self._scale + self._offset

        else:
            self._processed_actions = actions * self._scale + self._offset

        if self.cfg.no_reverse:
            self._processed_actions[:, 0] = torch.clamp(self._processed_actions[:, 0], min=0.0)


    def apply_actions(self):

        # vel mode
        left_rotator_angle, right_rotator_angle, wheel_speeds = self._calculate_ackermann_angles_and_velocities(
            target_velocity= self.processed_actions[:, 0]/CONFIG['env_config']['SPEED_SIM_TO_REAL_SCALING'], # Velocity for all cars
            target_steering_angle=self.processed_actions[:, 1]/CONFIG['env_config']['STEERING_SIM_TO_REAL_SCALING'] # Steering angle for all cars
        )
        
        front_wheel_angles = torch.stack([left_rotator_angle, right_rotator_angle], dim=1)

        self._asset.set_joint_velocity_target(wheel_speeds, joint_ids=self._wheel_ids)
        self._asset.set_joint_position_target(front_wheel_angles, joint_ids=self._steering_ids)




    def _calculate_ackermann_angles_and_velocities(self, target_steering_angle, target_velocity):
        """
        Calculates the steering angles for the left and right front wheels and the wheel velocities based on the
        Ackermann steering geometry.

        Parameters:
        ----------
        target_steering_angle : torch.Tensor
            Target steering angles in radians for each environment.
        target_velocity : torch.Tensor
            Target velocities for each environment in meters/second.

        Returns:
        -------
        delta_left : torch.Tensor
            Steering angles for the left front wheels.
        delta_right : torch.Tensor
            Steering angles for the right front wheels.
        wheel_speeds : torch.Tensor
            Speeds for each wheel.
        """
        L = self.base_length
        W = self.base_width
        wheel_radius = self.wheel_rad

        # Ensure inputs are PyTorch tensors
        target_steering_angle = target_steering_angle.float()
        target_velocity = target_velocity.float()

        # Calculating the turn radius from the steering angle
        tan_steering = torch.tan(target_steering_angle)
        R = torch.where(tan_steering == 0, torch.full_like(tan_steering, 1e6), L / tan_steering)

        # Calculate the steering angles for the left and right front wheels in radians
        delta_left = torch.atan(L / (R - W / 2))
        delta_right = torch.atan(L / (R + W / 2))

        # Assuming the rear wheels follow the path's radius adjusted for their position
        R_rear_left = torch.sqrt((R - W/2)**2 + L**2)
        R_rear_right = torch.sqrt((R + W/2)**2 + L**2)

        # Velocity adjustment based on wheel's distance from the IC
        v_front_left = target_velocity * torch.abs(R_rear_left / (R*wheel_radius))
        v_front_right = target_velocity * torch.abs(R_rear_right / (R*wheel_radius))

        v_back_left = target_velocity * torch.abs((R - W/2) / (R*wheel_radius))
        v_back_right = target_velocity * torch.abs((R + W/2) / (R*wheel_radius))

        # Calculate target rotation for each wheel based on its velocity
        wheel_speeds = torch.stack([v_back_left, v_back_right, v_front_left, v_front_right], dim=1)

        return delta_left, delta_right, wheel_speeds
    

class AckermannIncrementalAction(ActionTerm):
    r"""
    Incremental-action term for Ackermann steering vehicles.

    Unlike absolute commands (speed, steering) this class expects the policy to
    output **increments** (Δspeed, Δsteer) at every step:

        action[:, 0]  ->  Δspeed   (m/s per step, after scaling)
        action[:, 1]  ->  Δsteer   (rad per step, after scaling)

    These increments are integrated into **targets** using the most recent
    targets stored in the environment histories:
        - env._target_velocity_history[:, 0]         (latest target speed)
        - env._target_steering_angle_history[:, 0]   (latest target steer)

    Key processing steps
    --------------------
    1) `process_actions`:
       - Clamp raw actions to [-1, 1] (for safety and consistency).
       - Scale them to physical increments using
         CONFIG['env_config']['MAX_SPEED_INCREMENT'] and
         CONFIG['env_config']['MAX_STEERING_ANGLE_INCREMENT'].
       - No offset is used here because the actions represent *deltas*.

    2) `apply_actions`:
       - Update target speed and target steer by adding the increments to the
         latest values from the env histories.
       - Apply optional steering offset (CONFIG['env_config']['STEERING_OFFSET'])
         to bias steering if desired.
       - Clamp the target speed to be non-negative (min 0.5 m/s here).
       - Clamp the target steering angle within
         [-MAX_STEERING_ANGLE, +MAX_STEERING_ANGLE].
       - Convert targets from "sim" units to "real" units via
         SPEED_SIM_TO_REAL_SCALING and STEERING_SIM_TO_REAL_SCALING.
       - Compute wheel steering angles and wheel speeds with Ackermann geometry.
       - Send commands to the articulation joints:
            * set_joint_velocity_target for wheel speeds
            * set_joint_position_target for steering angles (front left/right)

    Assumptions / Dependencies
    --------------------------
    - The environment maintains ring buffers:
         * env._target_velocity_history      [num_envs, obs_hist_len]
         * env._target_steering_angle_history[num_envs, obs_hist_len]
      The class *reads* index [:, 0] (the most recent entry) and *does not*
      modify the history itself (other parts of the code handle history shifts).

    - The articulation has:
         * wheel joints listed in cfg.wheel_joint_names
         * steering joints listed in cfg.steering_joint_names

    - All geometry constants (base length/width, wheel radius) are provided
      by the action config, and sim↔real scaling factors by CONFIG.

    Attributes
    ----------
    cfg : actions_cfg.AckermannActionCfg
        Action configuration (joint names, geometry, per-axis scaling config).
    _asset : Articulation
        The controlled vehicle articulation.
    _scale, _offset, _bounding_strategy
        (Not used in the incremental math here—kept for symmetry with other
         action terms. Bounds are enforced explicitly in `process_actions`.)
    _raw_actions : torch.Tensor [num_envs, 2]
        Buffer for raw incoming actions.
    _processed_actions : torch.Tensor [num_envs, 2]
        Clamped and scaled increments (Δspeed, Δsteer).
    base_length, base_width, wheel_rad : torch.Tensor
        Vehicle geometry used by the Ackermann model.
    _target_velocity, _target_steering_angle : torch.Tensor [num_envs]
        Per-env targets computed from the last targets + increments.

    Notes
    -----
    - The incremental mode stabilizes training/control by preventing sudden
      jumps in absolute targets. The policy learns to "nudge" speed/steer.
    - You can control aggressiveness via the MAX_*_INCREMENT config values.
    - Steering offset can be used to compensate biases in the vehicle model.
    """

    cfg: actions_cfg.AckermannActionCfg
    _asset: Articulation
    _scale: torch.Tensor
    _offset: torch.Tensor
    _bounding_strategy: str | None

    def __init__(self, cfg: actions_cfg.AckermannActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Cache joint IDs (order matters: match your robot's joint ordering)
        wheel_ids, wheel_names = self._asset.find_joints(cfg.wheel_joint_names)
        self._wheel_ids = wheel_ids
        self._wheel_names = wheel_names

        steering_ids, steering_names = self._asset.find_joints(cfg.steering_joint_names)
        self._steering_ids = steering_ids
        self._steering_names = steering_names

        # Generic action meta (kept for interface consistency with other terms)
        self._scale = torch.tensor(cfg.scale, device=self.device, dtype=torch.float32)
        self._offset = torch.tensor(cfg.offset, device=self.device, dtype=torch.float32)
        self._bounding_strategy = cfg.bounding_strategy

        # Runtime buffers
        self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=self.device)
        # Will be filled in process_actions
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # Vehicle geometry
        self.base_length = torch.tensor(cfg.base_length, device=self.device)
        self.base_width = torch.tensor(cfg.base_width, device=self.device)
        self.wheel_rad = torch.tensor(cfg.wheel_radius, device=self.device)

        # Per-env target buffers (derived from history + increments)
        self._target_steering_angle = torch.zeros(env.num_envs, device=self.device, dtype=torch.float32)
        self._target_velocity       = torch.zeros(env.num_envs, device=self.device, dtype=torch.float32)

    # -------------------------
    # Properties
    # -------------------------

    @property
    def action_dim(self) -> int:
        """Two DOFs: Δspeed, Δsteer."""
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        """Unprocessed actions from the policy/network."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Clamped/scaled action increments (Δspeed, Δsteer)."""
        return self._processed_actions

    # -------------------------
    # Pipeline
    # -------------------------

    def process_actions(self, actions: torch.Tensor) -> None:
        """
        Clamp and scale raw actions into physical increments.

        Inputs are expected in [-∞, +∞]; we clip to [-1, 1] and then scale:

            Δspeed  = clip(a0) * MAX_SPEED_INCREMENT
            Δsteer  = clip(a1) * MAX_STEERING_ANGLE_INCREMENT
        """
        # Cache raw for logging/inspection
        self._raw_actions[:] = actions

        # Clamp to a safe range and scale to physical increments
        increments = torch.clip(actions, min=-1.0, max=1.0)
        inc_scales = torch.tensor(
            [
                CONFIG['env_config']['MAX_SPEED_INCREMENT'],
                CONFIG['env_config']['MAX_STEERING_ANGLE_INCREMENT'],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self._processed_actions = increments * inc_scales

    def apply_actions(self) -> None:
        """
        Integrate increments into targets, apply clamping, convert units, and send joint commands.

        Steps:
        - Read latest targets from env histories (index 0 is "most recent").
        - Add increments to build new targets.
        - Clamp targets to valid ranges.
        - Apply sim→real unit scaling.
        - Compute Ackermann angles & per-wheel speeds.
        - Issue joint position/velocity targets to the articulation.
        """
        # 1) Integrate Δspeed → new speed target (keep speed >= 0.5 m/s)
        #    Using history maintained by the env elsewhere in the code base.
        self._target_velocity = torch.clamp(
            self._env._target_velocity_history[:, 0] + self.processed_actions[:, 0],
            min=0.5,
        )

        # 2) Integrate Δsteer → new steering target (with optional bias/offset)
        steer_offset = CONFIG['env_config']['STEERING_OFFSET']
        max_steer = CONFIG['env_config']['MAX_STEERING_ANGLE']
        self._target_steering_angle = torch.clamp(
            self._env._target_steering_angle_history[:, 0] + self.processed_actions[:, 1] + steer_offset,
            min=-max_steer,
            max=+max_steer,
        )

        # 3) Convert "sim" targets to "real" using provided scaling factors
        v_scale = CONFIG['env_config']['SPEED_SIM_TO_REAL_SCALING']
        s_scale = CONFIG['env_config']['STEERING_SIM_TO_REAL_SCALING']

        # 4) Ackermann geometry → front steering angles + per-wheel speeds
        left_rotator_angle, right_rotator_angle, wheel_speeds = self._calculate_ackermann_angles_and_velocities(
            target_velocity=self._target_velocity / v_scale,
            target_steering_angle=self._target_steering_angle / s_scale,
        )
        front_wheel_angles = torch.stack([left_rotator_angle, right_rotator_angle], dim=1)

        # 5) Send commands to the robot
        self._asset.set_joint_velocity_target(wheel_speeds, joint_ids=self._wheel_ids)
        self._asset.set_joint_position_target(front_wheel_angles, joint_ids=self._steering_ids)

    # -------------------------
    # Ackermann geometry
    # -------------------------

    def _calculate_ackermann_angles_and_velocities(
        self,
        target_steering_angle: torch.Tensor,
        target_velocity: torch.Tensor,
    ):
        """
        Convert desired (vehicle) steering & speed into per-wheel commands.

        Ackermann model summary:
        - The vehicle turns around an instantaneous center (IC) at radius R.
        - Front-left and front-right have different steering angles so their
          wheel axes intersect at the IC.
        - Wheel linear speeds scale with their distance to the IC; here we
          output wheel angular velocities (rad/s) by dividing by wheel radius.

        Parameters
        ----------
        target_steering_angle : torch.Tensor [num_envs]
            Desired steering angle (radians) of the vehicle.
        target_velocity : torch.Tensor [num_envs]
            Desired vehicle forward speed (m/s).

        Returns
        -------
        delta_left  : torch.Tensor [num_envs]
            Front-left steering angle (radians).
        delta_right : torch.Tensor [num_envs]
            Front-right steering angle (radians).
        wheel_speeds : torch.Tensor [num_envs, 4]
            Angular wheel speeds (back_left, back_right, front_left, front_right).
        """
        L = self.base_length
        W = self.base_width
        wheel_radius = self.wheel_rad

        target_steering_angle = target_steering_angle.float()
        target_velocity = target_velocity.float()

        # Turn radius from vehicle steering angle
        tan_steering = torch.tan(target_steering_angle)
        # Avoid division by zero: use a very large radius for straight driving
        R = torch.where(tan_steering == 0, torch.full_like(tan_steering, 1e6), L / tan_steering)

        # Front steering angles so that wheel axes meet at the IC
        delta_left = torch.atan(L / (R - W / 2))
        delta_right = torch.atan(L / (R + W / 2))

        # Distances from IC to rear-left / rear-right wheel contact points
        R_rear_left = torch.sqrt((R - W / 2) ** 2 + L ** 2)
        R_rear_right = torch.sqrt((R + W / 2) ** 2 + L ** 2)

        # Convert vehicle speed to wheel angular speeds
        # (linear speed at each wheel divided by wheel radius)
        v_front_left = target_velocity * torch.abs(R_rear_left / (R * wheel_radius))
        v_front_right = target_velocity * torch.abs(R_rear_right / (R * wheel_radius))
        v_back_left = target_velocity * torch.abs((R - W / 2) / (R * wheel_radius))
        v_back_right = target_velocity * torch.abs((R + W / 2) / (R * wheel_radius))

        wheel_speeds = torch.stack([v_back_left, v_back_right, v_front_left, v_front_right], dim=1)
        
        return delta_left, delta_right, wheel_speeds
