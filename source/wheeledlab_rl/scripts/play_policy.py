"""
Play a policy in an environment and record the data.

Usage:

python play_policy.py -p <path-to-run> -sd --video

This command will save data and record a video of the playback using an existing run folder.

"""

###################################
###### BEGIN ISAACLAB SPINUP ######
###################################


from wheeledlab_rl.startup import startup
import argparse
from datetime import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser(description="Play a policy in WheeledLab.")

###################################
###### DEFINE POLICY TO PLAY ######
###################################
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]  
# if this file is .../wheeledlab_rl/config/paths.py, this goes 2 levels up to "wheeledlab_rl"

DEFAULT_LOGS_PATH = PROJECT_ROOT / "wheeledlab_rl" / "logs"
SAVE_DIR = PROJECT_ROOT / "wheeledlab_rl" / "output_play_policy"

POLICY = "CIR1_TT_20hz_vel12steer15fric75_nhor20ds10_del2505_buffer5_wall20_pendeltasteeringvariation"
SAVE_NAME = "test_1"
TIMESTAMP = datetime.now().strftime("%m%d_%H%M")

###################################
###################################
###################################f


parser.add_argument('-p', "--run-path", type=str, 
                   default=DEFAULT_LOGS_PATH/POLICY, 
                   help="Path to run folder")

parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint to load")
# If no run folder, the task and policy model must be provided
parser.add_argument("--task", type=str, default=None, help="Task name. Overrides run config env if provided")
parser.add_argument("--policy-path", type=str, default=None, help="Path to policy file.")

# Playback
parser.add_argument("--steps", type=int, default=250, help="Length of recorded video in steps")
# Logging
parser.add_argument('-sd', "--save-data", action="store_true", default=True, help="Save episode data")
parser.add_argument("--save-name", type=str, default=SAVE_NAME, help="Name save file.")

parser.add_argument("--video", action="store_true", help="Record video of the playback")
parser.add_argument("--log-dir", type=str, default="playback/",
                    help="Directory to save logs. If run path is provided, this is ignored.")
parser.add_argument("--play-name", type=str, default="play-name", help="Name of the playback")

simulation_app, args_cli = startup(parser=parser)
### Extract task_name and agent_cfg from run_config.pkl ###

# Validate arguments
if args_cli.run_path is None:
    if args_cli.task is None and args_cli.policy_path is None:
        raise ValueError("Either path to run directory or task/policy must be provided.")

import gymnasium as gym
import time
import torch
from tqdm import tqdm
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import load_pickle
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from wheeledlab_rl.configs import RunConfig
from wheeledlab_rl.utils import ClipAction

import matplotlib.pyplot as plt
import numpy as np

# Resolve paths
FROM_RUN = args_cli.run_path is not None
if FROM_RUN: # Load paths for run folder

    # Load run config
    path_to_run_cfg_pkl = os.path.join(args_cli.run_path, "run_config.pkl")
    run_cfg: RunConfig = load_pickle(path_to_run_cfg_pkl) # load_yaml does not work on slices
    run_agent_cfg = run_cfg.agent
    task = run_cfg.env_setup.task_name if args_cli.task is None else args_cli.task
    agent_entry_point = None

    # Get policy path
    chkpt = args_cli.checkpoint if args_cli.checkpoint is not None else ".*"
    fp = os.path.abspath(args_cli.run_path)
    run_dirname = os.path.dirname(fp)
    run_folder = os.path.basename(fp)
    policy_resume_path = get_checkpoint_path(log_path=run_dirname, run_dir=run_folder,
                                        other_dirs=["models"], checkpoint=chkpt)

    # Set playback directory to be in run folder
    playback_dir = os.path.join(args_cli.run_path, "playback")

else:

    task = args_cli.task
    agent_entry_point = "rsl_rl_cfg_entry_point" # rsl is the only supported library for now
    playback_dir = args_cli.log_dir
    policy_resume_path = args_cli.policy_path


@hydra_task_config(task, agent_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg): # TODO: Add SB3 config support

    if agent_cfg is None:
        agent_cfg = run_agent_cfg

    if not os.path.exists(playback_dir):
        os.makedirs(playback_dir)
    print(f"[INFO] Created playback directory: {playback_dir}")

    ####################################
    #### POLICY LOADING CODE ####
    ####################################

    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": playback_dir,
            "step_trigger": lambda step: step % args_cli.steps == 0,
            "video_length": args_cli.steps, # updated to use args_cli
            "disable_logger": True,
            "name_prefix": args_cli.play_name,
        }
        print(f"[INFO] Recording video of playback to: {playback_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)


    ############################################
    ########### BEGIN PLAYBACK SETUP ###########
    ############################################

    env.action_space.low = -1.
    env.action_space.high = 1.
    env = ClipAction(env) 
    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict())
    ppo_runner.load(policy_resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Data storage
    data = {
        'observations': [],
        'rewards': [],
        'actions': [],
        'pos_xy': [],
        'vel_x': [],
        'vel_y': [],
        'yaw_rate': [],
        'target_velocity': [],
        'target_steering': [],
        'theta': [],
        's_idx': [],
        'time': [],
        's_idx_max': [],
        'delta_s_opp_ego': []
    }

    ### PLAY POLICY ###

    # reset environment
    obs, _ = env.get_observations()

    from wheeledlab_tasks.config_loader import load_config
    CONFIG = load_config()
    
    # simulate environment
    for time_idx in tqdm(range(args_cli.steps), desc="Playing policy"):
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            obs, rew, _, extras = env.step(actions)
        # save data
        data['observations'].append(obs)
        data['rewards'].append(rew)
        data['actions'].append(actions)
        
        fields = [
            'pos_xy', 'theta',
            'vel_x', 'vel_y', 'yaw_rate',
            'target_velocity', 'target_steering',
            's_idx', 's_idx_max',
            'time',
        ]
        
        for field in fields:
            value = extras.get(field)
            if value is not None:
                data[field].append(value)
            else:
                print(f'WARNING: could not store {field}')
    ###


    ########################
    ###### SAVE DATA #######
    ########################

    if args_cli.save_data:
        # Create a dedicated folder for this episode’s data
        save_folder = os.path.join(SAVE_DIR, SAVE_NAME)
        os.makedirs(save_folder, exist_ok=True)  # creates folder if it doesn't exist

        # Base filename without extension
        base_name = os.path.join(save_folder, f"{args_cli.save_name}")
        
        # Check if file exists and find appropriate suffix
        suffix = ""
        counter = 0
        while True:
            save_pt_path = f"{base_name}{suffix}.pt"
            save_csv_path = f"{base_name}{suffix}.csv"
            if not os.path.exists(save_pt_path) and not os.path.exists(save_csv_path):
                break
            suffix = f"_{counter}"
            counter += 1

        # --- Save as .pt ---
        for key in data.keys():
            try:
                data[key] = torch.stack(data[key], dim=0)
            except Exception as e:
                print(f'WARNING: could not torch.stack {key} ({e})')
                continue

        torch.save(data, save_pt_path)
        print(f"[INFO] Saved episode data to: {save_pt_path}")

        # --- Save selected fields as .csv ---
        selected_fields = ["target_velocity", 'target_steering', "pos_xy", "theta", "vel_x", "vel_y", "yaw_rate", "s_idx", "time"]

        df_dict = {}
        lengths = []

        for key in selected_fields:
            if key not in data:
                print(f"WARNING: {key} not found in data")
                continue

            tensor = data[key].cpu().squeeze()  # remove singleton dimensions

            if key == "pos_xy":
                df_dict["x"] = tensor[:, 0].numpy().astype(float)
                df_dict["y"] = tensor[:, 1].numpy().astype(float)
                lengths.append(tensor.shape[0])
            else:
                arr = tensor.numpy().astype(float)
                df_dict[key] = arr
                lengths.append(arr.shape[0])

        # Align all arrays to the same length
        min_len = min(lengths)
        for k in df_dict:
            df_dict[k] = df_dict[k][:min_len]

        # Column order and renaming
        col_order = ["time", "x", "y", "target_velocity", "target_steering", "theta", "vel_x", "vel_y", "yaw_rate", "s_idx"]
        df = pd.DataFrame(df_dict)[col_order]
        df.columns = ["time", "x", "y", "speed_cmd", "steering_cmd", "theta_rad", "vx_mps", "vy_mps", "psi_radps", "s_idx"]

        df.to_csv(save_csv_path, index=False)
        print(f"[INFO] Saved selected episode data to: {save_csv_path}")


    print("Done playing policy. Closing environment.")
    env.close()

##############################################################
######################### PLOT DATA #########################
##############################################################


   # --------------------------
    # Load the saved CSV data
    # --------------------------
    sim_data = pd.read_csv(save_csv_path)

    # --------------------------
    # Convert sim CSV columns to numpy arrays
    # --------------------------
    actions      = sim_data.filter(like="actions").to_numpy()       # e.g., actions_0, actions_1...
    pos_xy       = sim_data[['x', 'y']].to_numpy()
    vel_x        = sim_data['vx_mps'].to_numpy()
    vel_y        = sim_data['vy_mps'].to_numpy()
    yaw_rate     = sim_data['psi_radps'].to_numpy()
    s_idx        = sim_data['s_idx'].to_numpy()
    theta        = sim_data['theta_rad'].to_numpy()
    time         = sim_data['time'].to_numpy()
    speed_cmd    = sim_data['speed_cmd'].to_numpy()
    steering_cmd = sim_data['steering_cmd'].to_numpy()


    # --------------------------
    # Compute derived quantities
    # --------------------------
    vel = np.sqrt(vel_x**2 + vel_y**2)
    acceleration = np.gradient(vel, time)
    jerk = np.gradient(acceleration, time)

    # --------------------------
    # Derived quantities
    # --------------------------
    vel = np.sqrt(vel_x**2 + vel_y**2)
    acceleration = np.gradient(vel, time)
    jerk = np.gradient(acceleration, time)


    # --------------------------
    # Plotting
    # --------------------------
    plt.figure(figsize=(15, 15))

    # ---- Subplot 1: Command Velocity and Steering ----
    ax1 = plt.subplot(4,1,1)
    ax1b = ax1.twinx()

    ax1.plot(time, speed_cmd,
            color='green', label='Model cmd velocity')

    ax1b.plot(time, steering_cmd,
            color='blue', label='Model cmd steering')


    ax1.set_ylabel('Velocity', color='green')
    ax1b.set_ylabel('Steering Angle', color='blue')
    ax1.tick_params(axis='y', colors='green')
    ax1b.tick_params(axis='y', colors='blue')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1b, labels1b = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines1b, labels1 + labels1b, loc='upper right')
    ax1.set_title(f"{POLICY}: Command Velocity and Steering")
    ax1.grid(True)

    # ---- Subplot 2: Linear Velocity, Acceleration, Jerk ----
    ax2 = plt.subplot(4,1,2, sharex=ax1)
    ax2b = ax2.twinx()
    ax2c = ax2.twinx()
    ax2c.spines['right'].set_position(('outward', 60))

    ax2.plot(time, vel_x, color='g', label='Lin Vel X (sim)')

    ax2.plot(time, -vel_y, color='b', label='Lin Vel Y (sim)')

    # real_jerk = np.gradient(real_data['ax'].values[real_mask], time)

    # ax2b.plot(time, acceleration, color='r', label='Acceleration (sim)')
    # ax2b.plot(time, real_data['ax'].values[real_mask], color='r', linestyle='--', label='Acceleration (real)')

    # ax2c.plot(time, jerk, color='purple', label='Jerk (sim)')

    ax2.set_ylabel("velocity [m/s]")
    ax2b.set_ylabel("acceleration [m/s²]", color='r')
    ax2b.tick_params(axis='y', labelcolor='r')
    ax2c.set_ylabel("jerk [m/s³]", color='purple')
    ax2c.tick_params(axis='y', labelcolor='purple')

    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2b, labels2b = ax2b.get_legend_handles_labels()
    lines2c, labels2c = ax2c.get_legend_handles_labels()
    ax2.legend(lines2 + lines2b + lines2c, labels2 + labels2b + labels2c, loc='upper right')

    ax2.set_title("Linear Velocity, Acceleration, Jerk")
    ax2.grid(True)

    # ---- Subplot 3: Angular Velocity and Heading ----
    ax3 = plt.subplot(4,1,3, sharex=ax1)
    ax3b = ax3.twinx()

    ax3.plot(time, yaw_rate, color='g', label='Sim Ang Vel Z')
    ax3b.plot(time, theta, color='r', label='Sim Theta')

    ax3.set_ylabel("ang velocity [rad/s]")
    ax3b.set_ylabel("heading angle [rad]", color='r')
    ax3b.tick_params(axis='y', labelcolor='r')

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines3b, labels3b = ax3b.get_legend_handles_labels()
    ax3.legend(lines3 + lines3b, labels3 + labels3b, loc='upper right')
    ax3.set_title("Angular Velocity and Heading Angle")
    ax3.grid(True)



    # --------------------------
    # Plot trajectories
    # --------------------------
    plt.figure(figsize=(10, 8))
    plt.plot(pos_xy[:,0], pos_xy[:,1], label='Simulated Trajectory', color='blue', linewidth=2)

    plt.xlabel("X Position [m]")
    plt.ylabel("Y Position [m]")
    plt.title("Trajectory Comparison: Simulated vs Real")
    plt.legend()
    plt.axis('equal')  # keep aspect ratio correct
    plt.grid(True)
    plt.show()


def resample_time_series(original_time, original_values, new_time):
    """
    Resample time series data using linear interpolation.
    
    Args:
        original_time: 1D tensor of original timestamps
        original_values: 1D tensor of original values
        new_time: 1D tensor of new timestamps for resampling
        
    Returns:
        Resampled values at new_time points
    """
    # Ensure the tensors are on the same device
    device = original_time.device
    original_values = original_values.to(device)
    new_time = new_time.to(device)
    
    # Perform linear interpolation
    # Find indices where new_time would be inserted to maintain order in original_time
    indices = torch.searchsorted(original_time, new_time)
    
    # Handle boundary cases
    indices = torch.clamp(indices, 1, len(original_time) - 1)
    
    # Get the surrounding values and times
    t0 = original_time[indices - 1]
    t1 = original_time[indices]
    v0 = original_values[indices - 1]
    v1 = original_values[indices]
    
    # Compute interpolation weights
    alpha = (new_time - t0) / (t1 - t0)
    
    # Interpolate
    resampled_values = v0 + alpha * (v1 - v0)
    
    return resampled_values

def resample_timeseries(original_time, original_values, new_time):
    """
    Resample timeseries data to new time points using linear interpolation
    Args:
        original_time: 1D tensor of original timestamps
        original_values: 1D tensor of original values
        new_time: 1D tensor of desired timestamps
    Returns:
        Resampled values at new_time points
    """
    # Ensure we have valid data
    if len(original_time) == 0 or len(original_values) == 0:
        return torch.zeros_like(new_time)
    
    # Perform linear interpolation
    resampled = torch.zeros_like(new_time)
    for i, t in enumerate(new_time):
        # Find where this time would fit in the original data
        idx = torch.searchsorted(original_time, t)
        
        if idx == 0:
            # Before first point - extrapolate
            resampled[i] = original_values[0]
        elif idx == len(original_time):
            # After last point - extrapolate
            resampled[i] = original_values[-1]
        else:
            # Linear interpolation between idx-1 and idx
            t0, t1 = original_time[idx-1], original_time[idx]
            v0, v1 = original_values[idx-1], original_values[idx]
            alpha = (t - t0) / (t1 - t0)
            resampled[i] = v0 + alpha * (v1 - v0)
    
    return resampled

if __name__ == "__main__":
    main()



        # Plot actions

    # plt.figure(figsize=(12, 4))
    # for env_idx in range(actions.shape[1]):  # Loop through environments
    #     plt.plot(time[start_idx:end_idx], actions[start_idx:end_idx, env_idx, 0], label=f'Env {env_idx} (Throttle)')
    #     plt.plot(time[start_idx:end_idx], actions[start_idx:end_idx, env_idx, 1], '--', label=f'Env {env_idx} (Steering)')
    #     # plt.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 5], '--', label=f'Env {env_idx} (ang vel y)')
    # plt.xlabel("time [s]")
    # plt.ylabel("Action Value")
    # plt.title(POLICY+": Policy Actions Over Time")
    # plt.legend()
    # plt.grid()
    # plt.show()
    # vel = np.sqrt(np.square(observations[:, 0, 0]) + np.square(observations[:, 0, 1]))


    # plt.figure(figsize=(15, 10))

    # Create subplots (2 rows, 1 column)
    # ax1 = plt.subplot(2, 1, 1)  # Velocity plot
    # ax2 = plt.subplot(2, 1, 2)  # Slip ratio plot

    # Calculate metrics
    # wheel_ang_vel_mean = np.mean(observations[:, 0, 5:9], axis=1)
    # wheel_lin_vel_mean = np.mean(observations[:, 0, 9:13], axis=1)
    # slip_ratio = (wheel_ang_vel_mean*0.06/wheel_lin_vel_mean-1)

    # Plot 1: Velocities
    # for env_idx in range(actions.shape[1]):
        # Angular velocities (converted to linear by multiplying with radius)
        # ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 5]*0.06, '--', color='red', alpha=0.5, label='BL ang_vel×r' if env_idx==0 else "")
        # ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 6]*0.06, '--', color='orange', alpha=0.5, label='BR ang_vel×r' if env_idx==0 else "")
        # ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 7]*0.06, '--', color='blue', alpha=0.5, label='FL ang_vel×r' if env_idx==0 else "")
        # ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 8]*0.06, '--', color='cyan', alpha=0.5, label='FR ang_vel×r' if env_idx==0 else "")
        
    #     # Linear velocities
    #     ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 9], color='red', alpha=0.5, label='BL lin_vel' if env_idx==0 else "")
    #     ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 10], color='orange', alpha=0.5, label='BR lin_vel' if env_idx==0 else "")
    #     ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 11], color='blue', alpha=0.5, label='FL lin_vel' if env_idx==0 else "")
    #     ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 12], color='cyan', alpha=0.5, label='FR lin_vel' if env_idx==0 else "")

    # # Plot mean values

    # ax1.plot(time[start_idx:end_idx], wheel_ang_vel_mean[start_idx:end_idx]*0.06, '--', color='black', label='Mean ang_vel×r')
    # ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 13]*0.06, '--', color='red', marker='x', label='Mean ang_speed×r')

    # ax1.plot(time[start_idx:end_idx], wheel_lin_vel_mean[start_idx:end_idx], color='blue', marker='x', label='Mean lin_vel')
    # ax1.plot(time[start_idx:end_idx], vel[start_idx:end_idx], color='purple', label='Base speed')

    # ax1.set_ylabel('Velocity (m/s)')
    # ax1.set_title('Wheel Velocities')
    # ax1.legend()
    # ax1.grid(True)

    # Plot 2: Slip Ratio
    # for env_idx in range(actions.shape[1]):
        # Individual wheel slip ratios
        # wheel_slip_BL = (observations[start_idx:end_idx, env_idx, 5]*0.06/observations[start_idx:end_idx, env_idx, 9])-1
        # wheel_slip_BR = (observations[start_idx:end_idx, env_idx, 6]*0.06/observations[start_idx:end_idx, env_idx, 10])-1
        # wheel_slip_FL = (observations[start_idx:end_idx, env_idx, 7]*0.06/observations[start_idx:end_idx, env_idx, 11])-1
        # wheel_slip_FR = (observations[start_idx:end_idx, env_idx, 8]*0.06/observations[start_idx:end_idx, env_idx, 12])-1
        
        # ax2.plot(time[start_idx:end_idx], wheel_slip_BL, color='red', alpha=0.5, label='BL slip' if env_idx==0 else "")
        # ax2.plot(time[start_idx:end_idx], wheel_slip_BR, color='orange', alpha=0.5, label='BR slip' if env_idx==0 else "")
        # ax2.plot(time[start_idx:end_idx], wheel_slip_FL, color='blue', alpha=0.5, label='FL slip' if env_idx==0 else "")
        # ax2.plot(time[start_idx:end_idx], wheel_slip_FR, color='cyan', alpha=0.5, label='FR slip' if env_idx==0 else "")

    # Mean slip ratio
    # ax2.plot(time[start_idx:end_idx], slip_ratio[start_idx:end_idx], color='black', label='Mean slip ratio')
    # ax2.axhline(0, color='gray', linestyle='--')  # Reference line at zero slip

    # ax2.set_xlabel('Time (s)')
    # ax2.set_ylabel('Slip Ratio')
    # ax2.set_title('Wheel Slip Ratios')
    # ax2.legend()
    # ax2.grid(True)

    # plt.tight_layout()
    # plt.show()
