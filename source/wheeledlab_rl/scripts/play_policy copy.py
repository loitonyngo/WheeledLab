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

parser = argparse.ArgumentParser(description="Play a policy in WheeledLab.")
# These arguments assume that a run folder can be found
# Add this line to accept the policy name as an argument
# parser.add_argument("--policy", type=str, default='revived-durian-924', 
#                    help="Policy name to use (default: revived-durian-924)")

# It would be nice to define policy as args_cli 30 N 3.8 kg
###################################
###### DEFINE POLICY TO PLAY ######
###################################
DEFAULT_LOGS_PATH = "/home/tongo/WheeledLab/source/wheeledlab_rl/logs/"
POLICY = 'misty-cloud-1041'
SAVE_NAME = 'test'
SAVE_DIR = '/home/tongo/WheeledLab/source/wheeledlab_rl/logs_play_policy'
TIMESTAMP = datetime.now().strftime("%m%d_%H%M")

REAL_DATA_DIR = "/home/tongo/WheeledLab/source/wheeledlab_rl/real_data/"
# REAL_DATA_NAME = "bb_speed_3_angle_3_p_2.csv"
REAL_DATA_NAME = "speed_3_angle_4_n.csv"

REAL_DATA_PATH = os.path.join(REAL_DATA_DIR, REAL_DATA_NAME)
###################################
###################################
###################################


parser.add_argument('-p', "--run-path", type=str, 
                   default=DEFAULT_LOGS_PATH+POLICY, 
                   help="Path to run folder")

parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint to load")
# If no run folder, the task and policy model must be provided
parser.add_argument("--task", type=str, default=None, help="Task name. Overrides run config env if provided")
parser.add_argument("--policy-path", type=str, default=None, help="Path to policy file.")

# Playback
parser.add_argument("--steps", type=int, default=150, help="Length of recorded video in steps")
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
        'theta': [],
        's_idx': [],
        'time': [],
        's_idx_max': [],
        'inner_bounds': [],
        'outer_bounds': []
    }

    ### PLAY POLICY ###

    # reset environment
    obs, _ = env.get_observations()

    action_testing = False  # Set to True if you want to test actions manually
    if action_testing:
        
        real_data = pd.read_csv(REAL_DATA_PATH)

        if args_cli.steps > len(real_data):
            cmd_steering = torch.zeros(args_cli.steps+1, device=env.unwrapped.device)
            cmd_velocity = torch.zeros(args_cli.steps+1, device=env.unwrapped.device)
            time_data = torch.zeros(args_cli.steps+1, device=env.unwrapped.device)
            cmd_steering[:len(real_data["cmd_steering_angle"])] = torch.tensor(real_data["cmd_steering_angle"].values)
            cmd_velocity[:len(real_data["cmd_velocity"])] = torch.tensor(real_data["cmd_velocity"].values)
            time_data[:len(real_data["Time"])] = torch.tensor(real_data["Time"].values)
        else:
            cmd_steering = torch.tensor(real_data["cmd_steering_angle"].values)
            cmd_velocity = torch.tensor(real_data["cmd_velocity"].values)
            time_data = torch.tensor(real_data["Time"].values)


        # Create new time points at fixed interval dt
        dt = env.cfg.sim.dt*env.cfg.decimation  # your desired time interval
        new_time = torch.arange(time_data.min(), time_data.max(), dt)

        # Resample both time series
        cmd_steering_resampled = resample_time_series(time_data, cmd_steering, new_time)
        cmd_velocity_resampled = resample_time_series(time_data, cmd_velocity, new_time)

        # plt.plot(new_time, cmd_steering_resampled, label='Resampled Steering')
        # plt.plot(new_time, cmd_velocity_resampled, label='Resampled Velocity')
        # plt.show()

    # simulate environment
    for time_idx in tqdm(range(args_cli.steps), desc="Playing policy"):
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # actions = torch.clip(actions, min=env.action_space.low, max=env.action_space.high)
            MAX_SPEED = 10
            MAX_ANGLE = 0.40
            CMD_TO_REAL_MULTIPLIER_SPEED = 1.2  # This is the multiplier to convert command speed to real speed
            CMD_TO_REAL_MULTIPLIER_ANGLE = 1  # This is the multiplier to convert command angle to real angle

            if action_testing:
                # CMD_SPEED = 5
                # SET_SPEED = CMD_SPEED / CMD_TO_REAL_MULTIPLIER 

                # CMD_ANGLE = 0.0
                # SET_ANGLE = CMD_ANGLE / CMD_TO_REAL_MULTIPLIER_ANGLE
                
                # actions[:,0] = torch.ones(actions.shape[0])*(SET_SPEED/MAX_SPEED)
                # actions[:,1] = torch.ones(actions.shape[0])*(SET_ANGLE/MAX_ANGLE)

                actions[:,0] = cmd_velocity[time_idx+1]/MAX_SPEED / CMD_TO_REAL_MULTIPLIER_SPEED
                actions[:,1] = cmd_steering[time_idx+1]/MAX_ANGLE /CMD_TO_REAL_MULTIPLIER_ANGLE

            # env stepping
            obs, rew, _, extras = env.step(actions)
        # save data
        data['observations'].append(obs)
        data['rewards'].append(rew)
        data['actions'].append(actions)
        try:
            data['theta'].append(extras['theta'])
        except:
            print('WARNING: could not store data')
        try:
            data['pos_xy'].append(extras['pos_xy'])
        except:
            print('WARNING: could not store data')
        try:
            data['s_idx'].append(extras['s_idx'])
            data['s_idx_max'].append(extras['s_idx_max'])
        except:
            print('WARNING: could not store data')            
        try:
            data['time'].append(extras['time'])
        except:
            print('WARNING: could not store data')
        try:
           data['inner_bounds'].append(extras['inner'])
           data['outer_bounds'].append(extras['outer'])
        except:
            print('WARNING: could not store data')

    print(time_idx)
    ###


    ########################
    ###### SAVE DATA #######
    ########################

    if args_cli.save_data:
        for key in data.keys():
            try:
                data[key] = torch.stack(data[key], dim=0)
            except:
                print('WARNING: could not torch.stack')
                continue
        
        # Base filename without extension
        base_name = os.path.join(SAVE_DIR, f"{args_cli.save_name}")
        
        # Check if file exists and find appropriate suffix
        suffix = ""
        counter = 0
        while True:
            save_path = f"{base_name}{suffix}.pt"
            if not os.path.exists(save_path):
                break
            suffix = f"_{counter}"
            counter += 1
        
        torch.save(data, save_path)
        print(f"[INFO] Saved episode data to: {save_path}")

    print("Done playing policy. Closing environment.")
    env.close()

##############################################################
######################### PLOT DATA #########################
##############################################################


    # Load the saved data
    data_path = save_path
    data = torch.load(data_path)

    # Load real data to compare
    real_data = pd.read_csv(REAL_DATA_PATH)
    # real_data = real_data.loc[real_data['cmd_velocity']>0]  # Ensure same length
    # real_data['Time'] = real_data['Time'] - real_data['Time'].iloc[0] - 0.10  # Limit to the same number of steps

    # Convert to numpy for plotting (if needed)
    actions = data['actions'].cpu().numpy()            # Shape: [timesteps, num_envs, action_dim]
    observations = data['observations'].cpu().numpy()  # Shape: [timesteps, num_envs, obs_dim]
    time = data['time'].cpu().numpy()  
    
    s_idx = torch.squeeze(data['s_idx']).cpu().numpy()  
    theta = torch.squeeze(data['theta']).cpu().numpy()

    reset_idx = np.where(np.diff(s_idx)<(-np.max(s_idx)+10))
    # start_idx = reset_idx[0][0] 
    # end_idx = reset_idx[0][1] 
    start_idx = 0 
    end_idx = 1000

    pos_xy = torch.squeeze(data['pos_xy']).cpu().numpy() 
    vel = np.sqrt(np.square(observations[:, 0, 0]) + np.square(observations[:, 0, 1]))
    acceleration = np.gradient(vel[start_idx:end_idx], time[start_idx:end_idx])

    ############ Align with real data length ############
    # Get the start and end times for both datasets
    # real_start_time = real_data['Time'].iloc[0]
    real_end_time = real_data['Time'].iloc[-1]

    # sim_start_time = time.min()  # Assuming 'time' is a numpy array
    sim_end_time = time.max()

    # # Find the overlapping time range
    # global_start = max(real_start_time, sim_start_time)
    global_end = min(real_end_time, sim_end_time)

    # # Trim real_data to the overlapping range
    # real_data = real_data[
    #     (real_data['Time'] >= global_start) & 
    #     (real_data['Time'] <= global_end)
    # ]

    real_data = real_data[
        (real_data['Time'] <= global_end)
    ]

    # # Trim simulated data (assuming 'time' is a 1D array)
    # sim_mask = (time >= global_start) & (time <= global_end)
    sim_mask = time <= global_end

    s_idx = s_idx[sim_mask]  # Trim s_idx accordingly
    theta = theta[sim_mask]  # Trim theta accordingly
    time = time[sim_mask]
    observations = observations[sim_mask]  # Trim observations accordingly
    actions = actions[sim_mask]           # Trim actions accordingly
    pos_xy = pos_xy[sim_mask]             # Trim pos_xy accordingly
    pos_xy[:,0] = pos_xy[:,0]-pos_xy[0,0]
    pos_xy[:,1] = pos_xy[:,1]-pos_xy[0,1]


    vel = vel[sim_mask]                 # Trim vel accordingly
    acceleration = acceleration[sim_mask]  # Trim acceleration accordingly

    from scipy.interpolate import interp1d

    # Step 1: Extract relevant data
    sim_time = time  # Already trimmed
    sim_vx = observations[:, 0, 0]  # Simulated vx

    real_time = real_data['Time'].values
    real_vx = real_data['vx'].values
    real_x = real_data['x'].values
    real_y = real_data['y'].values

    # Step 2: Interpolate sim vx to real_time points
    sim_vx_interp_func = interp1d(sim_time, sim_vx, kind='linear', fill_value="extrapolate")
    x_interp_func = interp1d(sim_time, pos_xy[:,0], axis=0, kind='linear', fill_value="extrapolate")
    y_interp_func = interp1d(sim_time, pos_xy[:,1], axis=0, kind='linear', fill_value="extrapolate")

    sim_vx_interp = sim_vx_interp_func(real_time)
    x_interp = x_interp_func(real_time)
    y_interp = y_interp_func(real_time)


    # Step 3: Compute the difference
    vx_diff = real_vx - sim_vx_interp
    xy_diff = np.sqrt((real_x - x_interp)**2 + (real_y - y_interp)**2)

    try:
        inner = torch.squeeze(data['inner_bounds']).cpu().numpy() 
        outer = torch.squeeze(data['outer_bounds']).cpu().numpy() 
    except:
        print('WARNING try except')


    plt.figure(figsize=(15, 15))  # Adjusted height for 4 subplots

    # ---------------------------
    # Subplot 1: Command Velocity and Steering
    # ---------------------------
    ax1 = plt.subplot(4, 1, 1)
    ax1b = ax1.twinx()

    # Plot velocity on ax1 (left y-axis)
    ax1.plot(time[start_idx:end_idx], actions[start_idx:end_idx, 0, 0]*MAX_SPEED, 
            color='green', label='Model cmd velocity')
    ax1.plot(real_data['Time'], real_data['cmd_velocity'], 
            color='green', linestyle='--', label='Real cmd velocity')

    # Plot steering on ax1b (right y-axis)
    ax1b.plot(time[start_idx:end_idx], actions[start_idx:end_idx, 0, 1]*MAX_ANGLE, 
            color='blue', label='Model cmd steering')
    ax1b.plot(real_data['Time'], real_data['cmd_steering_angle'], 
            color='blue', linestyle='--', label='Real cmd steering')

    # Customize axes
    ax1.set_ylabel('Velocity', color='green')
    ax1b.set_ylabel('Steering Angle', color='blue')
    ax1.tick_params(axis='y', colors='green')
    ax1b.tick_params(axis='y', colors='blue')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1b, labels1b = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines1b, labels1 + labels1b, loc='upper right')
    ax1.set_title(f"{POLICY}: Command Velocity and Steering")
    ax1.grid(True)

    # ---------------------------
    # Subplot 2: Linear Velocity and Acceleration
    # ---------------------------
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2b = ax2.twinx()

    # Plot linear velocities
    ax2.plot(time[start_idx:end_idx], observations[start_idx:end_idx, 0, 0], 
            color='g', label='Lin Vel X (sim)')
    ax2.plot(real_data['Time'], real_data['vx'], 
            label='Vel X (real)', color='g', linestyle='--')

    ax2.plot(time[start_idx:end_idx], -observations[start_idx:end_idx, 0, 1], 
            color='b', label='Lin Vel Y (sim)')
    ax2.plot(real_data['Time'], real_data['vy'], 
            label='Vel Y (real)', color='b', linestyle='--')

    ax2.plot(real_data['Time'], vx_diff, 
            color='grey', label='real vx - sim vx (diff)', 
            linestyle=':', linewidth=3)

    # Plot acceleration
    ax2b.plot(time[start_idx:end_idx], acceleration, 
            color='r', label='Acceleration (sim)')
    ax2b.plot(real_data['Time'], real_data['ax'], 
            color='r', linestyle='--', label='Acceleration (data)')

    # Customize axes
    ax2.set_ylabel("velocity [m/s]")
    ax2b.set_ylabel("acceleration [m/s²]", color='r')
    ax2b.tick_params(axis='y', labelcolor='r')

    # Combine legends
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2b, labels2b = ax2b.get_legend_handles_labels()
    ax2.legend(lines2 + lines2b, labels2 + labels2b, loc='upper right')
    ax2.set_title("Linear Velocity and Acceleration")
    ax2.grid(True)

    # ---------------------------
    # Subplot 3: Angular Velocity
    # ---------------------------
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3b = ax3.twinx()

    ax3.plot(time[start_idx:end_idx], observations[start_idx:end_idx, 0, 2], 
            color='g', label='Sim Ang Vel Z')
    ax3.plot(real_data['Time'], real_data['omega'], 
            label='Real Ang Vel Z', color='g', linestyle='--')

    ax3b.plot(time[start_idx:end_idx], theta[start_idx:end_idx], 
            color='r', label='Sim Theta')
    ax3b.plot(real_data['Time'], real_data['theta'], 
            label='Real theta', color='r', linestyle='--')

    ax3b.set_ylabel("heading angle [rad]", color='r')
    ax3b.tick_params(axis='y', labelcolor='r')

    ax3.set_ylabel("ang velocity [rad/s]")
    ax3.legend()
    ax3.grid(True)
    ax3.set_title("Angular Velocity")

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines3b, labels3b = ax3b.get_legend_handles_labels()
    ax3.legend(lines3 + lines3b, labels3 + labels3b, loc='upper right')
    ax3.set_title("Angular Velocity and Heading Angle")
    ax3.grid(True)

    # ---------------------------
    # Subplot 4: Position Difference
    # ---------------------------
    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    ax4.plot(real_data['Time'], xy_diff, 
            label='xy pos diff', color='black')

    ax4.set_xlabel("time [s]")
    ax4.set_ylabel("xy diff [m]")
    ax4.legend()
    ax4.grid(True)
    ax4.set_title("Position Difference (Real vs Sim)")

    plt.tight_layout()
    plt.show()

    # ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, 0, 1], color='b', label='Real Lin Vel ')
    # ax1.plot(real_data['Time'], real_data['vy'], label='Real Vel Y', color='b', linestyle='--')


    # ax1.plot(time[start_idx:end_idx], vel[start_idx:end_idx], label='Vel')
    # ax1.plot(time[start_idx:end_idx], s_idx[start_idx:end_idx]/np.max(s_idx[start_idx:end_idx]), label='s_idx (normalized)')
    # ax1.set_xlabel("time [s]")
    # ax1.set_ylabel("velocity [m/s]")
    # ax1.grid(True)

    plt.figure(figsize=(10,10))
    try:
        plt.scatter(inner[:,0], inner[:,1], color='black')
        plt.scatter(outer[:,0], outer[:,1], color='black')
    except:
        print('WARNING: could not plot data')
    # Start and End for Simulated Data (offset to start from 0)


    # Full Trajectories with Color Mapping
    real = plt.scatter(real_data['x'], real_data['y'],                
                    c=real_data['vx'], cmap='plasma', alpha=0.75)  

    sim = plt.scatter(pos_xy[start_idx:end_idx,0], 
                    pos_xy[start_idx:end_idx,1],                
                    c=vel[start_idx:end_idx], cmap='coolwarm', alpha=0.75)

    plt.scatter(0, 0, color='black', label='Start Position (sim)')  # Start is (0, 0)s
    plt.scatter(pos_xy[-1,0], pos_xy[-1,1], color='red', label='End Position (sim)')

    # Start and End for Real
    plt.scatter(real_data['x'].iloc[0], real_data['y'].iloc[0], color='black', label='Start Position (data)')
    plt.scatter(real_data['x'].iloc[-1], real_data['y'].iloc[-1], color='blue', label='End Position (data)')
    
    try:
        # Plot specific points for comparison
        plt.scatter(real_data['x'].iloc[40], real_data['y'].iloc[40], color='green', alpha=1)
        plt.scatter(pos_xy[40,0], pos_xy[40,1], color = 'green')

        plt.scatter(real_data['x'].iloc[80], real_data['y'].iloc[80], color='green', alpha=1)
        plt.scatter(pos_xy[80,0], pos_xy[80,1], color = 'green')

        plt.scatter(real_data['x'].iloc[120], real_data['y'].iloc[120], color='green', alpha=1)
        plt.scatter(pos_xy[120,0], pos_xy[120,1], color = 'green')
    except:
        print('WARNING: could not plot data')

    # Colorbars
    cbar_real = plt.colorbar(real)
    cbar = plt.colorbar(sim)

    cbar_real.set_label('Velocity (m/s) (sim)')
    cbar.set_label('Velocity (m/s) (real)')

    plt.legend()

    cbar_real.set_label('Velocity (m/s) (real)')
    cbar.set_label('Velocity (m/s) (sim)')


    plt.legend()
    plt.grid()
    plt.show()

    vel = np.sqrt(np.square(observations[:, 0, 0]) + np.square(observations[:, 0, 1]))


    plt.figure(figsize=(15, 10))

    # Create subplots (2 rows, 1 column)
    ax1 = plt.subplot(2, 1, 1)  # Velocity plot
    ax2 = plt.subplot(2, 1, 2)  # Slip ratio plot

    # Calculate metrics
    wheel_ang_vel_mean = np.mean(observations[:, 0, 5:9], axis=1)
    wheel_lin_vel_mean = np.mean(observations[:, 0, 9:13], axis=1)
    slip_ratio = (wheel_ang_vel_mean*0.06/wheel_lin_vel_mean-1)

    # Plot 1: Velocities
    for env_idx in range(actions.shape[1]):
        # Angular velocities (converted to linear by multiplying with radius)
        ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 5]*0.06, '--', color='red', alpha=0.5, label='BL ang_vel×r' if env_idx==0 else "")
        ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 6]*0.06, '--', color='orange', alpha=0.5, label='BR ang_vel×r' if env_idx==0 else "")
        ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 7]*0.06, '--', color='blue', alpha=0.5, label='FL ang_vel×r' if env_idx==0 else "")
        ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 8]*0.06, '--', color='cyan', alpha=0.5, label='FR ang_vel×r' if env_idx==0 else "")
        
        # Linear velocities
        ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 9], color='red', alpha=0.5, label='BL lin_vel' if env_idx==0 else "")
        ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 10], color='orange', alpha=0.5, label='BR lin_vel' if env_idx==0 else "")
        ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 11], color='blue', alpha=0.5, label='FL lin_vel' if env_idx==0 else "")
        ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 12], color='cyan', alpha=0.5, label='FR lin_vel' if env_idx==0 else "")

    # Plot mean values

    ax1.plot(time[start_idx:end_idx], wheel_ang_vel_mean[start_idx:end_idx]*0.06, '--', color='black', label='Mean ang_vel×r')
    ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 13]*0.06, '--', color='red', marker='x', label='Mean ang_speed×r')

    ax1.plot(time[start_idx:end_idx], wheel_lin_vel_mean[start_idx:end_idx], color='blue', marker='x', label='Mean lin_vel')
    ax1.plot(time[start_idx:end_idx], vel[start_idx:end_idx], color='purple', label='Base speed')

    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Wheel Velocities')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Slip Ratio
    for env_idx in range(actions.shape[1]):
        # Individual wheel slip ratios
        wheel_slip_BL = (observations[start_idx:end_idx, env_idx, 5]*0.06/observations[start_idx:end_idx, env_idx, 9])-1
        wheel_slip_BR = (observations[start_idx:end_idx, env_idx, 6]*0.06/observations[start_idx:end_idx, env_idx, 10])-1
        wheel_slip_FL = (observations[start_idx:end_idx, env_idx, 7]*0.06/observations[start_idx:end_idx, env_idx, 11])-1
        wheel_slip_FR = (observations[start_idx:end_idx, env_idx, 8]*0.06/observations[start_idx:end_idx, env_idx, 12])-1
        
        ax2.plot(time[start_idx:end_idx], wheel_slip_BL, color='red', alpha=0.5, label='BL slip' if env_idx==0 else "")
        ax2.plot(time[start_idx:end_idx], wheel_slip_BR, color='orange', alpha=0.5, label='BR slip' if env_idx==0 else "")
        ax2.plot(time[start_idx:end_idx], wheel_slip_FL, color='blue', alpha=0.5, label='FL slip' if env_idx==0 else "")
        ax2.plot(time[start_idx:end_idx], wheel_slip_FR, color='cyan', alpha=0.5, label='FR slip' if env_idx==0 else "")

    # Mean slip ratio
    ax2.plot(time[start_idx:end_idx], slip_ratio[start_idx:end_idx], color='black', label='Mean slip ratio')
    ax2.axhline(0, color='gray', linestyle='--')  # Reference line at zero slip

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Slip Ratio')
    ax2.set_title('Wheel Slip Ratios')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
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
