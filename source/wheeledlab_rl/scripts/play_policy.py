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

parser = argparse.ArgumentParser(description="Play a policy in WheeledLab.")
# These arguments assume that a run folder can be found
# Add this line to accept the policy name as an argument
# parser.add_argument("--policy", type=str, default='revived-durian-924', 
#                    help="Policy name to use (default: revived-durian-924)")

# It would be nice to define policy as args_cli
###################################
###### DEFINE POLICY TO PLAY ######
###################################
DEFAULT_LOGS_PATH = "/home/tongo/WheeledLab/source/wheeledlab_rl/logs/"
POLICY = 'divine-universe-1005'
SAVE_NAME = 'test'
SAVE_DIR = '/home/tongo/WheeledLab/source/wheeledlab_rl/logs_play_policy'
TIMESTAMP = datetime.now().strftime("%m%d_%H%M")

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
parser.add_argument("--steps", type=int, default=100, help="Length of recorded video in steps")
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

import os
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
        's_idx': [],
        'time': [],
        's_idx_max': [],
        'inner_bounds': [],
        'outer_bounds': []
    }

    ### PLAY POLICY ###

    # reset environment
    obs, _ = env.get_observations()
    # simulate environment
    for _ in tqdm(range(args_cli.steps), desc="Playing policy"):
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            actions = torch.clip(actions, min=env.action_space.low, max=env.action_space.high)
            # env stepping
            obs, rew, _, extras = env.step(actions)
        # save data
        data['observations'].append(obs)
        data['rewards'].append(rew)
        data['actions'].append(actions)
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


    # Load the saved data
    data_path = save_path
    data = torch.load(data_path)

    # Convert to numpy for plotting (if needed)
    actions = data['actions'].cpu().numpy()            # Shape: [timesteps, num_envs, action_dim]
    observations = data['observations'].cpu().numpy()  # Shape: [timesteps, num_envs, obs_dim]
    time = data['time'].cpu().numpy()  
    
    s_idx = torch.squeeze(data['s_idx']).cpu().numpy()  
    reset_idx = np.where(np.diff(s_idx)<(-np.max(s_idx)+10))
    # start_idx = reset_idx[0][0] 
    # end_idx = reset_idx[0][1] 
    start_idx = 0 
    end_idx = 200
    
    pos_xy = torch.squeeze(data['pos_xy']).cpu().numpy() 
    vel = np.sqrt(np.square(observations[:, 0, 0]) + np.square(observations[:, 0, 1]))
    acceleration = np.gradient(vel[start_idx:end_idx], time[start_idx:end_idx])

    try:
        inner = torch.squeeze(data['inner_bounds']).cpu().numpy() 
        outer = torch.squeeze(data['outer_bounds']).cpu().numpy() 
    except:
        print('WARNING try except')


    plt.figure(figsize=(12, 4))
    for env_idx in range(actions.shape[1]):  # Loop through environments
        plt.plot(time[start_idx:end_idx], actions[start_idx:end_idx, env_idx, 0], label=f'Env {env_idx} (Throttle)')
        plt.plot(time[start_idx:end_idx], actions[start_idx:end_idx, env_idx, 1], '--', label=f'Env {env_idx} (Steering)')
        # plt.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 5], '--', label=f'Env {env_idx} (ang vel y)')

    plt.xlabel("time [s]")
    plt.ylabel("Action Value")
    plt.title(POLICY+": Policy Actions Over Time")
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
        
    #     # Linear velocities
    #     ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 9], color='red', alpha=0.5, label='BL lin_vel' if env_idx==0 else "")
    #     ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 10], color='orange', alpha=0.5, label='BR lin_vel' if env_idx==0 else "")
    #     ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 11], color='blue', alpha=0.5, label='FL lin_vel' if env_idx==0 else "")
    #     ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, env_idx, 12], color='cyan', alpha=0.5, label='FR lin_vel' if env_idx==0 else "")

    # # Plot mean values

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

    plt.figure(figsize=(12, 6))

    # Create primary axis for velocities
    ax1 = plt.gca()
    ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, 0, 0], label='Base Lin Vel X')
    ax1.plot(time[start_idx:end_idx], observations[start_idx:end_idx, 0, 1], label='Base Lin Vel Y')
    ax1.plot(time[start_idx:end_idx], vel[start_idx:end_idx], label='Vel')
    ax1.plot(time[start_idx:end_idx], s_idx[start_idx:end_idx]/np.max(s_idx[start_idx:end_idx]), label='s_idx (normalized)')
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("velocity [m/s]")
    ax1.grid(True)


    # Create secondary axis for acceleration
    ax2 = ax1.twinx()
    ax2.plot(time[start_idx:end_idx], acceleration, 'r--', label='Acceleration')
    ax2.set_ylabel("acceleration [m/s²]", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(POLICY + ": Velocity and Acceleration")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10,10))
    try:
        plt.scatter(inner[:,0], inner[:,1], color='black')
        plt.scatter(outer[:,0], outer[:,1], color='black')
    except:
        print('WARNING: could not plot data')
    plt.scatter(pos_xy[start_idx,0], pos_xy[start_idx,1])
    sc = plt.scatter(pos_xy[start_idx:end_idx,0], pos_xy[start_idx:end_idx,1],                
                 c=vel[start_idx:end_idx], 
                 cmap='viridis',  # You can choose any colormap you like
                 label='Car trajectory')
    
    cbar = plt.colorbar(sc)
    cbar.set_label('Velocity (m/s)')
    plt.scatter(pos_xy[start_idx,0], pos_xy[start_idx,1], color='red')

    plt.legend()
    plt.grid()
    plt.show()
if __name__ == "__main__":
    main()