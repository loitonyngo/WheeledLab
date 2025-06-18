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
POLICY = 'efficient-bee-970'
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
parser.add_argument("--steps", type=int, default=200, help="Length of recorded video in steps")
# Logging
parser.add_argument('-sd', "--save-data", action="store_true", default=True, help="Save episode data")
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
        save_path = os.path.join(playback_dir, f"{args_cli.play_name}-rollouts.pt")
        torch.save(data, save_path)
        print(f"[INFO] Saved episode data to: {save_path}")

    print("Done playing policy. Closing environment.")
    env.close()


    # Load the saved data
    data_path = "/home/tongo/WheeledLab/source/wheeledlab_rl/logs/"+POLICY+"/playback/play-name-rollouts.pt"
    data = torch.load(data_path)

    # Convert to numpy for plotting (if needed)
    actions = data['actions'].cpu().numpy()            # Shape: [timesteps, num_envs, action_dim]
    observations = data['observations'].cpu().numpy()  # Shape: [timesteps, num_envs, obs_dim]
    time = data['time'].cpu().numpy()  
    s_idx = torch.squeeze(data['s_idx']).cpu().numpy()  
    pos_xy = torch.squeeze(data['pos_xy']).cpu().numpy() 
    
    try:
        inner = torch.squeeze(data['inner_bounds']).cpu().numpy() 
        outer = torch.squeeze(data['outer_bounds']).cpu().numpy() 
    except:
        print('WARNING try except')

    reset_idx = np.where(np.diff(s_idx)<(-np.max(s_idx)+10))
    # start_idx = reset_idx[0][0] 
    # end_idx = reset_idx[0][1] 
    start_idx = 0 
    end_idx = 200

    plt.figure(figsize=(12, 4))
    for env_idx in range(actions.shape[1]):  # Loop through environments
        plt.plot(time[start_idx:end_idx], actions[start_idx:end_idx, env_idx, 0], label=f'Env {env_idx} (Throttle)')
        plt.plot(time[start_idx:end_idx], actions[start_idx:end_idx, env_idx, 1], '--', label=f'Env {env_idx} (Steering)')
    plt.xlabel("time [s]")
    plt.ylabel("Action Value")
    plt.title(POLICY+": Policy Actions Over Time")
    plt.legend()
    plt.grid()
    plt.show()
    vel = np.sqrt(np.square(observations[:, 0, 0]) + np.square(observations[:, 0, 1]))

    # Example: Plot heading error and lateral deviation
    plt.figure(figsize=(12, 4))
    plt.plot(time[start_idx:end_idx], observations[start_idx:end_idx, 0, 0], label='Base Lin Vel X')  # Adjust indices based on your obs space
    plt.plot(time[start_idx:end_idx], observations[start_idx:end_idx, 0, 1], label='Base Lin Vel Y')
    plt.plot(time[start_idx:end_idx], vel[start_idx:end_idx], label='Vel')
    plt.plot(time[start_idx:end_idx], s_idx[start_idx:end_idx]/np.max(s_idx[start_idx:end_idx]),  label='s_idx')
    plt.xlabel("time [s]")
    plt.ylabel("velocity [m/s]")
    plt.title(POLICY+": Selected Observation Features")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(time[start_idx:end_idx], observations[start_idx:end_idx, 0, 3], label='Last Throttle')  # Adjust indices based on your obs space
    plt.plot(time[start_idx:end_idx], observations[start_idx:end_idx, 0, 4], label='Last Steering')
    plt.xlabel("time [s]")
    plt.ylabel("")
    plt.title(POLICY+": Selected Observation Features")
    plt.legend()
    plt.grid()
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