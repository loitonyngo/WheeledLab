"""
Play a policy in an environment and record the data.

Usage:

python play_policy.py -p <path-to-run> -sd --video

This command will save data and record a video of the playback using an existing run folder.

"""

###################################
###### BEGIN ISAACLAB SPINUP ######
###################################
import yaml
with open("/home/tongo/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/f1tenth/config/f1tenth_config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

from wheeledlab_rl.startup import startup
import argparse
from datetime import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Play a policy in WheeledLab.")

###################################
###### DEFINE POLICY TO PLAY ######
###################################
DEFAULT_LOGS_PATH = "/home/tongo/WheeledLab/source/wheeledlab_rl/logs/"
POLICY = 'GEN_TT_20z_vel_12_steer15_fric75_nhor20ds10_del3010'
SAVE_NAME = 'test_1'
SAVE_DIR = '/home/tongo/WheeledLab/source/wheeledlab_rl/output_compare_sim_real'
TIMESTAMP = datetime.now().strftime("%m%d_%H%M")

REAL_DATA_DIR = "/home/tongo/WheeledLab/source/wheeledlab_rl/real_data/"
REAL_DATA_NAME = "CIR1_STMPC_tt.csv"

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

from scipy.interpolate import interp1d

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
    ######## POLICY LOADING CODE #######
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
        'theta': [],
        's_idx': [],
        'time': [],
        's_idx_max': []
    }

    ### PLAY POLICY ###

    # reset environment
    obs, _ = env.get_observations()
        
    real_data = pd.read_csv(REAL_DATA_PATH)

    if args_cli.steps > len(real_data):
        cmd_steering = torch.zeros(args_cli.steps+1, device=env.unwrapped.device)
        cmd_velocity = torch.zeros(args_cli.steps+1, device=env.unwrapped.device)
        time_data = torch.zeros(args_cli.steps+1, device=env.unwrapped.device)
        cmd_steering[:len(real_data["cmd_steering_angle"])] = torch.tensor(real_data["cmd_steering_angle"].values)
        cmd_velocity[:len(real_data["cmd_velocity"])] = torch.tensor(real_data["cmd_velocity"].values)
        time_data[:len(real_data["time"])] = torch.tensor(real_data["time"].values)
    else:
        cmd_steering = torch.tensor(real_data["cmd_steering_angle"].values)
        cmd_velocity = torch.tensor(real_data["cmd_velocity"].values)
        time_data = torch.tensor(real_data["time"].values)

    # Create new time points at fixed interval dt
    dt = env.cfg.sim.dt*env.cfg.decimation  # your desired time interval
    new_time = torch.arange(time_data.min(), time_data.max(), dt)

    # Resample both time series
    cmd_steering_resampled = resample_time_series(time_data, cmd_steering, new_time)
    cmd_velocity_resampled = resample_time_series(time_data, cmd_velocity, new_time)

    # simulate environment
    for time_idx in tqdm(range(args_cli.steps), desc="Playing policy"):
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            actions[:,0] = cmd_velocity_resampled[time_idx+1]/CONFIG['env_config']['MAX_SPEED_SCALING']
            actions[:,1] = cmd_steering_resampled[time_idx+1]/CONFIG['env_config']['MAX_STEERING_SCALING']
            # env stepping
            obs, rew, _, extras = env.step(actions)
            
        # save data
        data['observations'].append(obs)
        data['rewards'].append(rew)
        data['actions'].append(actions)
        
        fields = [
            'pos_xy', 'theta',
            'vel_x', 'vel_y', 'yaw_rate',
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
        selected_fields = ["actions", "pos_xy", "theta", "vel_x", "vel_y", "yaw_rate", "s_idx", "time"]

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
            elif key == "actions":
                df_dict["speed_cmd"] = tensor[:, 0].numpy().astype(float)
                df_dict["steering_cmd"] = tensor[:, 1].numpy().astype(float)
            else:
                arr = tensor.numpy().astype(float)
                df_dict[key] = arr
                lengths.append(arr.shape[0])

        # Align all arrays to the same length
        min_len = min(lengths)
        for k in df_dict:
            df_dict[k] = df_dict[k][:min_len]

        # Column order and renaming
        col_order = ["time", "x", "y", "speed_cmd", "steering_cmd", "theta", "vel_x", "vel_y", "yaw_rate", "s_idx"]
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

    # Load real data
    real_data = pd.read_csv(REAL_DATA_PATH)
    real_data['time'] -= real_data['time'].iloc[0]  # normalize start at 0

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

    # Optional bounds (if saved)
    try:
        inner = sim_data['inner_bounds'].to_numpy()
        outer = sim_data['outer_bounds'].to_numpy()
    except KeyError:
        print('WARNING: bounds not available in CSV')

    # --------------------------
    # Compute derived quantities
    # --------------------------
    vel = np.sqrt(vel_x**2 + vel_y**2)
    acceleration = np.gradient(vel, time)
    jerk = np.gradient(acceleration, time)


    # --------------------------
    # Align sim and real data time window
    # --------------------------
    real_start, real_end = real_data['time'].iloc[0], real_data['time'].iloc[-1]
    sim_start, sim_end   = time.min(), time.max()

    window_start = max(real_start, sim_start)
    window_end   = min(real_end, sim_end)

    # Masks for slicing
    sim_mask  = (time >= window_start) & (time <= window_end)
    real_mask = (real_data['time'] >= window_start) & (real_data['time'] <= window_end)

    # --------------------------
    # Slice simulation data
    # --------------------------
    sim_time       = time[sim_mask]
    sim_pos_xy     = pos_xy[sim_mask]
    sim_vel_x      = vel_x[sim_mask]
    sim_vel_y      = vel_y[sim_mask]
    sim_theta      = theta[sim_mask]
    sim_speed_cmd  = speed_cmd[sim_mask]
    sim_steering   = steering_cmd[sim_mask]
    sim_yaw_rate   = yaw_rate[sim_mask]
    sim_s_idx      = s_idx[sim_mask]

    # --------------------------
    # Slice real data
    # --------------------------
    real_time  = real_data['time'].values[real_mask]
    real_x     = real_data['x'].values[real_mask]
    real_y     = real_data['y'].values[real_mask]
    real_vx    = real_data['vx'].values[real_mask]
    real_vy    = real_data['vy'].values[real_mask]
    real_theta = real_data['theta'].values[real_mask]


    # --------------------------
    # Shift trajectories to start at (0,0)
    # --------------------------
    sim_pos_xy -= sim_pos_xy[0]       # subtract first sim position
    real_x    -= real_x[0]           # subtract first real x
    real_y    -= real_y[0]           # subtract first real y

    # --------------------------
    # Derived quantities
    # --------------------------
    vel = np.sqrt(sim_vel_x**2 + sim_vel_y**2)
    acceleration = np.gradient(vel, sim_time)
    jerk = np.gradient(acceleration, sim_time)

    
    # Interpolation functions
    interp_vel_x  = interp1d(sim_time, sim_vel_x,  kind='linear', fill_value='extrapolate')
    interp_pos_x  = interp1d(sim_time, sim_pos_xy[:,0], kind='linear', fill_value='extrapolate')
    interp_pos_y  = interp1d(sim_time, sim_pos_xy[:,1], kind='linear', fill_value='extrapolate')

    # Apply interpolation to real_time
    sim_vel_x_interp = interp_vel_x(real_time)
    sim_pos_x_interp = interp_pos_x(real_time)
    sim_pos_y_interp = interp_pos_y(real_time)

    # --------------------------
    # Estimate real initial heading using first few points to reduce noise
    # --------------------------
    N_heading_points = 20  # use first 20 points to estimate heading
    dx_real = real_x[20] - real_x[0]
    dy_real = real_y[20] - real_y[0]
    real_init_angle = np.arctan2(dy_real, dx_real)

    # --------------------------
    # Compute sim initial heading
    dx_sim = sim_pos_xy[1,0] - sim_pos_xy[0,0]
    dy_sim = sim_pos_xy[1,1] - sim_pos_xy[0,1]
    sim_init_angle = np.arctan2(dy_sim, dx_sim)

    # --------------------------
    # Rotate sim trajectory to match real initial heading
    # --------------------------
    angle_diff = real_init_angle - sim_init_angle -np.pi *0.64
    R = np.array([[np.cos(angle_diff), -np.sin(angle_diff)],
                [np.sin(angle_diff),  np.cos(angle_diff)]])
    sim_pos_xy_rotated = (R @ sim_pos_xy.T).T

    # --------------------------
    # Interpolate sim position to real timestamps
    # --------------------------
    interp_pos_x = interp1d(sim_time, sim_pos_xy_rotated[:,0], kind='linear', fill_value='extrapolate')
    interp_pos_y = interp1d(sim_time, sim_pos_xy_rotated[:,1], kind='linear', fill_value='extrapolate')

    sim_x_interp = interp_pos_x(real_time)
    sim_y_interp = interp_pos_y(real_time)


    # Now you can calculate differences
    vx_diff = real_vx - sim_vel_x_interp
    xy_diff = np.sqrt((real_x - sim_pos_x_interp)**2 + (real_y - sim_pos_y_interp)**2)


    # --------------------------
    # Translate and rotate sim pos_xy to origin
    # --------------------------
    # sim_pos_xy -= sim_pos_xy[0]
    # dx, dy = sim_pos_xy[1] - sim_pos_xy[0]
    # init_angle = np.arctan2(dy, dx)
    # R = np.array([[np.cos(-init_angle), -np.sin(-init_angle)],
    #             [np.sin(-init_angle),  np.cos(-init_angle)]])
    # sim_pos_xy = (R @ sim_pos_xy.T).T

    real_x -= real_x[0]
    real_y -= real_y[0]

    # --------------------------
    # Plotting
    # --------------------------
    plt.figure(figsize=(15, 15))

    # ---- Subplot 1: Command Velocity and Steering ----
    ax1 = plt.subplot(4,1,1)
    ax1b = ax1.twinx()

    ax1.plot(sim_time, sim_speed_cmd * CONFIG['env_config']['MAX_SPEED_SCALING'],
            color='green', label='Model cmd velocity')
    ax1.plot(real_time, real_data['cmd_velocity'].values[real_mask],
            color='green', linestyle='--', label='Real cmd velocity')

    ax1b.plot(sim_time, sim_steering * CONFIG['env_config']['MAX_STEERING_SCALING'],
            color='blue', label='Model cmd steering')
    ax1b.plot(real_time, real_data['cmd_steering_angle'].values[real_mask],
            color='blue', linestyle='--', label='Real cmd steering')

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

    ax2.plot(sim_time, sim_vel_x, color='g', label='Lin Vel X (sim)')
    ax2.plot(real_time, real_vx, color='g', linestyle='--', label='Lin Vel X (real)')

    ax2.plot(sim_time, -sim_vel_y, color='b', label='Lin Vel Y (sim)')
    ax2.plot(real_time, real_vy, color='b', linestyle='--', label='Lin Vel Y (real)')

    ax2.plot(real_time, vx_diff, color='grey', linestyle=':', linewidth=3, label='vx diff')

    # real_jerk = np.gradient(real_data['ax'].values[real_mask], real_time)

    # ax2b.plot(sim_time, acceleration, color='r', label='Acceleration (sim)')
    # ax2b.plot(real_time, real_data['ax'].values[real_mask], color='r', linestyle='--', label='Acceleration (real)')

    # ax2c.plot(sim_time, jerk, color='purple', label='Jerk (sim)')

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

    ax3.plot(sim_time, sim_yaw_rate, color='g', label='Sim Ang Vel Z')
    ax3.plot(real_time, real_data['omega'].values[real_mask], color='g', linestyle='--', label='Real Ang Vel Z')

    ax3b.plot(sim_time, sim_theta, color='r', label='Sim Theta')
    ax3b.plot(real_time, real_theta, color='r', linestyle='--', label='Real Theta')

    ax3.set_ylabel("ang velocity [rad/s]")
    ax3b.set_ylabel("heading angle [rad]", color='r')
    ax3b.tick_params(axis='y', labelcolor='r')

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines3b, labels3b = ax3b.get_legend_handles_labels()
    ax3.legend(lines3 + lines3b, labels3 + labels3b, loc='upper right')
    ax3.set_title("Angular Velocity and Heading Angle")
    ax3.grid(True)

    # ---- Subplot 4: Velocity differences ----
    ax4 = plt.subplot(4,1,4, sharex=ax1)

    ax4.plot(sim_time, sim_speed_cmd*CONFIG['env_config']['MAX_SPEED_SCALING'] - sim_vel_x,
            color='red', label='vel error (sim)')
    ax4.plot(real_time, real_data['cmd_velocity'].values[real_mask] - real_vx,
            color='red', linestyle='--', label='vel error (real)')
    ax4.plot(real_time, vx_diff, color='blue', linestyle=':', linewidth=3, label='real vx - sim vx')

    ax4.set_xlabel("time [s]")
    ax4.set_ylabel("vel diff [m/s]")
    ax4.legend()
    ax4.grid(True)
    ax4.set_title("Velocities Differences")

    plt.tight_layout()
    plt.show()

    # --------------------------
    # Plot trajectories
    # --------------------------
    plt.figure(figsize=(10, 8))
    plt.plot(sim_x_interp, sim_y_interp, label='Simulated Trajectory', color='blue', linewidth=2)
    plt.plot(real_x, real_y, label='Real Trajectory', color='red', linestyle='--', linewidth=2)

    plt.scatter(sim_x_interp[0], sim_y_interp[0], color='blue', marker='o', s=100, label='Sim Start')
    plt.scatter(real_x[0], real_y[0], color='red', marker='o', s=100, label='Real Start')

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

if __name__ == "__main__":
    main()