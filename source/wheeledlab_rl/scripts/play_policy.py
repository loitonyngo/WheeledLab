"""
Play a trained policy in the Isaac Lab WheeledLab environment, optionally record a video,
and save time-series data (as .pt and .csv) for offline analysis / comparison with real data.

Typical usage
-------------
# 1) Use a run folder (recommended): load env+agent cfg and the latest checkpoint
python play_policy.py -p <path-to-run> --steps 200 --video

# 2) Use explicit task and policy path
python play_policy.py --task WheeledLab-MyTask-v0 --policy-path /path/to/policy.pt --steps 200

Flags
-----
-p / --run-path         Path to a training run directory that contains run_config.pkl and checkpoints.
--checkpoint            Which checkpoint (int) to load; if omitted, uses the latest.
--task                  Task name (overrides run config if given without --run-path).
--policy-path           Path to a saved policy checkpoint (when not using --run-path).

--steps                 Number of environment steps to play / video length if --video is enabled.
-sd / --save-data       Save episode data (.pt for full tensors, .csv for selected fields).
--save-name             Base name for saved files (auto-increments if a name collision occurs).

--video                 Record a video of the episode.
--log-dir               Where to store video/data when not using --run-path (default: playback/).
--play-name             Prefix for the video filename.

Outputs
-------
- <SAVE_DIR>/<SAVE_NAME>*.pt : all stacked tensors (obs, actions, rewards, selected extras)
- <SAVE_DIR>/<SAVE_NAME>*.csv: clean subset for quick analysis (time, position, commands, states, opponent)
- Optional: video MP4 in <playback_dir> if --video is provided
"""

###################################
###### BEGIN ISAACLAB SPINUP ######
###################################
from wheeledlab_rl.startup import startup

import argparse
import os
from pathlib import Path
from datetime import datetime

import gymnasium as gym
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import load_pickle
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from wheeledlab_rl.configs import RunConfig
from wheeledlab_rl.utils import ClipAction


###################################
###### DEFAULT PATHS / NAMES ######
###################################

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGS_PATH = PROJECT_ROOT / "wheeledlab_rl" / "logs"
SAVE_DIR = PROJECT_ROOT / "wheeledlab_rl" / "output_play_policy"

# These are just convenient defaults you can change quickly
POLICY = "OVERTAKE_ITA_MT_fast_mincurv"
SAVE_NAME = "OVERTAKE_ITA_MT_results"
TIMESTAMP = datetime.now().strftime("%m%d_%H%M")

###################################
############# ARGS ################
###################################

parser = argparse.ArgumentParser(description="Play a policy in WheeledLab.")

# Where do we load from?
parser.add_argument("-p", "--run-path", type=str, default=str(DEFAULT_LOGS_PATH / POLICY),
                    help="Path to run folder. If set, env+agent configs are loaded from here.")
parser.add_argument("--checkpoint", type=int, default=None,
                    help="Specific checkpoint to load (default: latest).")

# When not using a run folder, we need task + policy path
parser.add_argument("--task", type=str, default=None,
                    help="Task name. Overrides run config env if provided.")
parser.add_argument("--policy-path", type=str, default=None,
                    help="Path to a saved policy file (used when --run-path is not given).")

# Playback length / logging
parser.add_argument("--steps", type=int, default=200,
                    help="Number of env steps to play (also video length if --video).")

parser.add_argument("-sd", "--save-data", action="store_true", default=True,
                    help="Save time-series data (.pt and .csv).")
parser.add_argument("--save-name", type=str, default=SAVE_NAME,
                    help="Base name for outputs inside SAVE_DIR.")

# Video recording
parser.add_argument("--video", action="store_true",
                    help="Record an MP4 of the playback.")
parser.add_argument("--log-dir", type=str, default="playback/",
                    help="Output directory when NOT using --run-path. Ignored when --run-path is set.")
parser.add_argument("--play-name", type=str, default="play-name",
                    help="Prefix for the video filename.")

# Initialize Isaac Lab app + parse args (keeps IsaacLab launch consistent)
simulation_app, args_cli = startup(parser=parser)


###################################
######## PATH RESOLUTION ##########
###################################

# Validate inputs (either run-path or task+policy must exist)
if args_cli.run_path is None:
    if args_cli.task is None or args_cli.policy_path is None:
        raise ValueError("Provide either --run-path OR both --task and --policy-path.")

FROM_RUN = args_cli.run_path is not None

if FROM_RUN:
    # Load run configuration (env + agent) from saved pickle
    path_to_run_cfg_pkl = os.path.join(args_cli.run_path, "run_config.pkl")
    run_cfg: RunConfig = load_pickle(path_to_run_cfg_pkl)
    run_agent_cfg = run_cfg.agent
    task = run_cfg.env_setup.task_name if args_cli.task is None else args_cli.task

    # Resolve latest or specific checkpoint
    chkpt = args_cli.checkpoint if args_cli.checkpoint is not None else ".*"
    fp = os.path.abspath(args_cli.run_path)
    run_dirname = os.path.dirname(fp)
    run_folder = os.path.basename(fp)
    policy_resume_path = get_checkpoint_path(log_path=run_dirname, run_dir=run_folder,
                                             other_dirs=["models"], checkpoint=chkpt)

    # Put video in the run folder (keeps outputs near the model)
    playback_dir = os.path.join(args_cli.run_path, "playback")
else:
    # When not loading from a run folder, we take the explicit task and policy path
    task = args_cli.task
    run_agent_cfg = None  # We'll use hydra default or require cfg at decorator
    policy_resume_path = args_cli.policy_path
    playback_dir = args_cli.log_dir


###################################
############## MAIN ###############
###################################

@hydra_task_config(task, agent_entry_point=None if FROM_RUN else "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    """
    Hydra-wrapped entry point: constructs the environment, loads the policy, plays the episode,
    and optionally saves/plots the results.

    Parameters
    ----------
    env_cfg : ManagerBasedRLEnvCfg
        Environment configuration (from run_config.pkl or task registry via hydra).
    agent_cfg : Hydra config node
        Agent (PPO) configuration. If loading from run-path, we override with run_agent_cfg.
    """
    # If we came from a run folder, enforce that agent_cfg matches the run
    if agent_cfg is None:
        agent_cfg = run_agent_cfg

    # Ensure playback folder exists
    os.makedirs(playback_dir, exist_ok=True)
    print(f"[INFO] Playback directory: {playback_dir}")

    # ------------------------------
    # 1) Build env and video wrapper
    # ------------------------------
    # If --video is set, wrap env to dump a video of length `steps` every `steps` steps
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(task, cfg=env_cfg, render_mode=render_mode)

    if args_cli.video:
        video_kwargs = {
            "video_folder": playback_dir,
            "step_trigger": lambda step: step % args_cli.steps == 0,  # one video per run
            "video_length": args_cli.steps,
            "disable_logger": True,
            "name_prefix": args_cli.play_name,
        }
        print(f"[INFO] Recording video to: {playback_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # --------------------------------------------
    # 2) Match action range, wrap and create runner
    # --------------------------------------------
    # Many trained policies assume actions ∈ [-1, 1]. Enforce that here to be safe.
    env.action_space.low = -1.0
    env.action_space.high = 1.0
    env = ClipAction(env)                  # Hard-clip actions to the env bounds
    env = RslRlVecEnvWrapper(env)          # RSL-RL vectorized env adapter

    # RSL-RL PPO runner (loads policies and runs rollouts)
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict())
    ppo_runner.load(policy_resume_path)

    # Inference-only actor (no gradients, deterministic/eval mode by default)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # ------------------------------
    # 3) Episode data buffers
    # ------------------------------
    # We store raw tensors (stacked later) AND a small CSV-friendly subset for quick plots.
    data = {
        "observations": [],
        "rewards": [],
        "actions": [],
        # extras (exported by your env in rewards/terminations/utils)
        "pos_xy": [], "vel_x": [], "vel_y": [], "yaw_rate": [],
        "target_velocity": [], "target_steering": [], "theta": [],
        "s_idx": [], "time": [], "s_idx_max": [],
        "delta_s_opp_ego": [], "opp_pos_xy": [], "opp_speed": [],
    }

    # ------------------------------
    # 4) Rollout loop (play policy)
    # ------------------------------
    obs, _ = env.get_observations()  # first reset
    for _ in tqdm(range(args_cli.steps), desc="Playing policy"):
        with torch.inference_mode():
            actions = policy(obs)                 # policy forward
            obs, rew, _, extras = env.step(actions)

        # Store raw tensors (we stack later to keep shapes simple)
        data["observations"].append(obs)
        data["rewards"].append(rew)
        data["actions"].append(actions)

        # Store selected extras only if provided by the env
        fields = [
            "pos_xy", "theta",
            "vel_x", "vel_y", "yaw_rate",
            "target_velocity", "target_steering",
            "s_idx", "s_idx_max", "opp_pos_xy", "opp_speed",
            "time",
        ]
        for f in fields:
            val = extras.get(f)
            if val is not None:
                data[f].append(val)
            else:
                # Extras depend on your env instrumentation; warn if missing.
                # (Low noise but helpful to diagnose missing log keys.)
                print(f"[WARN] Extra '{f}' not present this step")

    # We’re done with the env; ensure files flush and GUI closes.
    print("Done playing policy. Closing environment.")
    env.close()

    # ------------------------------
    # 5) Save episode data (optional)
    # ------------------------------
    save_csv_path = None
    if args_cli.save_data:
        # Create a dedicated folder under SAVE_DIR (outside run folder) to collect episodes
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_folder = os.path.join(SAVE_DIR, SAVE_NAME)
        os.makedirs(save_folder, exist_ok=True)

        # Base filename (we auto-increment a suffix on collision)
        base_name = os.path.join(save_folder, f"{args_cli.save_name}")
        suffix, counter = "", 0
        while True:
            save_pt_path = f"{base_name}{suffix}.pt"
            save_csv_path = f"{base_name}{suffix}.csv"
            if not os.path.exists(save_pt_path) and not os.path.exists(save_csv_path):
                break
            suffix = f"_{counter}"
            counter += 1

        # .pt: stack everything we can (not all lists have the same shape)
        for key in list(data.keys()):
            try:
                data[key] = torch.stack(data[key], dim=0)
            except Exception as e:
                # Keep non-stackable entries as-is (e.g., differing shapes)
                print(f"[WARN] Could not stack '{key}' ({e}); keeping as list.")

        torch.save(data, save_pt_path)
        print(f"[INFO] Saved full episode tensors to: {save_pt_path}")

        # .csv: export a clean subset with consistent lengths/columns
        selected_fields = [
            "target_velocity", "target_steering", "pos_xy", "theta",
            "vel_x", "vel_y", "yaw_rate", "s_idx", "time", "opp_pos_xy", "opp_speed",
        ]

        df_dict, lengths = {}, []
        for key in selected_fields:
            if key not in data:
                print(f"[WARN] '{key}' not in data; skipping")
                continue

            # Each saved tensor has shape [T, ...]. Squeeze singleton dims and export to numpy.
            t = data[key]
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu().squeeze()
            else:
                # If we failed to stack earlier, skip exporting this key to CSV
                print(f"[WARN] '{key}' not saved as tensor; skipping CSV export for it")
                continue

            if key == "pos_xy":  # split into x,y
                df_dict["x"] = t[:, 0].numpy().astype(float)
                df_dict["y"] = t[:, 1].numpy().astype(float)
                lengths.append(t.shape[0])
            elif key == "opp_pos_xy":
                df_dict["opp_x"] = t[:, 0].numpy().astype(float)
                df_dict["opp_y"] = t[:, 1].numpy().astype(float)
                # we do not push its length; it's aligned with time later
            else:
                arr = t.numpy().astype(float)
                df_dict[key] = arr
                lengths.append(arr.shape[0])

        # Align arrays by trimming to the shortest time length (simple, robust)
        if lengths:
            min_len = int(np.min(lengths))
            for k in list(df_dict.keys()):
                df_dict[k] = df_dict[k][:min_len]

            # Stable column order and simple renaming for downstream plotting
            col_order = [
                "time", "x", "y",
                "target_velocity", "target_steering",
                "theta", "vel_x", "vel_y", "yaw_rate",
                "s_idx", "opp_x", "opp_y", "opp_speed",
            ]
            df = pd.DataFrame(df_dict)[[c for c in col_order if c in df_dict]]
            df = df.rename(columns={
                "target_velocity": "cmd_velocity",
                "target_steering": "cmd_steering_angle",
                "vel_x": "vx",
                "vel_y": "vy",
                "yaw_rate": "omega",
                "opp_speed": "opp_v",
            })
            df.to_csv(save_csv_path, index=False)
            print(f"[INFO] Saved compact episode CSV to: {save_csv_path}")
        else:
            print("[WARN] No CSV fields had consistent lengths; CSV not written.")

    # ------------------------------
    # 6) Quick plotting (if CSV exists)
    # ------------------------------
    if save_csv_path is not None and os.path.exists(save_csv_path):
        plot_episode_from_csv(save_csv_path)
    else:
        print("[INFO] Skipping plots (no CSV was written).")


# =============================================================================
# Helper: plotting from the saved CSV (kept separate to keep main() clean)
# =============================================================================

def plot_episode_from_csv(csv_path: str):
    """Load the saved CSV and generate a small set of diagnostic plots.
    The CSV uses compact columns: time, x, y, cmd_velocity, cmd_steering_angle, theta,
    vx, vy, omega, s_idx, opp_x, opp_y, opp_v
    """
    sim = pd.read_csv(csv_path)

    # Pull arrays
    time         = sim["time"].to_numpy()
    x, y         = sim["x"].to_numpy(), sim["y"].to_numpy()
    vx, vy       = sim["vx"].to_numpy(), sim["vy"].to_numpy()
    omega        = sim["omega"].to_numpy()
    theta        = sim["theta"].to_numpy()
    s_idx        = sim["s_idx"].to_numpy()
    speed_cmd    = sim["cmd_velocity"].to_numpy()
    steering_cmd = sim["cmd_steering_angle"].to_numpy()
    opp_has_pos  = {"opp_x", "opp_y"}.issubset(sim.columns)
    opp_has_v    = "opp_v" in sim.columns

    # Derived
    speed = np.sqrt(vx**2 + vy**2)
    slip_angle = np.arctan2(vy, vx)  # robust when vx≈0

    # --- Figure 1: Commands and states ---
    plt.figure(figsize=(15, 14))

    # 1) cmd vel + cmd steering
    ax1 = plt.subplot(4, 1, 1)
    ax1b = ax1.twinx()
    ax1.plot(time, speed_cmd, label="cmd velocity", linewidth=1.5)
    ax1b.plot(time, steering_cmd, label="cmd steering", linewidth=1.5)
    ax1.set_ylabel("Velocity [m/s]")
    ax1b.set_ylabel("Steering [rad]")
    ax1.set_title("Commanded velocity & steering")
    ax1.grid(True)
    lines, labels = ax1.get_legend_handles_labels()
    linesb, labelsb = ax1b.get_legend_handles_labels()
    ax1.legend(lines + linesb, labels + labelsb, loc="upper right")

    # 2) linear velocity components
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.plot(time, vx, label="vx", linewidth=1.5)
    ax2.plot(time, vy, label="vy", linewidth=1.0)
    ax2.set_ylabel("Velocity [m/s]")
    ax2.set_title("Linear velocities")
    ax2.legend(); ax2.grid(True)

    # 3) yaw rate + heading
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3b = ax3.twinx()
    ax3.plot(time, omega, label="yaw rate ω", linewidth=1.5)
    ax3b.plot(time, theta, label="heading θ", linewidth=1.0, color="tab:red")
    ax3.set_ylabel("ω [rad/s]")
    ax3b.set_ylabel("θ [rad]")
    ax3.set_title("Angular velocity & heading")
    ax3.grid(True)
    lines, labels = ax3.get_legend_handles_labels()
    linesb, labelsb = ax3b.get_legend_handles_labels()
    ax3.legend(lines + linesb, labels + labelsb, loc="upper right")

    # 4) slip angle
    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    ax4.plot(time, slip_angle, label="β (slip)", linewidth=1.5, color="tab:purple")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("β [rad]")
    ax4.set_title("Slip angle vs time")
    ax4.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Figure 2: Track velocity tracking and steering vs yaw rate ---
    plt.figure(figsize=(12, 8))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(time, speed_cmd, label="cmd velocity", linewidth=1.5)
    ax1.plot(time, vx, label="vx", linewidth=1.2)
    ax1.set_ylabel("Velocity [m/s]")
    ax1.set_title("Velocity command vs. actual vx")
    ax1.legend(); ax1.grid(True)

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(time, steering_cmd, label="cmd steering", linewidth=1.5, color="tab:red")
    ax2b = ax2.twinx()
    ax2b.plot(time, omega, label="yaw rate ω", linewidth=1.2, color="tab:purple")
    ax2.set_ylabel("Steering [rad]"); ax2b.set_ylabel("ω [rad/s]")
    ax2.set_title("Steering command vs. yaw rate")
    lines, labels = ax2.get_legend_handles_labels()
    linesb, labelsb = ax2b.get_legend_handles_labels()
    ax2.legend(lines + linesb, labels + labelsb, loc="upper right")
    ax2.grid(True)
    ax2.set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()

    # --- Figure 3: Trajectories and separation (if opponent exists) ---
    if opp_has_pos:
        opp_xy = sim[["opp_x", "opp_y"]].to_numpy()
        plt.figure(figsize=(10, 8))
        plt.scatter(x, y, s=6, label="Ego")
        plt.scatter(opp_xy[:, 0], opp_xy[:, 1], s=6, label="Opponent")
        plt.xlabel("X [m]"); plt.ylabel("Y [m]")
        plt.title("Trajectories: Ego vs Opponent")
        plt.axis("equal"); plt.grid(True); plt.legend()
        plt.show()

        # Separation vs time
        sep = np.linalg.norm(np.stack([x, y], axis=1) - opp_xy, axis=1)
        plt.figure(figsize=(10, 4))
        plt.plot(time, sep)
        plt.xlabel("Time [s]"); plt.ylabel("Separation [m]")
        plt.title("Ego–Opponent separation vs time")
        plt.grid(True); plt.tight_layout()
        plt.show()

    if opp_has_v:
        delta_v = vx - sim["opp_v"].to_numpy()
        plt.figure(figsize=(10, 4))
        plt.plot(time, delta_v)
        plt.xlabel("Time [s]"); plt.ylabel("Δv [m/s]")
        plt.title("Ego–Opponent Δv vs time")
        plt.grid(True); plt.tight_layout()
        plt.show()


# =============================================================================
# Utility: 1D linear interpolation (kept minimal; used if you add alignment later)
# =============================================================================

def resample_time_series(original_time: torch.Tensor,
                         original_values: torch.Tensor,
                         new_time: torch.Tensor) -> torch.Tensor:
    """
    Resample a 1D time series onto new timestamps using linear interpolation.

    Args
    ----
    original_time : (T,) tensor
        Monotonically increasing timestamps of the original series.
    original_values : (T,) tensor
        Values corresponding to `original_time`.
    new_time : (N,) tensor
        Desired timestamps to sample the series at.

    Returns
    -------
    (N,) tensor
        Interpolated values aligned to `new_time`.

    Notes
    -----
    - Extrapolates flat at the boundaries.
    - This helper is not used in the default flow, but is handy when aligning
      logged series of different rates.
    """
    device = original_time.device
    original_values = original_values.to(device)
    new_time = new_time.to(device)

    # Indices where each new_time would be inserted to keep order
    idx = torch.searchsorted(original_time, new_time)
    idx = torch.clamp(idx, 1, len(original_time) - 1)

    t0, t1 = original_time[idx - 1], original_time[idx]
    v0, v1 = original_values[idx - 1], original_values[idx]
    alpha = (new_time - t0) / torch.clamp(t1 - t0, min=1e-12)
    return v0 + alpha * (v1 - v0)


if __name__ == "__main__":
    main()
