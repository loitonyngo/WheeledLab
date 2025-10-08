"""
Compare simulation against real logs by replaying *real commands* in the simulator.

Overview
--------
This script loads a trained policy's environment (via an Isaac Lab run folder or explicit task),
then overrides the policy actions with *commands taken from a CSV captured on a real vehicle*.
It resamples the real commands to the simulator's control rate, feeds them to the sim, records
the simulated state, and writes aligned .pt/.csv files. It then produces plots contrasting:
- command velocity/steering (real vs sim),
- linear velocities, yaw rate, heading,
- slip angle,
- trajectory overlays and separation.

Typical usage
-------------
# Use an existing run folder (loads env+agent cfg and latest checkpoint)
python compare_sim_real.py -p <path-to-run> --steps 230 -sd --video

# Use an explicit task+policy path (when not using a run folder)
python compare_sim_real.py --task WheeledLab-MyTask-v0 --policy-path /path/to/policy.pt --steps 230

Inputs
------
- REAL_DATA_PATH: CSV with at least the columns:
    time, cmd_velocity, cmd_steering_angle, x, y, vx, vy, theta, omega  (names used below)
  Times should be seconds; the script normalizes them to start at 0.

Outputs
-------
- <SAVE_DIR>/<SAVE_NAME>*.pt  : full tensors (obs/actions/rewards/extras)
- <SAVE_DIR>/<SAVE_NAME>*.csv : compact, analysis-friendly table (sim + resampled real)
- Optional: MP4 video in the run's playback directory if --video is set
"""

###################################
###### BEGIN ISAACLAB SPINUP ######
###################################

from wheeledlab_rl.startup import startup

import argparse
from datetime import datetime
from pathlib import Path
import os

import gymnasium as gym
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import load_pickle
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from wheeledlab_rl.configs import RunConfig
from wheeledlab_rl.utils import ClipAction


# ----------------------------
# Project paths / defaults
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGS_PATH = PROJECT_ROOT / "wheeledlab_rl" / "logs"
SAVE_DIR = PROJECT_ROOT / "wheeledlab_rl" / "output_compare_sim_real" / "MT_results"

REAL_DATA_DIR = PROJECT_ROOT / "wheeledlab_rl" / "rosbag_data_csv"
REAL_DATA_NAME = "bb_speed_3_angle_3_p_2.csv"      # <-- your real CSV here
REAL_DATA_PATH = os.path.join(REAL_DATA_DIR, REAL_DATA_NAME)

# Which trained run to use by default if -p not provided
POLICY = "CIR0_0920_40hz_sp10_st10_off004_fr75_del3005_hist10_wall34_nhor20_3s_correctedobsnorm"
SAVE_NAME = "bb_speed_3_angle_3_p_2"
TIMESTAMP = datetime.now().strftime("%m%d_%H%M")

# Optional manual alignment: rotate sim trajectory to match real heading
ROTATE_TRAJ_ANGLE = -np.pi * 2.07


# ----------------------------
# CLI args
# ----------------------------
parser = argparse.ArgumentParser(description="Compare simulation against real logs in WheeledLab.")

parser.add_argument(
    "-p", "--run-path", type=str, default=str(DEFAULT_LOGS_PATH / POLICY),
    help="Path to training run folder (loads env+agent cfg + checkpoint)."
)
parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint to load (default: latest).")
parser.add_argument("--task", type=str, default=None,
                    help="Task name. Overrides run config env if provided (when not using --run-path).")
parser.add_argument("--policy-path", type=str, default=None,
                    help="Path to policy file (when not using --run-path).")

parser.add_argument("--steps", type=int, default=230,
                    help="Number of env steps to simulate / video length if --video.")

parser.add_argument("-sd", "--save-data", action="store_true", default=True,
                    help="Save episode data (.pt and .csv).")
parser.add_argument("--save-name", type=str, default=SAVE_NAME, help="Base name for results inside SAVE_DIR.")

parser.add_argument("--video", action="store_true", help="Record an MP4 of the playback.")
parser.add_argument("--log-dir", type=str, default="playback/",
                    help="Output directory when NOT using --run-path. Ignored if --run-path is set.")
parser.add_argument("--play-name", type=str, default="play-name", help="Prefix used in recorded video file.")

# Launch Isaac Lab and parse args (keeps sim startup consistent)
simulation_app, args_cli = startup(parser=parser)

# Validate inputs: must provide a run folder OR task+policy path
if args_cli.run_path is None:
    if args_cli.task is None or args_cli.policy_path is None:
        raise ValueError("Provide either --run-path OR both --task and --policy-path.")


# ----------------------------
# Utilities
# ----------------------------
def resample_1d(t_src, y_src, t_new):
    """Safe 1D resampling (numpy) with sort + duplicate-time handling for clean interpolation."""
    t_src = np.asarray(t_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    t_new = np.asarray(t_new, dtype=float)

    order = np.argsort(t_src)
    t_src, y_src = t_src[order], y_src[order]

    # drop duplicate timestamps (np.interp requires strictly increasing)
    uniq, idx = np.unique(t_src, return_index=True)
    t_src, y_src = uniq, y_src[idx]

    return np.interp(t_new, t_src, y_src)


# ----------------------------
# Resolve env/agent/ckpt paths
# ----------------------------
FROM_RUN = args_cli.run_path is not None

if FROM_RUN:
    # Load run config (env+agent)
    path_to_run_cfg_pkl = os.path.join(args_cli.run_path, "run_config.pkl")
    run_cfg: RunConfig = load_pickle(path_to_run_cfg_pkl)
    run_agent_cfg = run_cfg.agent
    task = run_cfg.env_setup.task_name if args_cli.task is None else args_cli.task

    # Resolve latest or specific checkpoint
    chkpt = args_cli.checkpoint if args_cli.checkpoint is not None else ".*"
    fp = os.path.abspath(args_cli.run_path)
    run_dirname, run_folder = os.path.dirname(fp), os.path.basename(fp)
    policy_resume_path = get_checkpoint_path(
        log_path=run_dirname, run_dir=run_folder, other_dirs=["models"], checkpoint=chkpt
    )
    playback_dir = os.path.join(args_cli.run_path, "playback")  # video next to run
else:
    # Use explicit task + policy
    task = args_cli.task
    run_agent_cfg = None
    policy_resume_path = args_cli.policy_path
    playback_dir = args_cli.log_dir


# ----------------------------
# Main: build env, align and replay real commands, save, plot
# ----------------------------
@hydra_task_config(task, agent_entry_point=None if FROM_RUN else "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    """Hydra entry: create env, load policy container, feed *real* commands into sim, save & plot."""
    # Ensure agent config matches run if we loaded from a run folder
    if agent_cfg is None:
        agent_cfg = run_agent_cfg

    os.makedirs(playback_dir, exist_ok=True)
    print(f"[INFO] Playback directory: {playback_dir}")

    # 1) Build env (optionally wrapped for video recording)
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(task, cfg=env_cfg, render_mode=render_mode)
    if args_cli.video:
        video_kwargs = {
            "video_folder": playback_dir,
            "step_trigger": lambda step: step % args_cli.steps == 0,  # one clip per run
            "video_length": args_cli.steps,
            "disable_logger": True,
            "name_prefix": args_cli.play_name,
        }
        print(f"[INFO] Recording video to: {playback_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 2) Adapt env to policy/action conventions (RSL-RL expects VecEnv and actions in [-1, 1])
    env.action_space.low, env.action_space.high = -1.0, 1.0
    env = ClipAction(env)              # hard-clip to env bounds
    env = RslRlVecEnvWrapper(env)      # vectorize for RSL-RL infra (even if num_envs=1)
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict())
    ppo_runner.load(policy_resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)  # we won't use it for commands

    # 3) Data buffers for saving (we stack tensors later)
    data = {
        "observations": [], "rewards": [], "actions": [],
        "pos_xy": [], "vel_x": [], "vel_y": [], "yaw_rate": [],
        "theta": [], "s_idx": [], "time": [], "s_idx_max": [],
    }

    # 4) Reset env and load real CSV (commands and ground-truth signals)
    obs, _ = env.get_observations()
    real_df = pd.read_csv(REAL_DATA_PATH).copy()
    real_df["time"] = real_df["time"] - real_df["time"].iloc[0]  # normalize to start at 0

    from wheeledlab_tasks.config_loader import load_config
    CONFIG = load_config()

    # 5) Prepare command streams from real data; resample onto sim control rate
    #    Sim "dt" is sim.dt * decimation (physics step * control decimation)
    dt = env.cfg.sim.dt * env.cfg.decimation
    t_min, t_max = real_df["time"].min(), real_df["time"].max()
    new_time = torch.arange(t_min, t_max, dt, device=env.unwrapped.device)

    # Torch-based linear interpolation, fast on either CPU/GPU
    def resample_time_series(original_time: torch.Tensor,
                             original_values: torch.Tensor,
                             new_time: torch.Tensor) -> torch.Tensor:
        """Linear interpolation with clamped boundaries (torch)."""
        idx = torch.searchsorted(original_time, new_time)
        idx = torch.clamp(idx, 1, len(original_time) - 1)
        t0, t1 = original_time[idx - 1], original_time[idx]
        v0, v1 = original_values[idx - 1], original_values[idx]
        alpha = (new_time - t0) / torch.clamp(t1 - t0, min=1e-12)
        return v0 + alpha * (v1 - v0)

    # Convert real commands to torch, resample to simulator update times
    t_real = torch.tensor(real_df["time"].values, device=env.unwrapped.device, dtype=torch.float32)
    cmd_vel_real = torch.tensor(real_df["cmd_velocity"].values, device=env.unwrapped.device, dtype=torch.float32)
    cmd_str_real = torch.tensor(real_df["cmd_steering_angle"].values, device=env.unwrapped.device, dtype=torch.float32)

    cmd_velocity_resampled = resample_time_series(t_real, cmd_vel_real, new_time)
    cmd_steering_resampled = resample_time_series(t_real, cmd_str_real, new_time)

    # 6) Replay loop: override actions with *real commands* (scaled to [-1,1] space)
    for k in tqdm(range(min(args_cli.steps, len(new_time) - 1)), desc="Replaying real commands in sim"):
        with torch.inference_mode():
            # dummy forward to keep RSL-RL runner happy (we overwrite actions next)
            actions = policy(obs)

            # Scale real commands into the normalized action space expected by the env
            # speed ∈ [0, MAX_SPEED_SCALING]  -> a_x ∈ [-1,1] after env scaling; we feed normalized here
            # steer ∈ [-MAX_STEER, MAX_STEER] -> a_y ∈ [-1,1]
            actions[:, 0] = cmd_velocity_resampled[k + 1] / CONFIG["env_config"]["MAX_SPEED_SCALING"]
            actions[:, 1] = cmd_steering_resampled[k + 1] / CONFIG["env_config"]["MAX_STEERING_ANGLE"]

            # Step the sim
            obs, rew, _, extras = env.step(actions)

        # Log tensors for saving
        data["observations"].append(obs)
        data["rewards"].append(rew)
        data["actions"].append(actions)

        # Log selected extras if available (your env must populate these)
        for key in ["pos_xy", "theta", "vel_x", "vel_y", "yaw_rate", "s_idx", "s_idx_max", "time"]:
            val = extras.get(key)
            if val is not None:
                data[key].append(val)
            else:
                print(f"[WARN] extra '{key}' missing at step {k}")

    # Close the env cleanly
    print("Playback complete. Closing environment.")
    env.close()

    # 7) Save tensors (.pt) and a compact CSV for quick plotting
    save_csv_path = None
    if args_cli.save_data:
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_folder = os.path.join(SAVE_DIR, SAVE_NAME)
        os.makedirs(save_folder, exist_ok=True)

        # Auto-increment file suffix to avoid clobbering
        base = os.path.join(save_folder, f"{args_cli.save_name}")
        suffix, i = "", 0
        while True:
            save_pt_path = f"{base}{suffix}.pt"
            save_csv_path = f"{base}{suffix}.csv"
            if not os.path.exists(save_pt_path) and not os.path.exists(save_csv_path):
                break
            suffix = f"_{i}"; i += 1

        # Stack tensors where possible
        for k in list(data.keys()):
            try:
                data[k] = torch.stack(data[k], dim=0)
            except Exception as e:
                print(f"[WARN] Could not stack '{k}': {e}")

        torch.save(data, save_pt_path)
        print(f"[INFO] Saved tensors to: {save_pt_path}")

        # Build compact CSV with consistent columns
        # Extract arrays (trim to common length)
        cols = ["actions", "pos_xy", "theta", "vel_x", "vel_y", "yaw_rate", "s_idx", "time"]
        arrays, lengths = {}, []
        for c in cols:
            t = data.get(c)
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu().squeeze()
                arrays[c] = t
                # length contribution (pos_xy excluded from length only if weird shape)
                lengths.append(t.shape[0])
            else:
                print(f"[WARN] '{c}' not present as tensor; skipping in CSV.")

        min_len = min(lengths) if lengths else 0
        df_dict = {}
        if min_len > 0:
            # Scalars per time
            df_dict["time"] = arrays["time"][:min_len].numpy().astype(float)
            df_dict["theta_rad"] = arrays["theta"][:min_len].numpy().astype(float)
            df_dict["vx_mps"] = arrays["vel_x"][:min_len].numpy().astype(float)
            df_dict["vy_mps"] = arrays["vel_y"][:min_len].numpy().astype(float)
            df_dict["psi_radps"] = arrays["yaw_rate"][:min_len].numpy().astype(float)
            df_dict["s_idx"] = arrays["s_idx"][:min_len].numpy().astype(float)

            # Position split
            pos = arrays["pos_xy"][:min_len].numpy().astype(float)
            df_dict["x"] = pos[:, 0]
            df_dict["y"] = pos[:, 1]

            # Command split (our overridden normalized actions)
            act = arrays["actions"][:min_len].numpy().astype(float)
            df_dict["speed_cmd"] = act[:, 0]  # normalized
            df_dict["steering_cmd"] = act[:, 1]  # normalized

            # Add *resampled* real signals aligned to sim time for convenience
            t_sim = df_dict["time"]
            for col, out_name in [
                ("x", "real_x_m"), ("y", "real_y_m"),
                ("vx", "real_vx_mps"), ("vy", "real_vy_mps"),
                ("theta", "real_theta_rad"),
                ("omega", "real_psi_radps"),
                ("cmd_velocity", "real_speed_cmd"),
                ("cmd_steering_angle", "real_steering_cmd"),
            ]:
                if col in real_df.columns:
                    df_dict[out_name] = resample_1d(real_df["time"].values, real_df[col].values, t_sim)

            # Final DataFrame and write
            df_out = pd.DataFrame(df_dict)
            df_out.to_csv(save_csv_path, index=False)
            print(f"[INFO] Saved compact CSV to: {save_csv_path}")

    # 8) Plot comparisons (only if CSV exists)
    if save_csv_path is not None and os.path.exists(save_csv_path):
        plot_comparisons(save_csv_path)
    else:
        print("[INFO] Skipping plots (no CSV was written).")


# ----------------------------
# Plotting utilities
# ----------------------------
def plot_comparisons(csv_path: str):
    """Generate side-by-side plots from the saved CSV (sim vs real)."""
    sim = pd.read_csv(csv_path)

    # Extract sim (already aligned and scaled where needed)
    time     = sim["time"].to_numpy()
    x, y     = sim["x"].to_numpy(), sim["y"].to_numpy()
    vx, vy   = sim["vx_mps"].to_numpy(), sim["vy_mps"].to_numpy()
    omega    = sim["psi_radps"].to_numpy()
    theta    = sim["theta_rad"].to_numpy()
    s_idx    = sim["s_idx"].to_numpy()
    # Actions are normalized; keep that in mind when interpreting
    speed_cmd_norm = sim["speed_cmd"].to_numpy()
    steer_cmd_norm = sim["steering_cmd"].to_numpy()

    # If real channels present, pull them
    has_real = "real_x_m" in sim.columns
    if has_real:
        rx, ry = sim["real_x_m"].to_numpy(), sim["real_y_m"].to_numpy()
        rvx = sim.get("real_vx_mps", pd.Series(np.nan, index=sim.index)).to_numpy()
        rvy = sim.get("real_vy_mps", pd.Series(np.nan, index=sim.index)).to_numpy()
        rtheta = sim.get("real_theta_rad", pd.Series(np.nan, index=sim.index)).to_numpy()
        romega = sim.get("real_psi_radps", pd.Series(np.nan, index=sim.index)).to_numpy()
        rcmd_v = sim.get("real_speed_cmd", pd.Series(np.nan, index=sim.index)).to_numpy()
        rcmd_s = sim.get("real_steering_cmd", pd.Series(np.nan, index=sim.index)).to_numpy()
    else:
        rx = ry = rvx = rvy = rtheta = romega = rcmd_v = rcmd_s = None

    # Load config for de-normalization (plot in physical units)
    from wheeledlab_tasks.config_loader import load_config
    CONFIG = load_config()

    # De-normalize commands to physical units for plots
    speed_cmd = speed_cmd_norm * CONFIG["env_config"]["MAX_SPEED_SCALING"]
    steer_cmd = steer_cmd_norm * CONFIG["env_config"]["MAX_STEERING_ANGLE"]

    # Derived
    slip_angle = np.arctan2(vy, vx)

    # ---- Figure 1: Commands and states ----
    plt.figure(figsize=(15, 14))

    # (1) Commands
    ax1 = plt.subplot(4, 1, 1)
    ax1b = ax1.twinx()
    ax1.plot(time, speed_cmd, label="Sim cmd velocity")
    if has_real and np.isfinite(rcmd_v).any():
        ax1.plot(time, rcmd_v, "--", label="Real cmd velocity")
    ax1b.plot(time, steer_cmd, color="tab:blue", label="Sim cmd steering")
    if has_real and np.isfinite(rcmd_s).any():
        ax1b.plot(time, rcmd_s, "--", color="tab:blue", label="Real cmd steering")
    ax1.set_ylabel("Velocity [m/s]")
    ax1b.set_ylabel("Steering [rad]")
    ax1.grid(True)
    l1, n1 = ax1.get_legend_handles_labels()
    l1b, n1b = ax1b.get_legend_handles_labels()
    ax1.legend(l1 + l1b, n1 + n1b, loc="upper right")
    ax1.set_title("Command velocity & steering (sim vs real)")

    # (2) Linear velocities
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.plot(time, vx, label="vx (sim)")
    ax2.plot(time, vy, label="vy (sim)")
    if has_real and np.isfinite(rvx).any():
        ax2.plot(time, rvx, "--", label="vx (real)")
    if has_real and np.isfinite(rvy).any():
        ax2.plot(time, rvy, "--", label="vy (real)")
    ax2.set_ylabel("Velocity [m/s]")
    ax2.grid(True); ax2.legend()
    ax2.set_title("Linear velocities")

    # (3) Yaw rate & heading
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3b = ax3.twinx()
    ax3.plot(time, omega, label="yaw rate ω (sim)")
    if has_real and np.isfinite(romega).any():
        ax3.plot(time, romega, "--", label="yaw rate ω (real)")
    ax3b.plot(time, theta, color="tab:red", label="heading θ (sim)")
    if has_real and np.isfinite(rtheta).any():
        ax3b.plot(time, rtheta, "--", color="tab:red", label="heading θ (real)")
    ax3.set_ylabel("ω [rad/s]")
    ax3b.set_ylabel("θ [rad]")
    ax3.grid(True)
    l3, n3 = ax3.get_legend_handles_labels()
    l3b, n3b = ax3b.get_legend_handles_labels()
    ax3.legend(l3 + l3b, n3 + n3b, loc="upper right")
    ax3.set_title("Yaw rate & heading")

    # (4) Slip angle
    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    ax4.plot(time, slip_angle, label="β (sim)")
    if has_real and np.isfinite(rvx).any() and np.isfinite(rvy).any():
        ax4.plot(time, np.arctan2(rvy, rvx), "--", label="β (real)")
    ax4.set_xlabel("Time [s]"); ax4.set_ylabel("Slip [rad]")
    ax4.grid(True); ax4.legend()
    ax4.set_title("Slip angle")
    plt.tight_layout(); plt.show()

    # ---- Figure 2: Trajectory overlay ----
    if has_real:
        # Translate both to start at (0,0) for clearer overlay
        sim_xy = np.column_stack([x, y]).copy()
        sim_xy -= sim_xy[0]
        real_xy = np.column_stack([rx, ry]).copy()
        real_xy -= real_xy[0]

        # Optionally rotate sim to better match real heading
        dx, dy = sim_xy[1] - sim_xy[0]
        sim_heading = np.arctan2(dy, dx)
        dxr, dyr = real_xy[min(20, len(real_xy)-1)] - real_xy[0]  # stable initial heading
        real_heading = np.arctan2(dyr, dxr)
        angle = (real_heading - sim_heading) + ROTATE_TRAJ_ANGLE
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        sim_xy_rot = (R @ sim_xy.T).T

        plt.figure(figsize=(10, 8))
        plt.plot(sim_xy_rot[:, 0], sim_xy_rot[:, 1], label="Sim traj")
        plt.plot(real_xy[:, 0], real_xy[:, 1], "--", label="Real traj")
        plt.scatter(sim_xy_rot[0, 0], sim_xy_rot[0, 1], s=80, label="Sim start")
        plt.scatter(real_xy[0, 0], real_xy[0, 1], s=80, label="Real start")
        plt.xlabel("X [m]"); plt.ylabel("Y [m]")
        plt.title("Trajectory overlay (translated & rotated)")
        plt.axis("equal"); plt.grid(True); plt.legend(); plt.show()


if __name__ == "__main__":
    main()
