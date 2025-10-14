<img src="docs/media/fig1.png" alt="fig1" />

---

# WheeledLab

[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.0.2-silver.svg)](https://isaac-sim.github.io/IsaacLab/v2.0.0/)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)

Environments, assets, workflow for open-source mobile robotics, integrated with IsaacLab.

[Website](https://uwrobotlearning.github.io/WheeledLab/) | [Paper](https://arxiv.org/abs/2502.07380)

## Installing IsaacLab (~20 min)

Note: Only use this pip installation approach if you're on Ubuntu 22.04+ or Windows. For Ubuntu 20.04, install from the binaries. [link](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

WheeledLab is built atop Isaac Lab. If you do not yet have Isaac Lab installed, it is open-source and installation instructions for Isaac Sim v4.5.0 and Isaac Lab v2.0.2 can be found below:

```bash
# Create a conda environment named WL and install Isaac Sim v4.5.0 in it:
conda create -n WL python=3.10
conda activate WL
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 # Or `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118` for CUDA 11
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# Install Isaac Lab v2.0.2 (make sure you have build dependencies first, e.g. `sudo apt install cmake build-essential` on ubuntu)
git clone --branch v2.0.2 https://github.com/isaac-sim/IsaacLab.git
./isaaclab.sh -i
```

Source: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

If you already have IsaacLab you can skip this and instead head [here](#create-new-isaaclab-conda-environment) to set up a new conda environment for this repository.

### Create New IsaacLab Conda Environment

We recommend setting up a new [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) environment to include both `IsaacLab` packages and `WheeledLab` packages. You can do this using Isaac Lab's convenient setup scripts:

```bash
cd <IsaacLab>
./isaaclab.sh --conda WL
conda activate WL
./isaaclab.sh -i
```

## Installing WheeledLab (~5 min)

```bash
# Activate the conda environment that was created via the IsaacLab setup.
conda activate <your IsaacLab env here> # 'WL' if you followed instructions above

git clone git@github.com:UWRobotLearning/WheeledLab.git
cd WheeledLab/source
pip install -e wheeledlab
pip install -e wheeledlab_tasks
pip install -e wheeledlab_assets
pip install -e wheeledlab_rl
```

After this, we recommend [Setting Up VSCode](https://github.com/UWRobotLearning/WheeledLab?tab=readme-ov-file#training-quick-start).

## Training Quick Start

Training runs can take a couple hours to produce a transferable policy.

To start a drifting run:

```bash
python source/wheeledlab_rl/scripts/train_rl.py --headless -r RSS_DRIFT_CONFIG
```

To start a elevation run:

```bash
python source/wheeledlab_rl/scripts/train_rl.py --headless -r RSS_ELEV_CONFIG
```

To start a visual run:

```bash
python source/wheeledlab_rl/scripts/train_rl.py --headless -r RSS_VISUAL_CONFIG
```

Though optional (and free), we strongly advise using [Weights & Biases](https://wandb.ai/site/) (`wandb`) to record and track training status. Logging to `wandb` is turned on by default. If you would like to disable it, add `train.log.no_wandb=True` to the CLI arguments.

See more details about training in the `wheeledlab_rl` [README.md](source/wheeledlab_rl/docs/README.md)

## Deployment

A separate repository is maintained for existing integrations and deployments. See https://github.com/UWRobotLearning/RealLab for code.

### Current Integrations

1. HOUND [1] - https://github.com/UWRobotLearning/RealLab/tree/hound
2. MuSHR [2] - https://github.com/UWRobotLearning/RealLab/tree/mushr
3. F1Tenth [3] - (coming soon)

If you have an integration or request for a platform not seen above, please contact us or consider contributing!

## Setting Up VSCode

It is a million times harder to develop in IsaacLab without Intellisense. Setting up the vscode workspace is
STRONGLY advised.

0. Find where your `IsaacLab` directory currently is. We'll refer to it as `<IsaacLab>` in this section. Move the VSCode tools to this workspace.

   ```bash
   cd <WheeledLab>
   cp -r <IsaacLab>/.vscode/tools ./.vscode/
   cp -r <IsaacLab>/.vscode/*.json ./.vscode/
   ```

1. Change `.vscode/tasks.json` line 11

   ```json
   "command": "${workspaceFolder}/../IsaacLab/isaaclab.sh -p ${workspaceFolder}/.vscode/tools/setup_vscode.py"
   ```

   to

   ```json
   "command": "<IsaacLabDir>/isaaclab.sh -p ${workspaceFolder}/.vscode/tools/setup_vscode.py"
   ```

2. `Ctrl` + `Shift` + `P` to bring up the VSCode command palette. type `Tasks:Run Task` or type until you see it show up and highlight it and press `Enter`.
3. Click on `setup_python_env`. Follow the prompts until you're able to run the task. You should see a console at the bottom and the status of the task.
4. If successful, you should now have `.vscode/{settings.json, launch.json}` in your `<WheeledLab>` repo and `settings.json` should have a populated list of paths under the `"python.analysis.extraPaths"` key.
5. Make sure you at least have Microsoft's Python extension installed for intellisense to work. 

### If it still doesn't work

The `setup_vscode` task doesn't work for me for whatever reason. If that's true for you too, add the following lines to the end of the list under the key `"python.analysis.extraPaths"` in the `.vscode/settings.json` file:

```json
    "<IsaacLab>/source/isaaclab",
    "<IsaacLab>/source/isaaclab_assets",
    "<IsaacLab>/source/isaaclab_tasks",
    "<IsaacLab>/source/isaaclab_rl",
```

## References

### This work

```
@misc{2502.07380,
Author = {Tyler Han and Preet Shah and Sidharth Rajagopal and Yanda Bao and Sanghun Jung and Sidharth Talia and Gabriel Guo and Bryan Xu and Bhaumik Mehta and Emma Romig and Rosario Scalise and Byron Boots},
Title = {Demonstrating WheeledLab: Modern Sim2Real for Low-cost, Open-source Wheeled Robotics},
Year = {2025},
Eprint = {arXiv:2502.07380},
}
```

### Cited

[1] Sidharth Talia, Matt Schmittle, Alexander Lambert, Alexander Spitzer, Christoforos Mavrogiannis, and Siddhartha S. Srinivasa.Demonstrating HOUND: A Low-cost Research Platform for High-speed Off-road Underactuated Nonholonomic Driving, July 2024.URL http://arxiv.org/abs/2311.11199.arXiv:2311.11199 [cs].

[2] Siddhartha S. Srinivasa, Patrick Lancaster, Johan Michalove, Matt Schmittle, Colin Summers, Matthew Rockett, Rosario Scalise, Joshua R. Smith, Sanjiban Choudhury, Christoforos Mavrogiannis, and Fereshteh Sadeghi.MuSHR: A Low-Cost, Open-Source Robotic Racecar for Education and Research, December 2023.URL http://arxiv.org/abs/1908.08031.arXiv:1908.08031 [cs].

[3] Matthew O’Kelly, Hongrui Zheng, Dhruv Karthik, and Rahul Mangharam. F1TENTH: An Open-source Eval- uation Environment for Continuous Control and Reinforcement Learning. In Proceedings of the NeurIPS 2019 Competition and Demonstration Track, pages 77– 89. PMLR, August 2020. URL https://proceedings.mlr. press/v123/o-kelly20a.html. ISSN: 2640-3498.

## NEW General Workflow F1TENTH Time Trial and Overtaking

The addition of this fork have been the new time trial and overtaking.
 
Before running any training, is it important to check the config file f1tenth_config.yaml (source/wheeledlab_tasks/wheeledlab_tasks/f1tenth/config/f1tenth_config.yaml)

### Training time-trial
To run a quick training run (if you don't specify the name it will be save with the timestamp)

```
python source/wheeledlab_rl/scripts/train_rl.py -r RSS_TIMETRIAL --headless train.log.run_name=SPECIFY_RUN_NAME
```
NB. that you can also run not headless but for many environments (e.g. 4096) it is suggested headless. Otherwise you need a display (see remote-novnc)

### Training overtaking


```
python source/wheeledlab_rl/scripts/train_rl.py -r RSS_OVERTAKE --headless train.log.run_name=SPECIFY_RUN_NAME
```

The model and logs should be stored in 


```
source/wheeledlab_rl/logs/
```

The overtaking task can still be improved. For example it is very sensible to its parameters in the "f1tenth_config.yaml". If after many episodes the agent is still not learning, it might be because the episode terminates too quickly. A solution is to increase "OPPONENT_ALWAYS_AHEAD_PERCENTAGE" that defines the percentage in [-] of opponents that moves in front of the agent (they are always ahead, donkey + carrot scenario). Similarly it is always suggested to start with "OPP_INIT_VEL_SCALING" close to 0. Another solution is to have the opponent static for a initial number of "STATIC_OPPONENT_UNTIL_EP" episodes.

Even when learning, the agent struggle to learn a policy reproducible in the real car, due to its very oscillatory steering (even with action regularization). Furhtermore, it struggles to learn never crashing to the opponent, without exploiting the reward functions (e.g. it might never crash to the opponent, but by learning just to trail behind it without attempting overtaking).

To improve the agent performance, it would be interesting to try add predicted trajectory information (+ with uncertainty/noise) of the opponent trajectory as observation feature, it might help the agent to learn smoother and more decisive trajectories.

### Playing trained policies


Once you are done with training, you can test and play the trained policy via 

```
python source/wheeledlab_rl/scripts/play_policy.py 
```

IMPORTANT: when you play a policy, the observation features (horizon length, history buffer length) set in "f1tenth_overtake_env_cfg.py"/"f1tenth_timetrial_env_cfg.py" must be the same of the ones used in training. However you might change other parameters in the "f1tenth_config.yaml" to study how the trained policy perform in a different environment (friction and mass randomization).
To improve reproducibility and tidiness of the code, the "f1tenth_config.yaml" used in each run should be stored in the logs of the training.

### Playing rosbag data in simulation

You can compare real data with simulation (e.g. for system identification).
N.B. In "f1tenth_config.yaml" the parameters "INCREMENTAL_MODE" and "NON_TRAVERSABLE_TERMINATION" must be set to false! 
N.B. You can also adjust "EPISODE_LENGTH_S_*" accordingly.

```
python source/wheeledlab_rl/scripts/compare_sim_real.py 
```

### Code parts to improve (1) - Logging and plotting data to "play_policy.py" and "compare_sim_real.py"

In "rewards.py" are stored environment information in the following way

```
    # ---- extras for play_policy.py / WandB (unchanged behavior) ----
    env.extras["theta"] = asset.data.heading_w
    env.extras["pos_xy"] = position_xy_world
    env.extras["vel_x"] = asset.data.root_lin_vel_b[:, 0]
    env.extras["vel_y"] = -asset.data.root_lin_vel_b[:, 1]
    env.extras["target_velocity"] = env._target_velocity_history[:, 0].clone()
    env.extras["target_steering"] = env._target_steering_angle_history[:, 0].clone()
    env.extras["yaw_rate"] = asset.data.root_ang_vel_b[:, 2]
    env.extras["s_idx"] = current_idx.clone()
    env.extras["time"] = torch.tensor(env.sim.current_time, device=env.device)
    env.extras["s_idx_max"] = torch.tensor(num_waypoints, device=env.device)
```

which are used and plotted later in "play_policy.py" and "compare_sim_real.py". 
Similarly informations are logged to W&B (in "rewards.py" and terminations.py")

```
    env.extras["log"]["Info/max_slip_angle"] = torch.max(slip_angle * moving_mask)
    env.extras["log"]["Info/mean_slip_angle"] = torch.mean(slip_angle * moving_mask)
    env.extras["log"]["Info/mean_speed"] = torch.mean(env._base_lin_vel_x_history)
    env.extras["log"]["Info/max_speed"] = torch.max(env._base_lin_vel_x_history)
```

Everytime the function in which they reside is called (every decimation step), these information are stored. But it might easily brake if e.g. the function is not called or called multiple times.

### Code parts to improve (2) - Maps initialization

Overall the initialization of the terrain is quite messy (but works). The part addressed are the initialization of the class "F1TenthOvertakeTerrainImporterCfg" and the functions in "maps_utils.py", where basically the information from the classic map file "global_waypoints.json" are transferred and stored as class variables.

### Code parts to improve (3) - Log f1tenth_config.yaml

To improve reproducibility and tidiness of the code, the "f1tenth_config.yaml" used in each run should be stored in the logs of the training. Much better than writing down run information in the run title.
