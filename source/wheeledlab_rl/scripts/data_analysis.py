import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# List of policies to compare
# log_LIST = [
#     'glad-microwave-949',
#     'earnest-deluge-952',
# ]
# increasing dynamic friction

# log_LIST = [
#     'glad-microwave-949',
#     'earnest-deluge-952',
#     'honest-dragon-953',
#     'curious-galaxy-954',
#     'brisk-butterfly-955'

# ]
LOG_LIST = [
    'test_3',
    'test_2'
]
# Color cycle for different policies
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Analysis parameters
start_idx = 0 
end_idx = 1000

# Create dictionary to store all data
log_data = {}

# Load data for each log
for log in LOG_LIST:
    try:
        data_path = f"/home/tongo/WheeledLab/source/wheeledlab_rl/logs_play_policy/{log}.pt"
        data = torch.load(data_path)
        
        # Convert to numpy and store
        log_data[log] = {
            'actions': data['actions'].cpu().numpy(),
            'observations': data['observations'].cpu().numpy(),
            'time': data['time'].cpu().numpy(),
            's_idx': torch.squeeze(data['s_idx']).cpu().numpy(),
            'pos_xy': torch.squeeze(data['pos_xy']).cpu().numpy()
        }
        
        # Try loading bounds if they exist
        try:
            log_data[log]['inner'] = torch.squeeze(data['inner_bounds']).cpu().numpy()
            log_data[log]['outer'] = torch.squeeze(data['outer_bounds']).cpu.numpy()
        except:
            pass
            
    except FileNotFoundError:
        print(f"Warning: Could not load data for log {log}")
        continue

# Plot Actions Comparison
plt.figure(figsize=(12, 6))
for i, (log, data) in enumerate(log_data.items()):
    color = COLORS[i % len(COLORS)]
    for env_idx in range(data['actions'].shape[1]):
        plt.plot(data['time'][start_idx:end_idx], 
                data['actions'][start_idx:end_idx, env_idx, 0], 
                color=color, 
                alpha=0.7,
                label=f'{log} Throttle' if env_idx == 0 else None)
        plt.plot(data['time'][start_idx:end_idx], 
                data['actions'][start_idx:end_idx, env_idx, 1], 
                '--', 
                color=color,
                alpha=0.7,
                label=f'{log} Steering' if env_idx == 0 else None)
plt.xlabel("Time [s]")
plt.ylabel("Action Value")
plt.title("log Actions Comparison")
plt.legend()
plt.grid()
plt.show()

# Plot Velocity and Acceleration Comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

for i, (log, data) in enumerate(log_data.items()):
    color = COLORS[i % len(COLORS)]
    time = data['time'][start_idx:end_idx]
    obs = data['observations'][start_idx:end_idx]
    
    # Calculate velocity and acceleration
    vel = np.sqrt(np.square(obs[:, 0, 0]) + np.square(obs[:, 0, 1]))
    dt = np.diff(time)
    acceleration = np.diff(vel) / dt
    acceleration = np.append(acceleration, np.nan)
    
    # Plot velocities
    ax1.plot(time, vel, color=color, label=log)
    
    # Plot acceleration
    ax2.plot(time, acceleration, color=color, label=log)

# Format velocity plot
ax1.set_ylabel("Velocity [m/s]")
ax1.legend()
ax1.grid()

# Format acceleration plot
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Acceleration [m/s²]")
ax2.legend()
ax2.grid()
ax2.set_ylim(0, 2)

plt.suptitle("Velocity and Acceleration Comparison")
plt.tight_layout()
plt.show()

# Plot Trajectory Comparison
plt.figure(figsize=(10, 10))

# Plot bounds (from first available log)
for log, data in log_data.items():
    try:
        plt.scatter(data['inner'][:,0], data['inner'][:,1], color='black')
        plt.scatter(data['outer'][:,0], data['outer'][:,1], color='black')
        break  # Just need to plot bounds once
    except:
        continue

# Plot trajectories
for i, (log, data) in enumerate(log_data.items()):
    color = COLORS[i % len(COLORS)]
    pos_xy = data['pos_xy'][start_idx:end_idx]
    vel = np.sqrt(np.square(data['observations'][start_idx:end_idx, 0, 0]) + 
                np.square(data['observations'][start_idx:end_idx, 0, 1]))
    
    plt.scatter(pos_xy[0,0], pos_xy[0,1], color=color, marker='x', s=100)
    plt.scatter(pos_xy[-1,0], pos_xy[-1,1], color=color, marker='x', s=100)
    sc = plt.scatter(pos_xy[:,0], pos_xy[:,1], 
                    c=vel, 
                    cmap='viridis', 
                    alpha=0.7,
                    label=log)

# Create custom legend
legend_elements = [Line2D([0], [0], marker='o', color='w', label=log,
                  markerfacecolor=COLORS[i], markersize=10) 
                  for i, log in enumerate(log_data.keys())]
plt.legend(handles=legend_elements)

plt.title("Trajectory Comparison")
plt.grid()
plt.show()