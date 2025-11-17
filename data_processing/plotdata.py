import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import math


def quaternion_to_euler(q):
  """
  Convert quaternion to Euler angles (roll, pitch, yaw).
  
  Args:
    q: Quaternion object with w, x, y, z attributes
  
  Returns:
    roll, pitch, yaw in radians
  """
  # Extract quaternion components
  w, x, y, z = q.w, q.x, q.y, q.z
  
  # Roll (rotation around x-axis)
  sinr_cosp = 2 * (w * x + y * z)
  cosr_cosp = 1 - 2 * (x * x + y * y)
  roll = math.atan2(sinr_cosp, cosr_cosp)
  
  # Pitch (rotation around y-axis)
  sinp = 2 * (w * y - z * x)
  if abs(sinp) >= 1:
    pitch = math.copysign(math.pi / 2, sinp)
  else:
    pitch = math.asin(sinp)
  
  # Yaw (rotation around z-axis)
  siny_cosp = 2 * (w * z + x * y)
  cosy_cosp = 1 - 2 * (y * y + z * z)
  yaw = math.atan2(siny_cosp, cosy_cosp)
  
  return roll, pitch, yaw


def plot_euler_angles(quaternions, covariances=None, save_dir='results', show=True):
  """
  Plot roll, pitch, yaw over time from quaternion estimates with covariance bounds.
  Creates a single figure with 3 vertically stacked subplots.
  
  Args:
    quaternions: List of Quaternion objects
    covariances: Optional list of 15x15 covariance matrices from MEKF (default: None)
                 If provided, extracts orientation error covariance and plots bounds
    save_dir: Directory to save plots (default: 'results')
    show: If True, display plots interactively (default: True)
  """
  import os
  os.makedirs(save_dir, exist_ok=True)
  
  # Convert quaternions to Euler angles
  euler_angles = [quaternion_to_euler(q) for q in quaternions]
  
  # Extract individual angles and convert to degrees
  t = np.arange(len(euler_angles))
  roll = np.array([np.degrees(e[0]) for e in euler_angles])
  pitch = np.array([np.degrees(e[1]) for e in euler_angles])
  yaw = np.array([np.degrees(e[2]) for e in euler_angles])
  
  # Extract covariance bounds if available
  roll_ub = None
  roll_lb = None
  pitch_ub = None
  pitch_lb = None
  yaw_ub = None
  yaw_lb = None
  
  if covariances is not None:
        # covariances is shape (15, 15, n_steps) - extract for each timestep
    try:
      n_steps = covariances.shape[2] if len(covariances.shape) == 3 else len(covariances)
      
      if n_steps == len(euler_angles):
        # Extract orientation error covariance diagonal (first 3 elements)
        # Note: MEKF error state covariance is in rad, convert bounds to degrees
        if len(covariances.shape) == 3:
          # 3D array: (15, 15, n_steps)
          roll_std = np.array([np.degrees(np.sqrt(max(0, covariances[0, 0, i]))) for i in range(n_steps)])
          pitch_std = np.array([np.degrees(np.sqrt(max(0, covariances[1, 1, i]))) for i in range(n_steps)])
          yaw_std = np.array([np.degrees(np.sqrt(max(0, covariances[2, 2, i]))) for i in range(n_steps)])
        else:
          # List of 2D arrays
          roll_std = np.array([np.degrees(np.sqrt(max(0, covariances[i][0, 0]))) for i in range(n_steps)])
          pitch_std = np.array([np.degrees(np.sqrt(max(0, covariances[i][1, 1]))) for i in range(n_steps)])
          yaw_std = np.array([np.degrees(np.sqrt(max(0, covariances[i][2, 2]))) for i in range(n_steps)])
        
        # Compute 3-sigma bounds (3 * std)
        roll_ub = roll + 3.0 * roll_std
        roll_lb = roll - 3.0 * roll_std
        pitch_ub = pitch + 3.0 * pitch_std
        pitch_lb = pitch - 3.0 * pitch_std
        yaw_ub = yaw + 3.0 * yaw_std
        yaw_lb = yaw - 3.0 * yaw_std
    except Exception as e:
      print(f"Warning: Could not extract covariance bounds: {e}")
      pass
  
  # Create figure with 3 subplots stacked vertically
  fig, axes = plt.subplots(3, 1, figsize=(12, 10))
  fig.suptitle('MEKF Attitude Estimates - Euler Angles', fontsize=16, fontweight='bold')
  
  # Plot Roll
  axes[0].plot(t, roll, linestyle='-', color='blue', linewidth=2, label='Roll Estimate')
  if roll_ub is not None:
    axes[0].plot(t, roll_ub, linestyle='--', color='cyan', linewidth=1, label='3σ Bounds')
    axes[0].plot(t, roll_lb, linestyle='--', color='cyan', linewidth=1)
    axes[0].fill_between(t, roll_ub, roll_lb, color='lightblue', alpha=0.3)
  axes[0].set_ylabel('Roll (degrees)', fontsize=11, fontweight='bold')
  axes[0].grid(True, alpha=0.3)
  axes[0].legend(loc='upper right')
  axes[0].set_title('Roll (φ) - Rotation around X-axis', fontsize=10)
  
  # Plot Pitch
  axes[1].plot(t, pitch, linestyle='-', color='green', linewidth=2, label='Pitch Estimate')
  if pitch_ub is not None:
    axes[1].plot(t, pitch_ub, linestyle='--', color='lightgreen', linewidth=1, label='3σ Bounds')
    axes[1].plot(t, pitch_lb, linestyle='--', color='lightgreen', linewidth=1)
    axes[1].fill_between(t, pitch_ub, pitch_lb, color='lightgreen', alpha=0.3)
  axes[1].set_ylabel('Pitch (degrees)', fontsize=11, fontweight='bold')
  axes[1].grid(True, alpha=0.3)
  axes[1].legend(loc='upper right')
  axes[1].set_title('Pitch (θ) - Rotation around Y-axis', fontsize=10)
  
  # Plot Yaw
  axes[2].plot(t, yaw, linestyle='-', color='red', linewidth=2, label='Yaw Estimate')
  if yaw_ub is not None:
    axes[2].plot(t, yaw_ub, linestyle='--', color='salmon', linewidth=1, label='3σ Bounds')
    axes[2].plot(t, yaw_lb, linestyle='--', color='salmon', linewidth=1)
    axes[2].fill_between(t, yaw_ub, yaw_lb, color='mistyrose', alpha=0.3)
  axes[2].set_xlabel('Time (steps)', fontsize=11, fontweight='bold')
  axes[2].set_ylabel('Yaw (degrees)', fontsize=11, fontweight='bold')
  axes[2].grid(True, alpha=0.3)
  axes[2].legend(loc='upper right')
  axes[2].set_title('Yaw (ψ) - Rotation around Z-axis', fontsize=10)
  
  # Adjust layout
  plt.tight_layout()
  
  # Save figure
  plt.savefig(f'{save_dir}/euler_angles.png', dpi=150, bbox_inches='tight')
  print(f"✓ Saved: {save_dir}/euler_angles.png")
  
  if show:
    print("Displaying Euler angles plot...\n")
    plt.show()
  else:
    plt.close(fig)

def make_ellipse(mean, cov):
  vals, vecs = np.linalg.eigh(cov)
  sort = vals.argsort()[::-1]
  vals = vals[sort]
  vecs = vecs[:, sort]
  theta = np.linspace(0, 2*np.pi, 100)
  phi = np.linspace(0, np.pi, 100)
  xs = np.outer(np.cos(theta), np.sin(phi))
  ys = np.outer(np.sin(theta), np.sin(phi))
  zs = np.outer(np.ones_like(theta), np.cos(phi))

  sphere = np.row_stack((xs, ys, zs))

  a = 1.96 * np.sqrt(vals)

  ellipsoid = mean[:,np.newaxis] + vecs @ (np.diag(a) @ sphere)

  return ellipsoid[0].reshape(xs.shape), ellipsoid[1].reshape(ys.shape), ellipsoid[2].reshape(zs.shape)

def animate(i, x, y, z):
  line.set_data_3d(x[:i], y[:i], z[:i])
  point.set_data_3d(x[i], y[i], z[i])
  return line, point

def plot_results(state, cov, save_dir='results', show=True):
  """
  Plot EKF results - saves to PNG and displays interactively.
  
  Args:
    state: List of 3-element position vectors
    cov: List of 3x3 covariance matrices
    save_dir: Directory to save plots (default: 'results')
    show: If True, display plots interactively after saving (default: True)
  """
  # Create results directory if it doesn't exist
  import os
  os.makedirs(save_dir, exist_ok=True)
  
  t = np.arange(len(state))
  x = np.array([matrix[0] for matrix in state])
  y = np.array([matrix[1] for matrix in state])
  z = np.array([matrix[2] for matrix in state])
  x_cov = np.array([matrix[0, 0] for matrix in cov])
  y_cov = np.array([matrix[1, 1] for matrix in cov])
  z_cov = np.array([matrix[2, 2] for matrix in cov])
  x_up = x + 3.0 * np.sqrt(x_cov)
  x_low = x - 3.0 * np.sqrt(x_cov)
  y_up = y + 3.0 * np.sqrt(y_cov)
  y_low = y - 3.0 * np.sqrt(y_cov)
  z_up = z + 3.0 * np.sqrt(z_cov)
  z_low = z - 3.0 * np.sqrt(z_cov)

  # Plot all three position components in a single figure with 3 stacked subplots
  fig, axes = plt.subplots(3, 1, figsize=(12, 10))
  fig.suptitle('EKF Position Estimates', fontsize=16, fontweight='bold')
  
  # Plot x position
  axes[0].plot(t, x, linestyle='-', color='blue', linewidth=1.5, label='X Estimate')
  axes[0].plot(t, x_up, linestyle='--', color='cyan', linewidth=1, label='3σ Bounds')
  axes[0].plot(t, x_low, linestyle='--', color='cyan', linewidth=1)
  axes[0].fill_between(t, x_up, x_low, color='lightblue', alpha=0.3)
  axes[0].set_ylabel('X Position (meters)', fontsize=11, fontweight='bold')
  axes[0].set_title('X Position vs. Time', fontsize=10)
  axes[0].grid(True, alpha=0.3)
  axes[0].legend(loc='upper right')
  
  # Plot y position
  axes[1].plot(t, y, linestyle='-', color='green', linewidth=1.5, label='Y Estimate')
  axes[1].plot(t, y_up, linestyle='--', color='lightgreen', linewidth=1, label='3σ Bounds')
  axes[1].plot(t, y_low, linestyle='--', color='lightgreen', linewidth=1)
  axes[1].fill_between(t, y_up, y_low, color='lightgreen', alpha=0.3)
  axes[1].set_ylabel('Y Position (meters)', fontsize=11, fontweight='bold')
  axes[1].set_title('Y Position vs. Time', fontsize=10)
  axes[1].grid(True, alpha=0.3)
  axes[1].legend(loc='upper right')
  
  # Plot z position
  axes[2].plot(t, z, linestyle='-', color='red', linewidth=1.5, label='Z Estimate')
  axes[2].plot(t, z_up, linestyle='--', color='salmon', linewidth=1, label='3σ Bounds')
  axes[2].plot(t, z_low, linestyle='--', color='salmon', linewidth=1)
  axes[2].fill_between(t, z_up, z_low, color='mistyrose', alpha=0.3)
  axes[2].set_xlabel('Time (steps)', fontsize=11, fontweight='bold')
  axes[2].set_ylabel('Z Position (meters)', fontsize=11, fontweight='bold')
  axes[2].set_title('Z Position vs. Time', fontsize=10)
  axes[2].grid(True, alpha=0.3)
  axes[2].legend(loc='upper right')
  
  # Adjust layout and save
  plt.tight_layout()
  plt.savefig(f'{save_dir}/positions.png', dpi=150, bbox_inches='tight')
  print(f"✓ Saved: {save_dir}/positions.png")

  #plot trajectory
  figure = plt.figure(figsize=(10, 8))
  traj = figure.add_subplot(111, projection='3d')

  traj.plot(x, y, z, linestyle='-', color='blue', linewidth=2, label='Trajectory')
  traj.scatter(x[0], y[0], z[0], marker='o', color='green', s=100, label='Start')
  traj.scatter(x[-1], y[-1], z[-1], marker='s', color='red', s=100, label='End')
  
  # Skip covariance ellipsoids for computational efficiency (comment out if desired)
  # for i in range(0, len(state), max(1, len(state)//10)):  # Plot every 10th ellipsoid
  #   xe, ye, ze = make_ellipse(state[i], cov[i])
  #   traj.plot_surface(xe, ye, ze, color='gray', alpha=0.1)
    
  traj.set_title("3D Position Trajectory")
  traj.set_xlabel("x position (m)")
  traj.set_ylabel("y position (m)")
  traj.set_zlabel("z position (m)")
  traj.legend()
  plt.savefig(f'{save_dir}/trajectory_3d.png', dpi=150, bbox_inches='tight')
  print(f"✓ Saved: {save_dir}/trajectory_3d.png")

  print(f"\n✓ All plots saved to '{save_dir}/' directory")
  
  if show:
    print("\nDisplaying interactive plots...\n")
    plt.show()
  else:
    plt.close('all')
