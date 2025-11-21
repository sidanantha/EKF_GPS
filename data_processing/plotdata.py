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


def plot_euler_angles(quaternions, covariances=None, save_dir='results', show=True, dt=0.01):
  """
  Plot roll, pitch, yaw over time from quaternion estimates with covariance bounds.
  Creates a single figure with 3 vertically stacked subplots.
  
  Args:
    quaternions: List of Quaternion objects
    covariances: Optional list of 15x15 covariance matrices from MEKF (default: None)
                 If provided, extracts orientation error covariance and plots bounds
    save_dir: Directory to save plots (default: 'results')
    show: If True, display plots interactively (default: True)
    dt: Time step in seconds (default: 0.01)
  """
  import os
  os.makedirs(save_dir, exist_ok=True)
  
  # Convert quaternions to Euler angles
  euler_angles = [quaternion_to_euler(q) for q in quaternions]
  
  # Extract individual angles and convert to degrees
  t = np.arange(len(euler_angles)) * dt
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
  axes[2].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
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

def plot_results(state, cov, save_dir='results', show=True, dt=0.01):
  """
  Plot EKF results in ENU frame - saves to PNG and displays interactively.
  
  Args:
    state: List of 3-element position vectors (East, North, Up)
    cov: List of 3x3 covariance matrices
    save_dir: Directory to save plots (default: 'results')
    show: If True, display plots interactively after saving (default: True)
    dt: Time step in seconds (default: 0.01)
  """
  # Create results directory if it doesn't exist
  import os
  os.makedirs(save_dir, exist_ok=True)
  
  t = np.arange(len(state)) * dt
  east = np.array([matrix[0] for matrix in state])
  north = np.array([matrix[1] for matrix in state])
  up = np.array([matrix[2] for matrix in state])
  east_cov = np.array([matrix[0, 0] for matrix in cov])
  north_cov = np.array([matrix[1, 1] for matrix in cov])
  up_cov = np.array([matrix[2, 2] for matrix in cov])
  east_up = east + 3.0 * np.sqrt(east_cov)
  east_low = east - 3.0 * np.sqrt(east_cov)
  north_up = north + 3.0 * np.sqrt(north_cov)
  north_low = north - 3.0 * np.sqrt(north_cov)
  up_up = up + 3.0 * np.sqrt(up_cov)
  up_low = up - 3.0 * np.sqrt(up_cov)

  # Plot all three position components in a single figure with 3 stacked subplots
  fig, axes = plt.subplots(3, 1, figsize=(12, 10))
  fig.suptitle('EKF Position Estimates (ENU Frame)', fontsize=16, fontweight='bold')
  
  # Plot East position
  axes[0].plot(t, east, linestyle='-', color='blue', linewidth=1.5, label='East Estimate')
  axes[0].plot(t, east_up, linestyle='--', color='cyan', linewidth=1, label='3σ Bounds')
  axes[0].plot(t, east_low, linestyle='--', color='cyan', linewidth=1)
  axes[0].fill_between(t, east_up, east_low, color='lightblue', alpha=0.3)
  axes[0].set_ylabel('East Position (meters)', fontsize=11, fontweight='bold')
  axes[0].set_title('East Position vs. Time', fontsize=10)
  axes[0].grid(True, alpha=0.3)
  axes[0].legend(loc='upper right')
  
  # Plot North position
  axes[1].plot(t, north, linestyle='-', color='green', linewidth=1.5, label='North Estimate')
  axes[1].plot(t, north_up, linestyle='--', color='lightgreen', linewidth=1, label='3σ Bounds')
  axes[1].plot(t, north_low, linestyle='--', color='lightgreen', linewidth=1)
  axes[1].fill_between(t, north_up, north_low, color='lightgreen', alpha=0.3)
  axes[1].set_ylabel('North Position (meters)', fontsize=11, fontweight='bold')
  axes[1].set_title('North Position vs. Time', fontsize=10)
  axes[1].grid(True, alpha=0.3)
  axes[1].legend(loc='upper right')
  
  # Plot Up position
  axes[2].plot(t, up, linestyle='-', color='red', linewidth=1.5, label='Up Estimate')
  axes[2].plot(t, up_up, linestyle='--', color='salmon', linewidth=1, label='3σ Bounds')
  axes[2].plot(t, up_low, linestyle='--', color='salmon', linewidth=1)
  axes[2].fill_between(t, up_up, up_low, color='mistyrose', alpha=0.3)
  axes[2].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
  axes[2].set_ylabel('Up Position (meters)', fontsize=11, fontweight='bold')
  axes[2].set_title('Up Position vs. Time', fontsize=10)
  axes[2].grid(True, alpha=0.3)
  axes[2].legend(loc='upper right')
  
  # Adjust layout and save
  plt.tight_layout()
  plt.savefig(f'{save_dir}/positions.png', dpi=150, bbox_inches='tight')
  print(f"✓ Saved: {save_dir}/positions.png")

  #plot trajectory
  figure = plt.figure(figsize=(10, 8))
  traj = figure.add_subplot(111, projection='3d')

  traj.plot(east, north, up, linestyle='-', color='blue', linewidth=2, label='Trajectory')
  traj.scatter(east[0], north[0], up[0], marker='o', color='green', s=100, label='Start')
  traj.scatter(east[-1], north[-1], up[-1], marker='s', color='red', s=100, label='End')
  
  # Skip covariance ellipsoids for computational efficiency (comment out if desired)
  # for i in range(0, len(state), max(1, len(state)//10)):  # Plot every 10th ellipsoid
  #   xe, ye, ze = make_ellipse(state[i], cov[i])
  #   traj.plot_surface(xe, ye, ze, color='gray', alpha=0.1)
    
  traj.set_title("3D Position Trajectory (ENU Frame)")
  traj.set_xlabel("East position (m)")
  traj.set_ylabel("North position (m)")
  traj.set_zlabel("Up position (m)")
  
  # Set equal aspect ratio based on data ranges
  max_range = np.array([east.max()-east.min(), north.max()-north.min(), up.max()-up.min()]).max() / 2.0
  mid_x = (east.max()+east.min()) * 0.5
  mid_y = (north.max()+north.min()) * 0.5
  mid_z = (up.max()+up.min()) * 0.5
  
  # Handle edge case where all positions are the same (max_range = 0)
  if max_range == 0 or not np.isfinite(max_range):
    max_range = 1.0  # Use default range if data is stationary or contains NaN/Inf
  
  # Only set limits if the midpoints are finite
  if np.isfinite(mid_x) and np.isfinite(mid_y) and np.isfinite(mid_z):
    traj.set_xlim(mid_x - max_range, mid_x + max_range)
    traj.set_ylim(mid_y - max_range, mid_y + max_range)
    traj.set_zlim(mid_z - max_range, mid_z + max_range)
  
  traj.legend()
  plt.savefig(f'{save_dir}/trajectory_3d.png', dpi=150, bbox_inches='tight')
  print(f"✓ Saved: {save_dir}/trajectory_3d.png")

  print(f"\n✓ All plots saved to '{save_dir}/' directory")
  
  if show:
    print("\nDisplaying interactive plots...\n")
    plt.show()
  else:
    plt.close('all')


def plot_attitude_complementary(attitude_storage, attitude_meas_storage, dt, results_dir='results'):
  '''
  Plot complementary filter attitude estimates vs measured attitudes (roll, pitch, yaw) vs time.
  
  Args:
    attitude_storage: Array of shape (3, N) containing complementary filter [roll, pitch, yaw] in radians for N timesteps
    attitude_meas_storage: Array of shape (3, N) containing measured [roll, pitch, yaw] in radians for N timesteps
    dt: Time step in seconds
    results_dir: Directory to save plots
  '''
  import os
  
  os.makedirs(results_dir, exist_ok=True)
  
  # Create time array
  time = np.arange(attitude_storage.shape[1]) * dt
  
  # Extract attitudes and convert to degrees
  roll_comp = np.degrees(attitude_storage[0, :])
  pitch_comp = np.degrees(attitude_storage[1, :])
  yaw_comp = np.degrees(attitude_storage[2, :])
  
  roll_meas = np.degrees(attitude_meas_storage[0, :])
  pitch_meas = np.degrees(attitude_meas_storage[1, :])
  yaw_meas = np.degrees(attitude_meas_storage[2, :])
  
  # Create figure with 3x1 layout
  fig, axes = plt.subplots(3, 1, figsize=(12, 10))
  fig.suptitle('Complementary Filter Attitude Estimates vs Measured', fontsize=16, fontweight='bold')
  
  # Plot Roll
  axes[0].plot(time, roll_meas, linestyle=':', color='cyan', linewidth=2, label='Accel Only')
  axes[0].plot(time, roll_comp, linestyle='-', color='blue', linewidth=2, label='Complementary Filter')
  axes[0].set_ylabel('Roll (degrees)', fontsize=11, fontweight='bold')
  axes[0].set_title('Roll (φ) - Rotation around X-axis', fontsize=11)
  axes[0].grid(True, alpha=0.3)
  axes[0].legend(loc='upper right')
  
  # Plot Pitch
  axes[1].plot(time, pitch_meas, linestyle=':', color='lightgreen', linewidth=2, label='Accel Only')
  axes[1].plot(time, pitch_comp, linestyle='-', color='green', linewidth=2, label='Complementary Filter')
  axes[1].set_ylabel('Pitch (degrees)', fontsize=11, fontweight='bold')
  axes[1].set_title('Pitch (θ) - Rotation around Y-axis', fontsize=11)
  axes[1].grid(True, alpha=0.3)
  axes[1].legend(loc='upper right')
  
  # Plot Yaw
  axes[2].plot(time, yaw_meas, linestyle=':', color='salmon', linewidth=2, label='Accel Only')
  axes[2].plot(time, yaw_comp, linestyle='-', color='red', linewidth=2, label='Complementary Filter')
  axes[2].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
  axes[2].set_ylabel('Yaw (degrees)', fontsize=11, fontweight='bold')
  axes[2].set_title('Yaw (ψ) - Rotation around Z-axis', fontsize=11)
  axes[2].grid(True, alpha=0.3)
  axes[2].legend(loc='upper right')
  
  # Adjust layout
  plt.tight_layout()
  
  # Save figure
  plot_path = os.path.join(results_dir, 'complementary_filter_attitudes.png')
  plt.savefig(plot_path, dpi=150, bbox_inches='tight')
  print(f"✓ Saved: {plot_path}")
  plt.close(fig)


def plot_residuals(prefit_residuals, postfit_residuals, dt, results_dir='results'):
  '''
  Plot pre-fit and post-fit residuals segregated by GPS and IMU measurements.
  
  Args:
    prefit_residuals: Pre-fit residuals array (6 x N), rows 0-2 are GPS (m), rows 3-5 are IMU (m/s^2)
    postfit_residuals: Post-fit residuals array (6 x N), rows 0-2 are GPS (m), rows 3-5 are IMU (m/s^2)
    dt: Time step in seconds
    results_dir: Directory to save plots
  '''
  import os
  
  os.makedirs(results_dir, exist_ok=True)
  
  # Create time array
  time = np.arange(prefit_residuals.shape[1]) * dt
  
  # Extract residuals
  # GPS residuals (rows 0-2, in meters)
  prefit_gps_e = prefit_residuals[0, :]
  prefit_gps_n = prefit_residuals[1, :]
  prefit_gps_u = prefit_residuals[2, :]
  
  postfit_gps_e = postfit_residuals[0, :]
  postfit_gps_n = postfit_residuals[1, :]
  postfit_gps_u = postfit_residuals[2, :]
  
  # IMU residuals (rows 3-5, in m/s^2)
  prefit_imu_e = prefit_residuals[3, :]
  prefit_imu_n = prefit_residuals[4, :]
  prefit_imu_u = prefit_residuals[5, :]
  
  postfit_imu_e = postfit_residuals[3, :]
  postfit_imu_n = postfit_residuals[4, :]
  postfit_imu_u = postfit_residuals[5, :]
  
  # Create figure with 3x2 layout
  fig, axes = plt.subplots(3, 2, figsize=(15, 12))
  fig.suptitle('Pre-fit vs Post-fit Residuals\nGPS (Left Column) and IMU (Right Column)', 
               fontsize=16, fontweight='bold')
  
  # ===== COLUMN 1: GPS Residuals (meters) =====
  # East GPS residual
  axes[0, 0].scatter(time, prefit_gps_e, alpha=0.6, s=10, color='blue', label='Pre-fit')
  axes[0, 0].scatter(time, postfit_gps_e, alpha=0.6, s=10, color='red', label='Post-fit')
  axes[0, 0].set_ylabel('East pos (m)', fontsize=11, fontweight='bold')
  axes[0, 0].set_title('GPS East Position Residual', fontsize=11)
  axes[0, 0].grid(True, alpha=0.3)
  axes[0, 0].legend(loc='best', fontsize=9)
  
  # North GPS residual
  axes[1, 0].scatter(time, prefit_gps_n, alpha=0.6, s=10, color='blue', label='Pre-fit')
  axes[1, 0].scatter(time, postfit_gps_n, alpha=0.6, s=10, color='red', label='Post-fit')
  axes[1, 0].set_ylabel('North pos (m)', fontsize=11, fontweight='bold')
  axes[1, 0].set_title('GPS North Position Residual', fontsize=11)
  axes[1, 0].grid(True, alpha=0.3)
  axes[1, 0].legend(loc='best', fontsize=9)
  
  # Up GPS residual
  axes[2, 0].scatter(time, prefit_gps_u, alpha=0.6, s=10, color='blue', label='Pre-fit')
  axes[2, 0].scatter(time, postfit_gps_u, alpha=0.6, s=10, color='red', label='Post-fit')
  axes[2, 0].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
  axes[2, 0].set_ylabel('Up pos (m)', fontsize=11, fontweight='bold')
  axes[2, 0].set_title('GPS Up Position Residual', fontsize=11)
  axes[2, 0].grid(True, alpha=0.3)
  axes[2, 0].legend(loc='best', fontsize=9)
  
  # ===== COLUMN 2: IMU Residuals (m/s^2) =====
  # East IMU residual
  axes[0, 1].scatter(time, prefit_imu_e, alpha=0.6, s=10, color='blue', label='Pre-fit')
  axes[0, 1].scatter(time, postfit_imu_e, alpha=0.6, s=10, color='red', label='Post-fit')
  axes[0, 1].set_ylabel('East accel (m/s²)', fontsize=11, fontweight='bold')
  axes[0, 1].set_title('IMU East Acceleration Residual', fontsize=11)
  axes[0, 1].grid(True, alpha=0.3)
  axes[0, 1].legend(loc='best', fontsize=9)
  
  # North IMU residual
  axes[1, 1].scatter(time, prefit_imu_n, alpha=0.6, s=10, color='blue', label='Pre-fit')
  axes[1, 1].scatter(time, postfit_imu_n, alpha=0.6, s=10, color='red', label='Post-fit')
  axes[1, 1].set_ylabel('North accel (m/s²)', fontsize=11, fontweight='bold')
  axes[1, 1].set_title('IMU North Acceleration Residual', fontsize=11)
  axes[1, 1].grid(True, alpha=0.3)
  axes[1, 1].legend(loc='best', fontsize=9)
  
  # Up IMU residual
  axes[2, 1].scatter(time, prefit_imu_u, alpha=0.6, s=10, color='blue', label='Pre-fit')
  axes[2, 1].scatter(time, postfit_imu_u, alpha=0.6, s=10, color='red', label='Post-fit')
  axes[2, 1].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
  axes[2, 1].set_ylabel('Up accel (m/s²)', fontsize=11, fontweight='bold')
  axes[2, 1].set_title('IMU Up Acceleration Residual', fontsize=11)
  axes[2, 1].grid(True, alpha=0.3)
  axes[2, 1].legend(loc='best', fontsize=9)
  
  plt.tight_layout()
  plot_path = os.path.join(results_dir, 'prefit_postfit_residuals.png')
  fig.savefig(plot_path, dpi=150, bbox_inches='tight')
  print(f"Saved residuals plot: {plot_path}")
  plt.close(fig)
