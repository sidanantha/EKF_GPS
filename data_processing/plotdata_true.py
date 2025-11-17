"""
Plotting functions for comparing estimates vs. true values and showing errors.
Includes position, attitude, and comprehensive error analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
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


def plot_positions_comparison(estimated_positions, true_positions, save_dir='results', show=True):
  """
  Plot estimated vs true positions for X, Y, Z.
  Creates a single figure with 3 vertically stacked subplots.
  
  Args:
    estimated_positions: (3, n_steps) array of estimated positions
    true_positions: (3, n_steps) array of true positions
    save_dir: Directory to save plots
    show: If True, display plots interactively
  """
  import os
  os.makedirs(save_dir, exist_ok=True)
  
  t = np.arange(estimated_positions.shape[1])
  
  # Create figure with 3 subplots stacked vertically
  fig, axes = plt.subplots(3, 1, figsize=(12, 10))
  fig.suptitle('Position Estimates vs. True Values', fontsize=16, fontweight='bold')
  
  # Plot X Position
  axes[0].plot(t, estimated_positions[0, :], linestyle='-', color='blue', linewidth=2, label='Estimated')
  axes[0].plot(t, true_positions[0, :], linestyle='--', color='red', linewidth=2, label='True')
  axes[0].set_ylabel('X Position (meters)', fontsize=11, fontweight='bold')
  axes[0].set_title('X Position Comparison', fontsize=10)
  axes[0].grid(True, alpha=0.3)
  axes[0].legend(loc='upper right')
  
  # Plot Y Position
  axes[1].plot(t, estimated_positions[1, :], linestyle='-', color='blue', linewidth=2, label='Estimated')
  axes[1].plot(t, true_positions[1, :], linestyle='--', color='red', linewidth=2, label='True')
  axes[1].set_ylabel('Y Position (meters)', fontsize=11, fontweight='bold')
  axes[1].set_title('Y Position Comparison', fontsize=10)
  axes[1].grid(True, alpha=0.3)
  axes[1].legend(loc='upper right')
  
  # Plot Z Position
  axes[2].plot(t, estimated_positions[2, :], linestyle='-', color='blue', linewidth=2, label='Estimated')
  axes[2].plot(t, true_positions[2, :], linestyle='--', color='red', linewidth=2, label='True')
  axes[2].set_xlabel('Time (steps)', fontsize=11, fontweight='bold')
  axes[2].set_ylabel('Z Position (meters)', fontsize=11, fontweight='bold')
  axes[2].set_title('Z Position Comparison', fontsize=10)
  axes[2].grid(True, alpha=0.3)
  axes[2].legend(loc='upper right')
  
  plt.tight_layout()
  plt.savefig(f'{save_dir}/positions_comparison.png', dpi=150, bbox_inches='tight')
  print(f"✓ Saved: {save_dir}/positions_comparison.png")
  
  if show:
    plt.show()
  else:
    plt.close(fig)


def plot_attitude_comparison(estimated_quaternions, true_quaternions, save_dir='results', show=True):
  """
  Plot estimated vs true attitude (roll, pitch, yaw).
  Creates a single figure with 3 vertically stacked subplots.
  
  Args:
    estimated_quaternions: List of estimated Quaternion objects
    true_quaternions: List of true Quaternion objects
    save_dir: Directory to save plots
    show: If True, display plots interactively
  """
  import os
  os.makedirs(save_dir, exist_ok=True)
  
  # Convert quaternions to Euler angles
  est_euler = [quaternion_to_euler(q) for q in estimated_quaternions]
  true_euler = [quaternion_to_euler(q) for q in true_quaternions]
  
  t = np.arange(len(est_euler))
  
  # Extract individual angles and convert to degrees
  est_roll = np.array([np.degrees(e[0]) for e in est_euler])
  est_pitch = np.array([np.degrees(e[1]) for e in est_euler])
  est_yaw = np.array([np.degrees(e[2]) for e in est_euler])
  
  true_roll = np.array([np.degrees(e[0]) for e in true_euler])
  true_pitch = np.array([np.degrees(e[1]) for e in true_euler])
  true_yaw = np.array([np.degrees(e[2]) for e in true_euler])
  
  # Create figure with 3 subplots stacked vertically
  fig, axes = plt.subplots(3, 1, figsize=(12, 10))
  fig.suptitle('Attitude Estimates vs. True Values (Euler Angles)', fontsize=16, fontweight='bold')
  
  # Plot Roll
  axes[0].plot(t, est_roll, linestyle='-', color='blue', linewidth=2, label='Estimated')
  axes[0].plot(t, true_roll, linestyle='--', color='red', linewidth=2, label='True')
  axes[0].set_ylabel('Roll (degrees)', fontsize=11, fontweight='bold')
  axes[0].set_title('Roll (φ) Comparison', fontsize=10)
  axes[0].grid(True, alpha=0.3)
  axes[0].legend(loc='upper right')
  
  # Plot Pitch
  axes[1].plot(t, est_pitch, linestyle='-', color='blue', linewidth=2, label='Estimated')
  axes[1].plot(t, true_pitch, linestyle='--', color='red', linewidth=2, label='True')
  axes[1].set_ylabel('Pitch (degrees)', fontsize=11, fontweight='bold')
  axes[1].set_title('Pitch (θ) Comparison', fontsize=10)
  axes[1].grid(True, alpha=0.3)
  axes[1].legend(loc='upper right')
  
  # Plot Yaw
  axes[2].plot(t, est_yaw, linestyle='-', color='blue', linewidth=2, label='Estimated')
  axes[2].plot(t, true_yaw, linestyle='--', color='red', linewidth=2, label='True')
  axes[2].set_xlabel('Time (steps)', fontsize=11, fontweight='bold')
  axes[2].set_ylabel('Yaw (degrees)', fontsize=11, fontweight='bold')
  axes[2].set_title('Yaw (ψ) Comparison', fontsize=10)
  axes[2].grid(True, alpha=0.3)
  axes[2].legend(loc='upper right')
  
  plt.tight_layout()
  plt.savefig(f'{save_dir}/attitude_comparison.png', dpi=150, bbox_inches='tight')
  print(f"✓ Saved: {save_dir}/attitude_comparison.png")
  
  if show:
    plt.show()
  else:
    plt.close(fig)


def plot_estimation_errors(estimated_positions, true_positions, estimated_quaternions, 
                          true_quaternions, save_dir='results', show=True):
  """
  Plot estimation errors for all 6 states [x, y, z, roll, pitch, yaw].
  Creates a single figure with 3x2 (6) vertically stacked subplots.
  
  Args:
    estimated_positions: (3, n_steps) array of estimated positions
    true_positions: (3, n_steps) array of true positions
    estimated_quaternions: List of estimated Quaternion objects
    true_quaternions: List of true Quaternion objects
    save_dir: Directory to save plots
    show: If True, display plots interactively
  """
  import os
  os.makedirs(save_dir, exist_ok=True)
  
  t = np.arange(estimated_positions.shape[1])
  
  # Calculate position errors
  pos_error = estimated_positions - true_positions
  x_error = pos_error[0, :]
  y_error = pos_error[1, :]
  z_error = pos_error[2, :]
  
  # Calculate attitude errors (Euler angles)
  est_euler = [quaternion_to_euler(q) for q in estimated_quaternions]
  true_euler = [quaternion_to_euler(q) for q in true_quaternions]
  
  att_error = np.array([
    [np.degrees(est_euler[i][0] - true_euler[i][0]) for i in range(len(est_euler))],
    [np.degrees(est_euler[i][1] - true_euler[i][1]) for i in range(len(est_euler))],
    [np.degrees(est_euler[i][2] - true_euler[i][2]) for i in range(len(est_euler))]
  ])
  
  roll_error = att_error[0, :]
  pitch_error = att_error[1, :]
  yaw_error = att_error[2, :]
  
  # Create figure with 3x2 (6) subplots
  fig, axes = plt.subplots(3, 2, figsize=(14, 11))
  fig.suptitle('Estimation Errors Over Time', fontsize=16, fontweight='bold')
  
  # Row 0: Position errors
  # X Position Error
  axes[0, 0].plot(t, x_error, linestyle='-', color='blue', linewidth=1.5)
  axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
  axes[0, 0].fill_between(t, x_error, 0, alpha=0.3, color='blue')
  axes[0, 0].set_ylabel('Error (meters)', fontsize=10, fontweight='bold')
  axes[0, 0].set_title('X Position Error', fontsize=10)
  axes[0, 0].grid(True, alpha=0.3)
  
  # Y Position Error
  axes[0, 1].plot(t, y_error, linestyle='-', color='green', linewidth=1.5)
  axes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
  axes[0, 1].fill_between(t, y_error, 0, alpha=0.3, color='green')
  axes[0, 1].set_ylabel('Error (meters)', fontsize=10, fontweight='bold')
  axes[0, 1].set_title('Y Position Error', fontsize=10)
  axes[0, 1].grid(True, alpha=0.3)
  
  # Row 1: More position and attitude errors
  # Z Position Error
  axes[1, 0].plot(t, z_error, linestyle='-', color='red', linewidth=1.5)
  axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
  axes[1, 0].fill_between(t, z_error, 0, alpha=0.3, color='red')
  axes[1, 0].set_ylabel('Error (meters)', fontsize=10, fontweight='bold')
  axes[1, 0].set_title('Z Position Error', fontsize=10)
  axes[1, 0].grid(True, alpha=0.3)
  
  # Roll Error
  axes[1, 1].plot(t, roll_error, linestyle='-', color='purple', linewidth=1.5)
  axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
  axes[1, 1].fill_between(t, roll_error, 0, alpha=0.3, color='purple')
  axes[1, 1].set_ylabel('Error (degrees)', fontsize=10, fontweight='bold')
  axes[1, 1].set_title('Roll (φ) Error', fontsize=10)
  axes[1, 1].grid(True, alpha=0.3)
  
  # Row 2: Attitude errors
  # Pitch Error
  axes[2, 0].plot(t, pitch_error, linestyle='-', color='orange', linewidth=1.5)
  axes[2, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
  axes[2, 0].fill_between(t, pitch_error, 0, alpha=0.3, color='orange')
  axes[2, 0].set_xlabel('Time (steps)', fontsize=10, fontweight='bold')
  axes[2, 0].set_ylabel('Error (degrees)', fontsize=10, fontweight='bold')
  axes[2, 0].set_title('Pitch (θ) Error', fontsize=10)
  axes[2, 0].grid(True, alpha=0.3)
  
  # Yaw Error
  axes[2, 1].plot(t, yaw_error, linestyle='-', color='brown', linewidth=1.5)
  axes[2, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
  axes[2, 1].fill_between(t, yaw_error, 0, alpha=0.3, color='brown')
  axes[2, 1].set_xlabel('Time (steps)', fontsize=10, fontweight='bold')
  axes[2, 1].set_ylabel('Error (degrees)', fontsize=10, fontweight='bold')
  axes[2, 1].set_title('Yaw (ψ) Error', fontsize=10)
  axes[2, 1].grid(True, alpha=0.3)
  
  plt.tight_layout()
  plt.savefig(f'{save_dir}/estimation_errors.png', dpi=150, bbox_inches='tight')
  print(f"✓ Saved: {save_dir}/estimation_errors.png")
  
  if show:
    plt.show()
  else:
    plt.close(fig)

