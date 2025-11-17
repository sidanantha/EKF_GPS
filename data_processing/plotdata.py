import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

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
  x_up = x + 1.96 * np.sqrt(x_cov)
  x_low = x - 1.96 * np.sqrt(x_cov)
  y_up = y + 1.96 * np.sqrt(y_cov)
  y_low = y - 1.96 * np.sqrt(y_cov)
  z_up = z + 1.96 * np.sqrt(z_cov)
  z_low = z - 1.96 * np.sqrt(z_cov)

  #plot x position vs. time
  plt.figure()
  plt.title("x position estimate vs. time")
  plt.xlabel("t (time steps)")
  plt.ylabel("x (meters)")
  plt.plot(t, x, linestyle='-', color='blue', linewidth=1, label='Estimate')
  plt.plot(t, x_up, linestyle='--', color='cyan', linewidth=1, label='95% CI')
  plt.plot(t, x_low, linestyle='--', color='cyan', linewidth=1)
  plt.fill_between(t, x_up, x_low, color='gray', alpha=0.2)
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.savefig(f'{save_dir}/x_position.png', dpi=150, bbox_inches='tight')
  print(f"✓ Saved: {save_dir}/x_position.png")

  #plot y position vs. time
  plt.figure()
  plt.title("y position estimate vs. time")
  plt.xlabel("t (time steps)")
  plt.ylabel("y (meters)")
  plt.plot(t, y, linestyle='-', color='blue', linewidth=1, label='Estimate')
  plt.plot(t, y_up, linestyle='--', color='cyan', linewidth=1, label='95% CI')
  plt.plot(t, y_low, linestyle='--', color='cyan', linewidth=1)
  plt.fill_between(t, y_up, y_low, color='gray', alpha=0.2)
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.savefig(f'{save_dir}/y_position.png', dpi=150, bbox_inches='tight')
  print(f"✓ Saved: {save_dir}/y_position.png")

  #plot z position vs. time
  plt.figure()
  plt.title("z position estimate vs. time")
  plt.xlabel("t (time steps)")
  plt.ylabel("z (meters)")
  plt.plot(t, z, linestyle='-', color='blue', linewidth=1, label='Estimate')
  plt.plot(t, z_up, linestyle='--', color='cyan', linewidth=1, label='95% CI')
  plt.plot(t, z_low, linestyle='--', color='cyan', linewidth=1)
  plt.fill_between(t, z_up, z_low, color='gray', alpha=0.2)
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.savefig(f'{save_dir}/z_position.png', dpi=150, bbox_inches='tight')
  print(f"✓ Saved: {save_dir}/z_position.png")

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
