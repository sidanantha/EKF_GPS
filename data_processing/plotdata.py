import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation import FuncAnimation

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

def plot_results(state, cov):
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
  plt.xlabel("t")
  plt.ylabel("x")
  plt.plot(t, x, linestyle='-', color='blue', linewidth=1)
  plt.plot(t, x_up, linestyle='--', color='cyan', linewidth=1)
  plt.plot(t, x_low, linestyle='--', color='cyan', linewidth=1)
  plt.fill_between(t, x_up, x_low, color='gray', alpha=0.2)

  #plot y position vs. time
  plt.figure()
  plt.title("y position estimate vs. time")
  plt.xlabel("t")
  plt.ylabel("y")
  plt.plot(t, y, linestyle='-', color='blue', linewidth=1)
  plt.plot(t, y_up, linestyle='--', color='cyan', linewidth=1)
  plt.plot(t, y_low, linestyle='--', color='cyan', linewidth=1)
  plt.fill_between(t, y_up, y_low, color='gray', alpha=0.2)

  #plot z position vs. time
  plt.figure()
  plt.title("z position estimate vs. time")
  plt.xlabel("t")
  plt.ylabel("z")
  plt.plot(t, z, linestyle='-', color='blue', linewidth=1)
  plt.plot(t, z_up, linestyle='--', color='cyan', linewidth=1)
  plt.plot(t, z_low, linestyle='--', color='cyan', linewidth=1)
  plt.fill_between(t, z_up, z_low, color='gray', alpha=0.2)

  #plot trajectory
  figure = plt.figure()
  traj = figure.add_subplot(111, projection='3d')

  traj.plot(x, y, z, linestyle='-', color='blue', linewidth=1)
  traj.scatter(x[0], y[0], z[0], marker='.', color='cyan')
  traj.scatter(x[-1], y[-1], z[-1], marker='.', color='cyan')
  for i in range(0, len(state)):
    xe, ye, ze = make_ellipse(state[i], cov[i])
    traj.plot_surface(xe, ye, ze, color='gray', alpha=0.2)
    
  traj.set_title("Trajectory")
  traj.set_xlabel("x position")
  traj.set_ylabel("y position")
  traj.set_zlabel("z position")

  #trajectory animation
  animation = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  line = ax.plot([], [], [])
  point = ax.plot([], [], [])
  animated = FuncAnimation(animation, animate, frames=len(x), interval = 50)

  plt.show()
