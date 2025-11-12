import numpy as np
import matplotlib.pyplot as plt

def plot_results(state, cov):
  t = np.arange(len(state))
  x = np.array([matrix[0] for matrix in state])
  y = np.array([matrix[1] for matrix in state])
  z = np.array([matrix[2] for matrix in state])
  x_cov = np.array([matrix[0, 0] for matrix in cov])
  y_cov = np.array([matrix[1, 1] for matrix in cov])
  z_cov = np.array([matrix[2, 2] for matrix in cov])
  x_up = x + 1.96 * np.sqrt(x_cov)
  x_low = x - 1.96 * np.squrt(x_cov)
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
  plt.plot(t, x_up, color='cyan', linestyle='--', linewidth=1)
  plt.plot(t, x_low, color='cyan', linestyle='--', linewidth=1)
  plt.fill_between(t, x_up, x_low, color='gray', alpha=0.2)

  #plot y position vs. time
  plt.figure()
  plt.title("y position estimate vs. time")
  plt.xlabel("t")
  plt.ylabel("y")
  plt.plot(t, y, linestyle='-', color='blue', linewidth=1)
  plt.plot(t, y_up, color='cyan', linestyle='--', linewidth=1)
  plt.plot(t, y_low, color='cyan', linestyle='--', linewidth=1)
  plt.fill_between(t, y_up, y_low, color='gray', alpha=0.2)

  #plot z position vs. time
  plt.figure()
  plt.title("z position estimate vs. time")
  plt.xlabel("t")
  plt.ylabel("z")
  plt.plot(t, z, linestyle='-', color='blue', linewidth=1)
  plt.plot(t, z_up, color='cyan', linestyle='--', linewidth=1)
  plt.plot(t, z_low, color='cyan', linestyle='--', linewidth=1)
  plt.fill_between(t, z_up, z_low, color='gray', alpha=0.2)

  plt.show()
