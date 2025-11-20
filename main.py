# main.py
# Main file to run the EKF algorithm for position and MEKF for attitude

import sys
import EKF.EKF as EKF
import data_processing.dataprocessing as data_processing
import utils.utils as utils
import numpy as np
from pyquaternion import Quaternion
from mekf.MEKF import MEKF
from data_processing.plotdata import plot_results, plot_euler_angles, plot_residuals


# Define constants
def define_constants():
    '''
    Define constants for the EKF and MEKF algorithms
    '''
    constants = {
        'dt': 0.1, # Time step
        'm': 1000, # Mass
        'I': np.eye(3), # Inertial matrix
        'sigma_IMU_x': 0.024693*100, # Standard deviation of the IMU x acceleration
        'sigma_IMU_y': 0.019272*100, # Standard deviation of the IMU y acceleration
        'sigma_IMU_z': 0.032716*100, # Standard deviation of the IMU z acceleration
        'sigma_GPS_x': 2.3300000/10, # Standard deviation of the GPS x position
        'sigma_GPS_y': 2.3600000/10, # Standard deviation of the GPS y position
        'sigma_GPS_z': 2.9500000/10, # Standard deviation of the GPS z position
        'initial_uncertainty': 100, # Initial uncertainty for the state vector
        'coordinates_observer': np.array([0, 0, 0]), # Observer position, latitude, longitude, altitude
        # MEKF parameters
        'estimate_covariance': 1.0,
        'gyro_cov': 0.1,
        'gyro_bias_cov': 0.1,
        'accel_proc_cov': 0.1,
        'accel_bias_cov': 0.1,
        'accel_obs_cov': 0.1,
        # IMU Calibration Parameters:
        'R_accel_to_body': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), # accelerometer to body frame rotation matrix
        'accel_bias': np.array([0.398412, 0.532099, 0.051545]), # accelerometer bias
        'gyro_bias': np.array([0.177013, -0.001051, -0.020324]), # gyro bias
        # GPS calibration parameters:
        'gps_bias': np.array([-0.05622031074017286, -0.4507213160395622, -1.5610635122284293]), # GPS bias
        # a_z free fall calibration exists but it seems very noisy. Should not use.
    }
    return constants



# Main Function:
def main(data_file='data.csv', results_dir='results', gps_spoof_noise=0, gps_spoof_start_time=-1):
    '''
    Main function to run both EKF (for position) and MEKF (for attitude) algorithms
    
    Args:
        data_file: Path to CSV file with measurement data (default: 'data.csv')
        results_dir: Directory to save results (default: 'results')
        gps_spoof_noise: Additional GPS noise to inject in meters (default: 0)
        gps_spoof_start_time: Time to start GPS spoofing in seconds (default: -1 for disabled)
    '''
    # Define constants
    constants = define_constants()
    # Add spoofing parameters to constants
    constants['GPS_spoof_noise'] = gps_spoof_noise
    constants['GPS_spoof_start_time'] = gps_spoof_start_time
    # Determine observer position:
    x_observer = utils.ecef_to_lla(constants['coordinates_observer'][0], constants['coordinates_observer'][1], constants['coordinates_observer'][2])
    
    # Load data
    time, starttime, endtime, timestep, lat, lon, alt, x, y, z, ax, ay, az, gx, gy, gz = data_processing.data_processing(data_file)

    # Remove GPS bias
    x -= constants['gps_bias'][0]
    y -= constants['gps_bias'][1]
    z -= constants['gps_bias'][2]
    
    # ============ EKF INITIALIZATION (for position estimation) ============
    # Build EKF matrices:
    A = EKF.build_A(constants['dt'])
    B = np.zeros((9, 3))    # assume no control input
    C = EKF.build_C(constants['m'], constants['dt'])
    Q = EKF.build_Q(constants['sigma_IMU_x'], constants['sigma_IMU_y'], constants['sigma_IMU_z'])
    # Build R with much higher noise for IMU measurements to trust GPS significantly more
    # GPS noise: 1.0 m, IMU acceleration noise: heavily downweighted to prevent drift
    R = EKF.build_R(constants['sigma_GPS_x'] + constants['GPS_spoof_noise'], constants['sigma_GPS_y'] + constants['GPS_spoof_noise'], constants['sigma_GPS_z'] + constants['GPS_spoof_noise'], constants['sigma_IMU_x'], constants['sigma_IMU_y'], constants['sigma_IMU_z'])
    
    # Init EKF state vector and covariance matrix:
    x_k = np.zeros((9,1))
    P_k = np.eye(9)*constants['initial_uncertainty']
    
    # Init EKF storage:
    x_k_storage = np.zeros((9, len(time)))
    P_k_storage = np.zeros((9, 9, len(time)))
    prefit_residuals_storage = np.zeros((6, len(time)))  # Store pre-fit residuals
    postfit_residuals_storage = np.zeros((6, len(time)))  # Store post-fit residuals
    
    # ============ MEKF INITIALIZATION (for attitude estimation) ============
    # Initialize with identity quaternion
    initial_quaternion = Quaternion(axis=[1, 0, 0], angle=0)
    mekf = MEKF(initial_quaternion, 
                 constants['estimate_covariance'],
                 constants['gyro_cov'],
                 constants['gyro_bias_cov'],
                 constants['accel_proc_cov'],
                 constants['accel_bias_cov'],
                 constants['accel_obs_cov'])
    
    # Init MEKF storage:
    quaternion_storage = []  # Store quaternions
    mekf_cov_storage = np.zeros((15, 15, len(time)))
    
    # Extract the reference position, in terms of its coordinates:
    ref_lat = lat[0]
    ref_lon = lon[0]
    ref_ecef_x = x[0]
    ref_ecef_y = y[0]
    ref_ecef_z = z[0]
    
    # ============ MAIN LOOP ============
    for i in range(len(time)):
        
        # Create acceleration and gyro vectors
        accel_meas = np.array([ax[i], ay[i], az[i]])
        gyro_meas = np.array([gx[i], gy[i], gz[i]])
        # There is a bias, subtract it out from the measurements:
        accel_meas = accel_meas - constants['accel_bias']
        gyro_meas = gyro_meas - constants['gyro_bias']
        # These are in terms of the accelerometer axes, need to transform into the body frame
        accel_body = constants['R_accel_to_body'] @ accel_meas
        gyro_body = constants['R_accel_to_body'] @ gyro_meas
        
        
        # Convert the ECEF x,y,z positions to ENU positions
        pos_enu = utils.ecef_to_enu(x[i] - ref_ecef_x, y[i] - ref_ecef_y, z[i] - ref_ecef_z, ref_lat, ref_lon)
        
        
        # # ========== RUN MEKF FIRST (Attitude Estimation) ==========
        # # MEKF expects NORMALIZED accelerometer measurements (unit vector in gravity direction)
        # # Normalize the acceleration vector to unit magnitude
        # accel_norm = np.linalg.norm(accel_body)
        # if accel_norm > 0:
        #     accel_normalized = accel_body / accel_norm
        # else:
        #     accel_normalized = accel_body
        
        # # Update MEKF with body-frame measurements
        # mekf.update(gyro_body, accel_normalized, constants['dt'])
        
        # # Get current attitude estimate (after update) for frame conversion
        # current_attitude = mekf.estimate
        
        # ========== RUN EKF (Position Estimation) ==========
        # Convert acceleration from body frame to inertial frame using MEKF attitude estimate
        # accel_inertial = R_b2i * accel_body (rotate body frame acceleration to inertial)
        # accel_inertial = current_attitude.rotate(accel_meas)
        
        
        
        # Rotate the acceleration vector from the body-fixed frame to the ENU frame
        roll = np.arcsin(accel_body[0]/9.81)
        pitch = np.arcsin(accel_body[1]/accel_body[2])
        # Compute current yaw angle from magnetometer data
        yaw = utils.compute_yaw(gx[i], gy[i], gz[i], 0, 0, constants['R_accel_to_body'])
        # Comoute DCM for body to ENU frame
        Rx_body_to_enu = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry_body_to_enu = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz_body_to_enu = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
        R_body_to_enu = Rx_body_to_enu @ Ry_body_to_enu @ Rz_body_to_enu
        # Rotate the acceleration vector from the body-fixed frame to the ENU frame
        accel_enu = R_body_to_enu @ accel_body
        
        # Remove gravity component from Z axis for EKF measurement
        accel_enu_no_g = accel_enu.copy()
        accel_enu_no_g[2] -= 9.81
        
        # Define the measurement vector: GPS position (inertial) + IMU acceleration (inertial, no gravity)
        y_t = np.zeros((6, 1))
        
        # Inject GPS spoofing noise if specified and after start time
        current_time = time[i]
        spoof_active = (constants['GPS_spoof_noise'] > 0 and 
                       constants['GPS_spoof_start_time'] >= 0 and 
                       current_time >= constants['GPS_spoof_start_time'])
        gps_spoof = np.random.normal(0, constants['GPS_spoof_noise'], 3) if spoof_active else np.zeros(3)
        
        y_t[0] = pos_enu[0] + gps_spoof[0]  # GPS x position (inertial frame) + spoof noise
        y_t[1] = pos_enu[1] + gps_spoof[1]  # GPS y position (inertial frame) + spoof noise
        y_t[2] = pos_enu[2] + gps_spoof[2]  # GPS z position (inertial frame) + spoof noise
        y_t[3] = accel_enu_no_g[0]  # IMU acceleration x (inertial frame, gravity removed)
        y_t[4] = accel_enu_no_g[1]  # IMU acceleration y (inertial frame, gravity removed)
        y_t[5] = accel_enu_no_g[2]  # IMU acceleration z (inertial frame, gravity removed)
        
        # Run EKF iteration
        x_k, P_k, prefit_residual, postfit_residual = EKF.EKF_iteration(x_k, P_k, y_t, np.zeros((3,1)), R, Q, A, B, C, 
                np.array([constants['sigma_IMU_x'], constants['sigma_IMU_y'], constants['sigma_IMU_z']]), 
                np.array([constants['sigma_GPS_x'], constants['sigma_GPS_y'], constants['sigma_GPS_z']]))
        x_k_storage[:, i] = x_k.flatten()
        P_k_storage[:, :, i] = P_k
        prefit_residuals_storage[:, i] = prefit_residual.flatten()
        postfit_residuals_storage[:, i] = postfit_residual.flatten()
        
        # ========== STORE MEKF RESULTS ==========
        quaternion_storage.append(mekf.estimate)
        mekf_cov_storage[:, :, i] = mekf.estimate_covariance
    
    return x_k_storage, P_k_storage, quaternion_storage, mekf_cov_storage, constants, prefit_residuals_storage, postfit_residuals_storage


def run_spoofing_analysis(data_file, results_dir='results/spoofing_analysis'):
    '''
    Run EKF with different levels of GPS spoofing noise and compare results.
    
    Generates 10 different noise levels from 1 to 100 meters, all starting at time 0.
    Creates comparison plots showing position estimates and 3D trajectories.
    
    Args:
        data_file: Path to CSV file with measurement data
        results_dir: Directory to save results
    '''
    import matplotlib.pyplot as plt
    from data_processing.plotdata import plot_results, plot_euler_angles
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate 10 noise levels from 1 to 100 meters
    noise_levels = np.linspace(1, 100, 10)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    print("\n" + "=" * 60)
    print("RUNNING GPS SPOOFING ANALYSIS")
    print("=" * 60)
    print(f"Testing {len(noise_levels)} noise levels: {noise_levels}")
    print("=" * 60 + "\n")
    
    # Store results for comparison
    all_results = {'no_spoof': None}
    
    # First, run with no spoofing
    print("Running baseline (no spoofing)...")
    x_k_storage, P_k_storage, quaternion_storage, mekf_cov_storage, constants, prefit_res, postfit_res = main(
        data_file, results_dir, gps_spoof_noise=0, gps_spoof_start_time=-1
    )
    all_results['no_spoof'] = {
        'state': x_k_storage,
        'cov': P_k_storage,
        'quaternion': quaternion_storage,
        'dt': constants['dt']
    }
    
    # Run with each spoofing level
    for i, noise in enumerate(noise_levels):
        print(f"Running with {noise:.1f}m spoofing noise ({i+1}/10)...")
        x_k_storage_spoof, P_k_storage_spoof, quaternion_storage_spoof, mekf_cov_storage_spoof, constants, _, _ = main(
            data_file, results_dir, gps_spoof_noise=noise, gps_spoof_start_time=0
        )
        all_results[f'spoof_{i}'] = {
            'state': x_k_storage_spoof,
            'cov': P_k_storage_spoof,
            'noise': noise,
            'color': colors[i]
        }
    
    # Create comparison plot for positions
    print("\nGenerating comparison plots...")
    dt = constants['dt']
    time = np.arange(all_results['no_spoof']['state'].shape[1]) * dt
    
    # Extract East, North, Up positions and covariances
    east_no_spoof = all_results['no_spoof']['state'][0, :]
    north_no_spoof = all_results['no_spoof']['state'][1, :]
    up_no_spoof = all_results['no_spoof']['state'][2, :]
    
    cov_no_spoof = all_results['no_spoof']['cov']
    east_cov = np.array([cov_no_spoof[0, 0, i] for i in range(cov_no_spoof.shape[2])])
    north_cov = np.array([cov_no_spoof[1, 1, i] for i in range(cov_no_spoof.shape[2])])
    up_cov = np.array([cov_no_spoof[2, 2, i] for i in range(cov_no_spoof.shape[2])])
    
    east_up = east_no_spoof + 3.0 * np.sqrt(east_cov)
    east_low = east_no_spoof - 3.0 * np.sqrt(east_cov)
    north_up = north_no_spoof + 3.0 * np.sqrt(north_cov)
    north_low = north_no_spoof - 3.0 * np.sqrt(north_cov)
    up_up = up_no_spoof + 3.0 * np.sqrt(up_cov)
    up_low = up_no_spoof - 3.0 * np.sqrt(up_cov)
    
    # Create position comparison plots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('GPS Spoofing Analysis: Position Estimates', fontsize=16, fontweight='bold')
    
    # Plot baseline in black with covariance bounds
    axes[0].fill_between(time, east_up, east_low, color='black', alpha=0.1, label='3σ Bounds (No Spoof)')
    axes[0].plot(time, east_no_spoof, linewidth=2.5, color='black', label='No Spoofing', zorder=10)
    
    axes[1].fill_between(time, north_up, north_low, color='black', alpha=0.1, label='3σ Bounds (No Spoof)')
    axes[1].plot(time, north_no_spoof, linewidth=2.5, color='black', label='No Spoofing', zorder=10)
    
    axes[2].fill_between(time, up_up, up_low, color='black', alpha=0.1, label='3σ Bounds (No Spoof)')
    axes[2].plot(time, up_no_spoof, linewidth=2.5, color='black', label='No Spoofing', zorder=10)
    
    # Plot spoofed results
    for i, noise in enumerate(noise_levels):
        key = f'spoof_{i}'
        state_spoof = all_results[key]['state']
        cov_spoof = all_results[key]['cov']
        color = colors[i]
        
        # Calculate covariance bounds for spoofed results
        spoof_east_cov = np.array([cov_spoof[0, 0, j] for j in range(cov_spoof.shape[2])])
        spoof_north_cov = np.array([cov_spoof[1, 1, j] for j in range(cov_spoof.shape[2])])
        spoof_up_cov = np.array([cov_spoof[2, 2, j] for j in range(cov_spoof.shape[2])])
        
        spoof_east_up = state_spoof[0, :] + 3.0 * np.sqrt(spoof_east_cov)
        spoof_east_low = state_spoof[0, :] - 3.0 * np.sqrt(spoof_east_cov)
        spoof_north_up = state_spoof[1, :] + 3.0 * np.sqrt(spoof_north_cov)
        spoof_north_low = state_spoof[1, :] - 3.0 * np.sqrt(spoof_north_cov)
        spoof_up_up = state_spoof[2, :] + 3.0 * np.sqrt(spoof_up_cov)
        spoof_up_low = state_spoof[2, :] - 3.0 * np.sqrt(spoof_up_cov)
        
        # Add transparent covariance bounds
        axes[0].fill_between(time, spoof_east_up, spoof_east_low, color=color, alpha=0.15)
        axes[0].plot(time, state_spoof[0, :], linewidth=1.5, color=color, label=f'{noise:.1f}m', alpha=0.8)
        
        axes[1].fill_between(time, spoof_north_up, spoof_north_low, color=color, alpha=0.15)
        axes[1].plot(time, state_spoof[1, :], linewidth=1.5, color=color, label=f'{noise:.1f}m', alpha=0.8)
        
        axes[2].fill_between(time, spoof_up_up, spoof_up_low, color=color, alpha=0.15)
        axes[2].plot(time, state_spoof[2, :], linewidth=1.5, color=color, label=f'{noise:.1f}m', alpha=0.8)
    
    # Configure subplots
    axes[0].set_ylabel('East (m)', fontsize=11, fontweight='bold')
    axes[0].set_title('East Position vs. Time', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=9, ncol=4)
    
    axes[1].set_ylabel('North (m)', fontsize=11, fontweight='bold')
    axes[1].set_title('North Position vs. Time', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=9, ncol=4)
    
    axes[2].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Up (m)', fontsize=11, fontweight='bold')
    axes[2].set_title('Up Position vs. Time', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='best', fontsize=9, ncol=4)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'spoofing_positions_comparison.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close(fig)
    
    # Create 3D trajectory comparison plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot baseline in black
    ax.plot(east_no_spoof, north_no_spoof, up_no_spoof, linewidth=3, color='black', 
            label='No Spoofing', zorder=10)
    ax.scatter(east_no_spoof[0], north_no_spoof[0], up_no_spoof[0], marker='o', 
              color='black', s=100, label='Start (No Spoof)', zorder=10)
    ax.scatter(east_no_spoof[-1], north_no_spoof[-1], up_no_spoof[-1], marker='s', 
              color='black', s=100, label='End (No Spoof)', zorder=10)
    
    # Plot spoofed trajectories
    for i, noise in enumerate(noise_levels):
        key = f'spoof_{i}'
        state_spoof = all_results[key]['state']
        color = colors[i]
        
        ax.plot(state_spoof[0, :], state_spoof[1, :], state_spoof[2, :], 
               linewidth=1.5, color=color, label=f'{noise:.1f}m', alpha=0.7)
    
    ax.set_xlabel('East (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('North (m)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Up (m)', fontsize=11, fontweight='bold')
    ax.set_title('3D Trajectory Comparison - GPS Spoofing Analysis', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.view_init(elev=20, azim=45)
    
    # Set equal aspect ratio based on data ranges
    all_east = np.concatenate([east_no_spoof] + [all_results[f'spoof_{i}']['state'][0, :] for i in range(len(noise_levels))])
    all_north = np.concatenate([north_no_spoof] + [all_results[f'spoof_{i}']['state'][1, :] for i in range(len(noise_levels))])
    all_up = np.concatenate([up_no_spoof] + [all_results[f'spoof_{i}']['state'][2, :] for i in range(len(noise_levels))])
    
    max_range = np.array([all_east.max()-all_east.min(), all_north.max()-all_north.min(), all_up.max()-all_up.min()]).max() / 2.0
    mid_x = (all_east.max()+all_east.min()) * 0.5
    mid_y = (all_north.max()+all_north.min()) * 0.5
    mid_z = (all_up.max()+all_up.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'spoofing_trajectory_3d_comparison.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close(fig)
    
    # Create individual comparison plots for each spoofing case
    print("Generating individual spoofing comparison plots...")
    
    for i, noise in enumerate(noise_levels):
        key = f'spoof_{i}'
        state_spoof = all_results[key]['state']
        cov_spoof = all_results[key]['cov']
        color = colors[i]
        
        # Extract East, North, Up for spoofed case
        east_spoof = state_spoof[0, :]
        north_spoof = state_spoof[1, :]
        up_spoof = state_spoof[2, :]
        
        # Extract covariances for spoofed case
        spoof_east_cov = np.array([cov_spoof[0, 0, j] for j in range(cov_spoof.shape[2])])
        spoof_north_cov = np.array([cov_spoof[1, 1, j] for j in range(cov_spoof.shape[2])])
        spoof_up_cov = np.array([cov_spoof[2, 2, j] for j in range(cov_spoof.shape[2])])
        
        spoof_east_up = east_spoof + 3.0 * np.sqrt(spoof_east_cov)
        spoof_east_low = east_spoof - 3.0 * np.sqrt(spoof_east_cov)
        spoof_north_up = north_spoof + 3.0 * np.sqrt(spoof_north_cov)
        spoof_north_low = north_spoof - 3.0 * np.sqrt(spoof_north_cov)
        spoof_up_up = up_spoof + 3.0 * np.sqrt(spoof_up_cov)
        spoof_up_low = up_spoof - 3.0 * np.sqrt(spoof_up_cov)
        
        # Calculate error between spoofed and unspoofed (difference)
        error_east = east_spoof - east_no_spoof
        error_north = north_spoof - north_no_spoof
        error_up = up_spoof - up_no_spoof
        
        # Calculate combined error bounds (quadrature: √(σ_unspoofed² + σ_spoofed²))
        error_east_cov = np.sqrt(east_cov + spoof_east_cov)
        error_north_cov = np.sqrt(north_cov + spoof_north_cov)
        error_up_cov = np.sqrt(up_cov + spoof_up_cov)
        
        error_east_up = 3.0 * error_east_cov
        error_east_low = -3.0 * error_east_cov
        error_north_up = 3.0 * error_north_cov
        error_north_low = -3.0 * error_north_cov
        error_up_up = 3.0 * error_up_cov
        error_up_low = -3.0 * error_up_cov
        
        # Create individual comparison plot (3x2)
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'GPS Spoofing Analysis: {noise:.1f}m vs No Spoofing', fontsize=16, fontweight='bold')
        
        # ===== COLUMN 1: Positions =====
        # East position
        axes[0, 0].fill_between(time, east_up, east_low, color='black', alpha=0.25, label='3σ Bounds (No Spoofing)')
        axes[0, 0].plot(time, east_no_spoof, linewidth=2.5, color='black', label='No Spoofing', zorder=10)
        axes[0, 0].fill_between(time, spoof_east_up, spoof_east_low, color=color, alpha=0.25, label=f'3σ Bounds ({noise:.1f}m)')
        axes[0, 0].plot(time, east_spoof, linewidth=2.5, color=color, label=f'{noise:.1f}m Spoofing', zorder=9)
        axes[0, 0].set_ylabel('East Position (m)', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('East Position vs. Time', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(loc='best', fontsize=9)
        
        # North position
        axes[1, 0].fill_between(time, north_up, north_low, color='black', alpha=0.25, label='3σ Bounds (No Spoofing)')
        axes[1, 0].plot(time, north_no_spoof, linewidth=2.5, color='black', label='No Spoofing', zorder=10)
        axes[1, 0].fill_between(time, spoof_north_up, spoof_north_low, color=color, alpha=0.25, label=f'3σ Bounds ({noise:.1f}m)')
        axes[1, 0].plot(time, north_spoof, linewidth=2.5, color=color, label=f'{noise:.1f}m Spoofing', zorder=9)
        axes[1, 0].set_ylabel('North Position (m)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('North Position vs. Time', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(loc='best', fontsize=9)
        
        # Up position
        axes[2, 0].fill_between(time, up_up, up_low, color='black', alpha=0.25, label='3σ Bounds (No Spoofing)')
        axes[2, 0].plot(time, up_no_spoof, linewidth=2.5, color='black', label='No Spoofing', zorder=10)
        axes[2, 0].fill_between(time, spoof_up_up, spoof_up_low, color=color, alpha=0.25, label=f'3σ Bounds ({noise:.1f}m)')
        axes[2, 0].plot(time, up_spoof, linewidth=2.5, color=color, label=f'{noise:.1f}m Spoofing', zorder=9)
        axes[2, 0].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        axes[2, 0].set_ylabel('Up Position (m)', fontsize=11, fontweight='bold')
        axes[2, 0].set_title('Up Position vs. Time', fontsize=11)
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].legend(loc='best', fontsize=9)
        
        # ===== COLUMN 2: Errors (Spoofed - Unspoofed) =====
        # East error
        axes[0, 1].fill_between(time, error_east_up, error_east_low, color=color, alpha=0.25, label='3σ Combined Error Bounds')
        axes[0, 1].plot(time, error_east, linewidth=2.5, color=color, label=f'{noise:.1f}m Error', zorder=9)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=2, label='Zero Error', zorder=8)
        axes[0, 1].set_ylabel('East Error (m)', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('East Position Error (Spoofed - Unspoofed)', fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(loc='best', fontsize=9)
        
        # North error
        axes[1, 1].fill_between(time, error_north_up, error_north_low, color=color, alpha=0.25, label='3σ Combined Error Bounds')
        axes[1, 1].plot(time, error_north, linewidth=2.5, color=color, label=f'{noise:.1f}m Error', zorder=9)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=2, label='Zero Error', zorder=8)
        axes[1, 1].set_ylabel('North Error (m)', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('North Position Error (Spoofed - Unspoofed)', fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(loc='best', fontsize=9)
        
        # Up error
        axes[2, 1].fill_between(time, error_up_up, error_up_low, color=color, alpha=0.25, label='3σ Combined Error Bounds')
        axes[2, 1].plot(time, error_up, linewidth=2.5, color=color, label=f'{noise:.1f}m Error', zorder=9)
        axes[2, 1].axhline(y=0, color='black', linestyle='--', linewidth=2, label='Zero Error', zorder=8)
        axes[2, 1].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        axes[2, 1].set_ylabel('Up Error (m)', fontsize=11, fontweight='bold')
        axes[2, 1].set_title('Up Position Error (Spoofed - Unspoofed)', fontsize=11)
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        plot_path = os.path.join(results_dir, f'spoofing_comparison_{noise:.1f}m.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_path}")
        plt.close(fig)
    
    print(f"\n✓ Spoofing analysis complete. Results saved to: {results_dir}/")


if __name__ == "__main__":
    import sys
    import os
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run EKF and MEKF algorithms on field data')
    parser.add_argument('file', nargs='?', default='data.csv', help='Data file to process')
    parser.add_argument('--trials', action='store_true', help='Process all trial CSV files (Trial_1.csv, Trial_2.csv, Trial_3.csv)')
    parser.add_argument('--spoofing', action='store_true', help='Run GPS spoofing analysis with 10 different noise levels (1-100m)')
    
    args = parser.parse_args()
    
    # Determine which mode to run in
    if args.spoofing:
        # Run spoofing analysis on all trial files
        trial_files = ['field_data/Trial_1.csv', 'field_data/Trial_2.csv', 'field_data/Trial_3.csv']
        
        for trial_file in trial_files:
            if os.path.exists(trial_file):
                trial_name = trial_file.split('/')[-1].replace('.csv', '')
                results_dir = f'results/spoofing_analysis/{trial_name}'
                print(f"\n{'='*60}")
                print(f"Running spoofing analysis for {trial_name}")
                print(f"{'='*60}")
                run_spoofing_analysis(trial_file, results_dir)
            else:
                print(f"Warning: Trial file not found: {trial_file}")
        
        sys.exit(0)
    elif args.trials:
        # Process all trial files
        trial_files = ['field_data/Trial_1.csv', 'field_data/Trial_2.csv', 'field_data/Trial_3.csv']
        files_to_process = [(f, f'results/{f.split("/")[1].replace(".csv", "")}') for f in trial_files]
    else:
        # Process single file specified as argument
        data_file = args.file
        # Extract test name from filename (e.g., simulated_data_linear_motion.csv -> linear_motion)
        if 'simulated_data_' in data_file:
            test_name = data_file.replace('simulated_data_', '').replace('.csv', '')
            results_dir = f'results/{test_name}'
        else:
            results_dir = 'results'
        files_to_process = [(data_file, results_dir)]
    
    # Process each file
    for data_file, results_dir in files_to_process:
        print("\n" + "=" * 60)
        print("Starting EKF and MEKF algorithms")
        print("=" * 60)
        print(f"Loading data from: {data_file}")
        print(f"Saving results to: {results_dir}")
        print("=" * 60)
        
        # Call main function with custom results directory
        x_k_storage, P_k_storage, quaternion_storage, mekf_cov_storage, constants, prefit_residuals, postfit_residuals = main(data_file, results_dir)
        
        print("=" * 60)
        print("EKF and MEKF algorithms completed")
        print(f"Processed {x_k_storage.shape[1]} time steps")
        
        # ============ SAVE RESULTS ============
        print("\nSaving results...")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save EKF results
        np.save(f'{results_dir}/ekf_position_estimates.npy', x_k_storage)
        np.save(f'{results_dir}/ekf_covariance.npy', P_k_storage)
        print(f"✓ EKF results saved to {results_dir}/ekf_position_estimates.npy and {results_dir}/ekf_covariance.npy")
        
        # Save MEKF results
        with open(f'{results_dir}/mekf_quaternions.txt', 'w') as f:
            f.write("Time_Step,Qw,Qx,Qy,Qz\n")
            for i, q in enumerate(quaternion_storage):
                f.write(f"{i},{q.w},{q.x},{q.y},{q.z}\n")
        print(f"✓ MEKF quaternion estimates saved to {results_dir}/mekf_quaternions.txt")
        
        np.save(f'{results_dir}/mekf_covariance.npy', mekf_cov_storage)
        print(f"✓ MEKF covariance saved to {results_dir}/mekf_covariance.npy")
        
        
        # ============ PLOT RESULTS ============
        print("\nPlotting results...")
        

        # Convert EKF results to format expected by plot_results
        # plot_results expects: state as list of 3-element vectors, cov as list of 3x3 matrices
        state_list = [x_k_storage[:3, i] for i in range(x_k_storage.shape[1])]
        cov_list = []
        # Extract 3x3 position covariances from 9x9 matrices
        for i in range(P_k_storage.shape[2]):
            cov_list.append(P_k_storage[0:3, 0:3, i])
        
        # Plot EKF position estimates (with 3σ bounds)
        print("Generating EKF position plots...")
        plot_results(state_list, cov_list, save_dir=results_dir, show=False, dt=constants['dt'])
        
        # Plot MEKF attitudes (Euler angles with 3σ bounds)
        print("Generating MEKF attitude plots...")
        plot_euler_angles(quaternion_storage, covariances=mekf_cov_storage, 
                        save_dir=results_dir, show=False, dt=constants['dt'])
        
        # Plot pre-fit and post-fit residuals
        print("Generating residual plots...")
        plot_residuals(prefit_residuals, postfit_residuals, constants['dt'], results_dir)
        
        print(f"✓ Results plotted and saved to {results_dir}/")

        # ============ PRINT SUMMARY ============å
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60) 
        print(f"EKF Position State Vector Shape: {x_k_storage.shape}")
        print(f"  - Position estimates (9 states) x {x_k_storage.shape[1]} time steps")
        print(f"  - State: [x, y, z, vx, vy, vz, ax, ay, az]")
        
        print(f"\nMEKF Attitude Estimates: {len(quaternion_storage)} quaternions")
        print(f"  - Initial quaternion: {quaternion_storage[0]}")
        print(f"  - Final quaternion: {quaternion_storage[-1]}")
        
        print(f"\nAll results have been saved to the '{results_dir}/' directory")
        print("=" * 60)
    
