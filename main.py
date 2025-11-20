# main.py
# Main file to run the EKF algorithm for position and MEKF for attitude

import sys
import EKF.EKF as EKF
import data_processing.dataprocessing as data_processing
import utils.utils as utils
import numpy as np
from pyquaternion import Quaternion
from mekf.MEKF import MEKF
from data_processing.plotdata import plot_results, plot_euler_angles


# Define constants
def define_constants():
    '''
    Define constants for the EKF and MEKF algorithms
    '''
    constants = {
        'dt': 0.1, # Time step
        'm': 1000, # Mass
        'I': np.eye(3), # Inertial matrix
        'sigma_IMU': 0.05, # Standard deviation of the IMU (accelerometer noise)
        'sigma_GPS': 1.0, # Standard deviation of the GPS (1 meter noise in simulated data)
        'GPS_spoof_noise': 100, # Additional GPS noise to inject (meters) - set to 0 for no spoofing
        'GPS_spoof_start_time': -1, # Time (seconds) to start GPS spoofing - set to -1 to disable
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
        # a_z free fall calibration exists but it seems very noisy. Should not use.
    }
    return constants



# Main Function:
def main(data_file='data.csv', results_dir='results'):
    '''
    Main function to run both EKF (for position) and MEKF (for attitude) algorithms
    
    Args:
        data_file: Path to CSV file with measurement data (default: 'data.csv')
        results_dir: Directory to save results (default: 'results')
    '''
    # Define constants
    constants = define_constants()
    # Determine observer position:
    x_observer = utils.ecef_to_lla(constants['coordinates_observer'][0], constants['coordinates_observer'][1], constants['coordinates_observer'][2])
    
    # Load data
    time, starttime, endtime, timestep, lat, lon, alt, x, y, z, ax, ay, az, gx, gy, gz = data_processing.data_processing(data_file)

    # Remove GPS bias
    x -= -0.05622031074017286
    y -= -0.4507213160395622
    z -= -1.5610635122284293
    
    # ============ EKF INITIALIZATION (for position estimation) ============
    # Build EKF matrices:
    A = EKF.build_A(constants['dt'])
    B = np.zeros((9, 3))    # assume no control input
    C = EKF.build_C(constants['m'], constants['dt'])
    Q = EKF.build_Q(constants['sigma_IMU'])
    # Build R with much higher noise for IMU measurements to trust GPS significantly more
    # GPS noise: 1.0 m, IMU acceleration noise: heavily downweighted to prevent drift
    R = EKF.build_R(constants['sigma_GPS'], sigma_IMU_meas=constants['sigma_IMU'] * 1)
    
    # Init EKF state vector and covariance matrix:
    x_k = np.zeros((9,1))
    P_k = np.eye(9)*constants['initial_uncertainty']
    
    # Init EKF storage:
    x_k_storage = np.zeros((9, len(time)))
    P_k_storage = np.zeros((9, 9, len(time)))
    
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
        roll = 0    # Temp, do not use MEKF
        pitch = 0   # Temp, do not use MEKF
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
        x_k, P_k = EKF.EKF_iteration(x_k, P_k, y_t, np.zeros((3,1)), R, Q, A, B, C, constants['sigma_IMU'], constants['sigma_GPS'])
        x_k_storage[:, i] = x_k.flatten()
        P_k_storage[:, :, i] = P_k
        
        # ========== STORE MEKF RESULTS ==========
        quaternion_storage.append(mekf.estimate)
        mekf_cov_storage[:, :, i] = mekf.estimate_covariance
    
    return x_k_storage, P_k_storage, quaternion_storage, mekf_cov_storage, constants




if __name__ == "__main__":
    import sys
    import os
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run EKF and MEKF algorithms on field data')
    parser.add_argument('file', nargs='?', default='data.csv', help='Data file to process')
    parser.add_argument('--trials', action='store_true', help='Process all trial CSV files (Trial_1.csv, Trial_2.csv, Trial_3.csv)')
    
    args = parser.parse_args()
    
    # Determine which files to process
    if args.trials:
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
        x_k_storage, P_k_storage, quaternion_storage, mekf_cov_storage, constants = main(data_file, results_dir)
        
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
    
