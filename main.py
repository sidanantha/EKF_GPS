# main.py
# Main file to run the EKF algorithm for position and MEKF for attitude

import sys
import EKF.EKF as EKF
import data_processing.dataprocessing as data_processing
import utils.utils as utils
import numpy as np
from pyquaternion import Quaternion
from mekf.MEKF import MEKF


# Define constants
def define_constants():
    '''
    Define constants for the EKF and MEKF algorithms
    '''
    constants = {
        'dt': 0.01, # Time step
        'm': 1000, # Mass
        'I': np.eye(3), # Inertial matrix
        'sigma_IMU': 0.1, # Standard deviation of the IMU
        'sigma_GPS': 0.1, # Standard deviation of the GPS
        'initial_uncertainty': 1000, # Initial uncertainty for the state vector
        'coordinates_observer': np.array([0, 0, 0]), # Observer position, latitude, longitude, altitude
        # MEKF parameters
        'estimate_covariance': 1.0,
        'gyro_cov': 0.1,
        'gyro_bias_cov': 0.1,
        'accel_proc_cov': 0.1,
        'accel_bias_cov': 0.1,
        'accel_obs_cov': 0.1,
    }
    return constants



# Main Function:
def main():
    '''
    Main function to run both EKF (for position) and MEKF (for attitude) algorithms
    '''
    # Define constants
    constants = define_constants()
    # Determine observer position:
    x_observer = utils.ecef_to_lla(constants['coordinates_observer'][0], constants['coordinates_observer'][1], constants['coordinates_observer'][2])
    
    # Load data
    time, starttime, endtime, timestep, lat, lon, alt, x, y, z, ax, ay, az, gx, gy, gz = data_processing.data_processing('data.csv')
    
    # ============ EKF INITIALIZATION (for position estimation) ============
    # Build EKF matrices:
    A = EKF.build_A(constants['dt'])
    B = np.zeros((9, 3))    # assume no control input
    C = EKF.build_C(constants['m'], constants['dt'])
    Q = EKF.build_Q(constants['sigma_IMU'])
    R = EKF.build_R(constants['sigma_GPS'])
    
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
    
    # ============ MAIN LOOP ============
    for i in range(len(time)):
        
        # Create acceleration and gyro vectors
        accel_meas = np.array([ax[i], ay[i], az[i]])
        gyro_meas = np.array([gx[i], gy[i], gz[i]])
        
        # ========== RUN EKF (Position Estimation) ==========
        # Define the measurement vector:
        x_drone = np.array([x[i], y[i], z[i]])
        y_t = np.zeros((6,1))
        y_t[0] = utils.relative_position(x_observer, x_drone)[0]
        y_t[1] = utils.relative_position(x_observer, x_drone)[1]
        y_t[2] = utils.relative_position(x_observer, x_drone)[2]
        y_t[3] = ax[i]
        y_t[4] = ay[i]
        y_t[5] = az[i]
        
        # Run EKF iteration
        x_k, P_k = EKF.EKF_iteration(x_k, P_k, y_t, np.zeros((3,1)), R, Q, A, B, C, constants['sigma_IMU'], constants['sigma_GPS'])
        x_k_storage[:, i] = x_k.flatten()
        P_k_storage[:, :, i] = P_k
        
        # ========== RUN MEKF (Attitude Estimation) ==========
        mekf.update(gyro_meas, accel_meas, constants['dt'])
        quaternion_storage.append(mekf.estimate)
        mekf_cov_storage[:, :, i] = mekf.estimate_covariance
    
    return x_k_storage, P_k_storage, quaternion_storage, mekf_cov_storage




if __name__ == "__main__":
    print("Starting EKF and MEKF algorithms")
    print("=" * 60)
    
    # Call main function
    x_k_storage, P_k_storage, quaternion_storage, mekf_cov_storage = main()
    
    print("=" * 60)
    print("EKF and MEKF algorithms completed")
    print(f"Processed {x_k_storage.shape[1]} time steps")
    
    # ============ SAVE RESULTS ============
    print("\nSaving results...")
    
    # Save EKF results
    np.save('results/ekf_position_estimates.npy', x_k_storage)
    np.save('results/ekf_covariance.npy', P_k_storage)
    print("✓ EKF results saved to results/ekf_position_estimates.npy and results/ekf_covariance.npy")
    
    # Save MEKF results
    with open('results/mekf_quaternions.txt', 'w') as f:
        f.write("Time_Step,Qw,Qx,Qy,Qz\n")
        for i, q in enumerate(quaternion_storage):
            f.write(f"{i},{q.w},{q.x},{q.y},{q.z}\n")
    print("✓ MEKF quaternion estimates saved to results/mekf_quaternions.txt")
    
    np.save('results/mekf_covariance.npy', mekf_cov_storage)
    print("✓ MEKF covariance saved to results/mekf_covariance.npy")
    
    
    # ============ PLOT RESULTS ============
    print("\nPlotting results...")
    plot_results(x_k_storage, P_k_storage, quaternion_storage, mekf_cov_storage)
    print("✓ Results plotted")
    
    # ============ PRINT SUMMARY ============
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"EKF Position State Vector Shape: {x_k_storage.shape}")
    print(f"  - Position estimates (9 states) x {x_k_storage.shape[1]} time steps")
    print(f"  - State: [x, y, z, vx, vy, vz, ax, ay, az]")
    
    print(f"\nMEKF Attitude Estimates: {len(quaternion_storage)} quaternions")
    print(f"  - Initial quaternion: {quaternion_storage[0]}")
    print(f"  - Final quaternion: {quaternion_storage[-1]}")
    
    print("\nAll results have been saved to the 'results/' directory")
    print("=" * 60)
    
