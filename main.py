# main.py
# Main file to run the EKF algorithm

import EKF.EKF as EKF
import data_processing.dataprocessing as data_processing
import utils.utils as utils
import numpy as np


# Define constants
def define_constants():
    '''
    Define constants for the EKF algorithm in a struct
    '''
    constants = {
        'dt': 0.01, # Time step
        'm': 1000, # Mass
        'I': np.eye(3), # Inertial matrix
        'sigma_IMU': 0.1, # Standard deviation of the IMU
        'sigma_GPS': 0.1, # Standard deviation of the GPS
        'initial_uncertainty': 1000, # Initial uncertainty for the state vector
        'coordinates_observer': np.array([0, 0, 0]), # Observer position, latitude, longitude, altitude
    }
    return constants



# Main Function:
def main():
    '''
    Main function to run the EKF algorithm
    '''
    # Define constants
    constants = define_constants()
    # Determine observer position:
    x_observer = utils.ecef_to_lla(constants['coordinates_observer'][0], constants['coordinates_observer'][1], constants['coordinates_observer'][2])
    
    # Load data
    time, starttime, endtime, timestep, lat, lon, alt, x, y, z, ax, ay, az, gx, gy, gz = data_processing.data_processing('data.csv')
    
    # Build each matrix:
    A = EKF.build_A(constants['dt'])
    B = B = np.zeros((9, 3))    # assume no control input
    C = EKF.build_C(constants['m'], constants['dt'])
    Q = EKF.build_Q(constants['sigma_IMU'])
    R = EKF.build_R(constants['sigma_GPS'])
    
    # Init state vector and covariance matrix:
    x_k = np.zeros((9,1))
    P_k = np.eye(9)*constants['initial_uncertainty']
    
    # Init storage:
    x_k_storage = np.zeros((9, len(time)))
    P_k_storage = np.zeros((9, 9, len(time)))
    
    # Loop through data and run EKF
    for i in range(len(time)):
        
        # Define the measurement vector:
        x_drone = np.array([x[i], y[i], z[i]])
        y_t =np.zeros((6,1))
        y_t[0] = utils.relative_position(x_observer, x_drone)[0]
        y_t[1] = utils.relative_position(x_observer, x_drone)[1]
        y_t[2] = utils.relative_position(x_observer, x_drone)[2]
        y_t[3] = ax[i]
        y_t[4] = ay[i]
        y_t[5] = az[i]
        
        # Run EKF
        x_k, P_k = EKF.EKF_iteration(x_k, P_k, y_t, np.zeros((3,1)), R, Q, A, B, C, constants['sigma_IMU'], constants['sigma_GPS'])
        x_k_storage[:, i] = x_k
        P_k_storage[:, :, i] = P_k
    
    return x_k_storage, P_k_storage




if __name__ == "__main__":
    print("Starting EKF algorithm")
    # Call main function
    main()
    print("EKF algorithm completed")
    # Call plot function
    # TODO @Nidhi: Implement plot function
    plot_results(x_k_storage, P_k_storage)
    print("Plotting completed")
    
