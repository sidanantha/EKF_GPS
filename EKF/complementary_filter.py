# Complementary Filter for Attitude Estimation

import numpy as np

def complementary_filter(gyro_meas, state_accel, state_prev, alpha=np.array([0.5, 0.5, 0.5]), time_delta=0.1):
    '''
    Complementary filter for attitude estimation.
    Inputs:
        gyro_meas: Gyroscope measurements, 3x1 matrix
        acc_meas: Accelerometer measurements, 3x1 matrix
        state_prev: Previous state, 3x1 matrix
        alpha: Alpha parameter, 3x1 vector
        time_delta: Time step in seconds (default: 0.1)
    Output:
        state_comp: Complementary filtered state, 3x1 matrix
    '''
    
    # Solve for the gyro-based state:
    state_gyro = state_prev + gyro_meas * time_delta
    
    # Solve the complementary filter:
    state_comp = alpha * state_gyro + (1 - alpha) * state_accel
    
    return state_comp
