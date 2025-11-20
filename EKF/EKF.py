# EKF.py
# Main file to run EKF algorithm

import csv
import pandas as pd
import numpy as np


# Generate Matricies:
def build_A(dt):
    '''
    Builds the A matrix for the EKF.
    Inputs:
        dt: Time step, scalar
    Output:
        A: A matrix, 9x9 matrix
    '''
    
    A = np.zeros((9, 9))
    # Diag:
    A[0:3, 0:3] = np.eye(3)
    A[3:6, 3:6] = np.eye(3)
    A[6:9, 6:9] = np.eye(3)
    # Off-diag:
    A[0:3, 3:6] = np.eye(3) * dt
    A[0:3, 6:9] = (1/2) * dt**2 * np.eye(3)
    A[3:6, 6:9] = np.eye(3) * dt
    return A

def build_B(m, dt):
    '''
    Builds the B matrix for the EKF.
    Inputs:
        m: Mass, scalar
        dt: Time step, scalar
    Output:
        B: B matrix, 9x3 matrix
    '''
    B = np.zeros((9, 3))
    
    B[6:9, 0:3] = (1/m) * dt * np.eye(3)
    return B

def build_C(m, dt):
    '''
    Builds the C matrix for the EKF.
    Inputs:
        m: Mass, scalar
        dt: Time step, scalar
    Output:
        C: C matrix, 6x9 matrix
        Maps: [x, y, z, vx, vy, vz, ax, ay, az] to [px, py, pz, ax, ay, az]
    '''
    C = np.zeros((6, 9))
    C[0:3, 0:3] = np.eye(3)  # Position measurement
    C[3:6, 6:9] = np.eye(3)  # Acceleration measurement
    return C

def build_Q(sigma_IMU_x, sigma_IMU_y, sigma_IMU_z):
    '''
    Builds the Q matrix for the EKF.
    Inputs:
        sigma_IMU_x: Standard deviation of the IMU x acceleration
        sigma_IMU_y: Standard deviation of the IMU y acceleration
        sigma_IMU_z: Standard deviation of the IMU z acceleration
    Output:
        Q: Q matrix (process noise covariance)
    '''
    Q = np.zeros((9, 9))
    # Reduce acceleration noise by squaring sigma_IMU to make GPS measurements more influential
    Q[6:9, 6:9] = np.diag([sigma_IMU_x**2, sigma_IMU_y**2, sigma_IMU_z**2])
    return Q


def build_R(sigma_GPS_x, sigma_GPS_y, sigma_GPS_z, sigma_IMU_x, sigma_IMU_y, sigma_IMU_z):
    '''
    Builds the R matrix for the EKF.
    Inputs:
        sigma_GPS_x: Standard deviation of the GPS x position
        sigma_GPS_y: Standard deviation of the GPS y position
        sigma_GPS_z: Standard deviation of the GPS z position
        sigma_IMU_x: Standard deviation of the IMU x acceleration
        sigma_IMU_y: Standard deviation of the IMU y acceleration
        sigma_IMU_z: Standard deviation of the IMU z acceleration
    Output:
        R: R matrix (6x6 for position and acceleration measurements)
    '''
    
    R = np.zeros((6, 6))
    R[0:3, 0:3] = np.diag([sigma_GPS_x**2, sigma_GPS_y**2, sigma_GPS_z**2])  # GPS position noise (lower, trusted more)
    R[3:6, 3:6] = np.diag([sigma_IMU_x**2, sigma_IMU_y**2, sigma_IMU_z**2])  # IMU acceleration noise (higher, trusted less)
    return R
    
# Main EKF Algorithm for one iteration:
def EKF_iteration(x_k, P_k, z_k, u_k, R_k, Q_k, A_k, B_k, C_k, sigma_IMU, sigma_GPS):
    '''
    Main EKF algorithm for one iteration.
    Inputs:
        x_k: State vector at time k, 9x1 matrix
        P_k: Covariance matrix at time k, 9x9 matrix
        z_k: Measurement vector at time k, 6x1 matrix
        R_k: Measurement noise covariance matrix, 6x6 matrix
        Q_k: Process noise covariance matrix, 9x9 matrix
        A_k: A matrix at time k, 9x9 matrix
        B_k: B matrix at time k, 9x3 matrix
        C_k: C matrix at time k, 6x9 matrix
        sigma_IMU: Standard deviation of the IMU, vector of 3 elements
        sigma_GPS: Standard deviation of the GPS, vector of 3 elements
    Output:
        x_k+1: State vector at time k+1, 9x1 matrix
        P_k+1: Covariance matrix at time k+1, 9x9 matrix
    '''
    
    # Prediction Step:
    # Only acceleration components should have noise
    # w_k = np.zeros((9, 1))
    # w_k[6:9, 0] = np.random.normal(0, sigma_IMU, 3) # Process noise for acceleration
    
    x_k_next = A_k @ x_k + B_k @ u_k # + w_k
    P_k_next = A_k @ P_k @ A_k.T + Q_k
    
    # Pre-fit residual (innovation)
    prefit_residual = z_k - C_k @ x_k_next
    
    # Update Step:
    K = P_k_next @ C_k.T @ np.linalg.inv(C_k @ P_k_next @ C_k.T + R_k)
    x_k_next = x_k_next + K @ prefit_residual
    P_k_next = (np.eye(9) - K @ C_k) @ P_k_next
    
    # Post-fit residual
    postfit_residual = z_k - C_k @ x_k_next
    
    return x_k_next, P_k_next, prefit_residual, postfit_residual