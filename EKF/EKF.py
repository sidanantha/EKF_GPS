# EKF.py
# Main file to run EKF algorithm

import csv
import pandas as pd


def data_processing(filepath):
    df = pd.read_csv(filepath)
    time = df['UTC'].to_numpy()
    lat = df['Lat'].to_numpy()
    long = df['Long'].to_numpy()
    alt = df['Alt'].to_numpy()
    x = df['ECEF_X'].to_numpy()
    y = df['ECEF_Y'].to_numpy()
    z = df['ECEF_Z'].to_numpy()
    ax = df['ax'].to_numpy()
    ay = df['ay'].to_numpy()
    az = df['az'].to_numpy()
    gx = df['gx'].to_numpy()
    gy = df['gy'].to_numpy()
    gz = df['gz'].to_numpy()

    return time, lat, long, alt, x, y, z, ax, ay, az, gx, gy, gz



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
    A[0:3, 0:3] = eye(3)
    A[3:6, 3:6] = eye(3)
    A[6:9, 6:9] = eye(3)
    # Off-diag:
    A[0:3, 3:6] = eye(3) * dt
    A[0:3, 6:9] = (1/2) * dt**2 * eye(3)
    A[3:6, 6:9] = eye(3) * dt
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
    
    B[6:9, 0:3] = (1/m) * dt * eye(3)
    

def build_C(m, dt):
    '''
    Builds the C matrix for the EKF.
    Inputs:
        m: Mass, scalar
        dt: Time step, scalar
    Output:
        C: C matrix, 6x9 matrix
    '''
    C = np.zeros((6, 9))
    C[0:3, 0:3] = np.eye(3)
    C[6:9, 6:9] = np.eye(3)
    return C

def build_Q(sigma_IMU):
    '''
    Builds the Q matrix for the EKF.
    Inputs:
        sigma_IMU: Standard deviation of the IMU, scalar
    Output:
        Q: Q matrix
    '''
    Q = np.zeros((9, 9))
    Q[6:9, 6:9] = sigma_IMU**2 * eye(3)
    return Q


def build_R(sigma_GPS):
    '''
    Builds the R matrix for the EKF.
    Inputs:
        sigma_GPS: Standard deviation of the GPS
    Output:
        R: R matrix
    '''
    R = np.zeros((6, 6))
    R[0:3, 0:3] = sigma_GPS**2 * np.eye(3)
    
    
# Main EKF Algorithm for one iteration:
def EKF_iteration(x_k, P_k, z_k, R_k, Q_k, A_k, B_k, C_k):
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
        
    Output:
        x_k+1: State vector at time k+1, 9x1 matrix
        P_k+1: Covariance matrix at time k+1, 9x9 matrix
    '''
    
    # Prediction Step:
    x_k+1 = A_k * x_k + B_k * u_k
    P_k+1 = A_k * P_k * A_k.T + Q_k
    
    # Update Step:
    K = P_k+1 * C_k.T * (C_k * P_k+1 * C_k.T + R_k).inv()
    x_k+1 = x_k+1 + K * (z_k - C_k * x_k+1)
    P_k+1 = (eye(9) - K * C_k) * P_k+1