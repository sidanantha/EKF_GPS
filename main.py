# main.py
# Main file to run the EKF algorithm

import EKF

# Load data
data = pd.read_csv('data.csv')

# Run EKF
x_k, P_k = EKF.EKF_iteration(data)

