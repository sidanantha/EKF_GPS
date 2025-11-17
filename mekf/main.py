import numpy as np
import numpy.random as npr
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import math
import model
from referencevectorgauge import ReferenceVectorGauge
from noisydevice import NoisyDeviceDecorator
import gyro
from MEKF import MEKF

def quatToEuler(q):
    euler = []
    x = math.atan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]*q[1] + q[2]*q[2]))
    euler.append(x)

    x = math.asin(2*(q[0]*q[2] - q[3]*q[1]))
    euler.append(x)

    x = math.atan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]*q[2] + q[3]*q[3]))
    euler.append(x)

    return euler

def quatListToEulerArrays(qs):
    euler = np.ndarray(shape=(3, len(qs)), dtype=float)

    for (i, q) in enumerate(qs):
        e = quatToEuler(q)
        euler[0, i] = e[0]
        euler[1, i] = e[1]
        euler[2, i] = e[2]

    return euler

def eulerError(estimate, truth):
    return np.minimum(np.minimum(np.abs(estimate - truth), np.abs(2*math.pi + estimate - truth)),
                                 np.abs(-2*math.pi + estimate - truth))

def eulerArraysToErrorArrays(estimate, truth):
    errors = []
    for i in range(3):
        errors.append(eulerError(estimate[i], truth[i]))
    return errors

def quatListToErrorArrays(estimate, truth):
    return eulerArraysToErrorArrays(quatListToEulerArrays(estimate), quatListToEulerArrays(truth))

def rmse_euler(estimate, truth):
    def rmse(vec1, vec2):
        return np.sqrt(np.mean((vec1 - vec2)**2))

    return[rmse(estimate[0], truth[0]), 
           rmse(estimate[1], truth[1]),
           rmse(estimate[2], truth[2])]

if __name__ == '__main__':
    accel_cov = 0.001
    accel_bias = np.array([0.6, 0.02, 0.05])
    mag_cov = 0.001
    mag_bias = np.array([0.03, 0.08, 0.04])
    gyro_cov = 0.1
    gyro_bias = np.array([0.25, 0.01, 0.05])
    gyro_bias_drift = 0.0001
    gyro = NoisyDeviceDecorator(gyro.Gyro(), gyro_bias, gyro_cov, gyro_bias_drift)
    accelerometer = NoisyDeviceDecorator(ReferenceVectorGauge(np.array([0, 0, -1])), 
            accel_bias, accel_cov, bias_drift_covariance = 0.0) 
    magnetometer = NoisyDeviceDecorator(ReferenceVectorGauge(np.array([1, 0, 0])), 
            mag_bias, mag_cov, bias_drift_covariance = 0.0) 
    real_measurement = np.array([0.0, 0.0, 0.0])
    time_delta = 0.005
    true_orientation = model.Model(Quaternion(axis = [1, 0, 0], angle=0))
    dead_reckoning_estimate = model.Model(Quaternion(axis = [1, 0, 0], angle=0))
    true_rotations = []
    dead_reckoning_rotation_estimates = []
    filtered_rotation_estimates = []

    kalman_filter = MEKF(true_orientation.orientation, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1)
    for i in range(4000):

        if (i % 10 == 0):
          real_measurement = npr.normal(0.0, 1.0, 3)

        gyro_measurement = gyro.measure(time_delta, real_measurement)

        dead_reckoning_estimate.update(time_delta, gyro_measurement)
        dead_reckoning_rotation_estimates.append(dead_reckoning_estimate.orientation)

        true_orientation.update(time_delta, real_measurement)
        true_rotations.append(true_orientation.orientation)
        
        measured_acc = accelerometer.measure(time_delta, true_orientation.orientation)
        measured_mag = magnetometer.measure(time_delta, true_orientation.orientation)

        kalman_filter.update(gyro_measurement, measured_acc, time_delta)
        filtered_rotation_estimates.append(kalman_filter.estimate)

    #print "gyro bias: ", kalman_filter.gyro_bias
    #print "accel bias: ", kalman_filter.accelerometer_bias
    #print "mag bias: ", kalman_filter.magnetometer_bias

    dead_reckoning_errors = quatListToErrorArrays(dead_reckoning_rotation_estimates, true_rotations)
    filtered_errors = quatListToErrorArrays(filtered_rotation_estimates, true_rotations)

    # Original error plot (as it was before)
    fig_orig, ax_orig = plt.subplots(figsize=(12, 8))
    unfiltered_roll, = ax_orig.plot(dead_reckoning_errors[0], 'b--', linewidth=1.5, label='dead reckoning roll')
    unfiltered_pitch, = ax_orig.plot(dead_reckoning_errors[1], 'b-', linewidth=1.5, alpha=0.7, label='dead reckoning pitch')
    unfiltered_yaw, = ax_orig.plot(dead_reckoning_errors[2], 'b:', linewidth=1.5, alpha=0.7, label='dead reckoning yaw')
    filtered_roll, = ax_orig.plot(filtered_errors[0], 'r--', linewidth=1.5, label='mekf roll')
    filtered_pitch, = ax_orig.plot(filtered_errors[1], 'r-', linewidth=1.5, alpha=0.7, label='mekf pitch')
    filtered_yaw, = ax_orig.plot(filtered_errors[2], 'r:', linewidth=1.5, alpha=0.7, label='mekf yaw')
    ax_orig.legend(handles=[unfiltered_roll, 
                        unfiltered_pitch, 
                        unfiltered_yaw, 
                        filtered_roll, 
                        filtered_pitch, 
                        filtered_yaw], loc='upper right')
    ax_orig.set_xlabel("Discrete time (5ms steps)", fontweight='bold')
    ax_orig.set_ylabel("Error (in radians)", fontweight='bold')
    ax_orig.set_title("Original Error Plot: Dead Reckoning vs MEKF", fontsize=12, fontweight='bold')
    ax_orig.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Convert quaternions to Euler angles for all three trajectories
    true_euler = quatListToEulerArrays(true_rotations)
    dead_reckoning_euler = quatListToEulerArrays(dead_reckoning_rotation_estimates)
    filtered_euler = quatListToEulerArrays(filtered_rotation_estimates)
    
    time_steps = np.arange(len(true_rotations))
    
    # Create 3x1 figure for Euler angle comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Attitude Estimation Comparison: True vs Dead Reckoning vs MEKF', fontsize=14, fontweight='bold')
    
    # Roll (φ)
    axes[0].plot(time_steps, np.degrees(true_euler[0]), 'k-', linewidth=2, label='True Roll')
    axes[0].plot(time_steps, np.degrees(dead_reckoning_euler[0]), 'b--', linewidth=1.5, alpha=0.7, label='Dead Reckoning Roll')
    axes[0].plot(time_steps, np.degrees(filtered_euler[0]), 'r-', linewidth=1.5, alpha=0.7, label='MEKF Roll')
    axes[0].set_ylabel('Roll (degrees)', fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Roll (φ) - Rotation around X-axis')
    
    # Pitch (θ)
    axes[1].plot(time_steps, np.degrees(true_euler[1]), 'k-', linewidth=2, label='True Pitch')
    axes[1].plot(time_steps, np.degrees(dead_reckoning_euler[1]), 'b--', linewidth=1.5, alpha=0.7, label='Dead Reckoning Pitch')
    axes[1].plot(time_steps, np.degrees(filtered_euler[1]), 'r-', linewidth=1.5, alpha=0.7, label='MEKF Pitch')
    axes[1].set_ylabel('Pitch (degrees)', fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Pitch (θ) - Rotation around Y-axis')
    
    # Yaw (ψ)
    axes[2].plot(time_steps, np.degrees(true_euler[2]), 'k-', linewidth=2, label='True Yaw')
    axes[2].plot(time_steps, np.degrees(dead_reckoning_euler[2]), 'b--', linewidth=1.5, alpha=0.7, label='Dead Reckoning Yaw')
    axes[2].plot(time_steps, np.degrees(filtered_euler[2]), 'r-', linewidth=1.5, alpha=0.7, label='MEKF Yaw')
    axes[2].set_ylabel('Yaw (degrees)', fontweight='bold')
    axes[2].set_xlabel('Time steps (5ms each)', fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Yaw (ψ) - Rotation around Z-axis')
    
    plt.tight_layout()
    plt.show()
    
    # Also create error plot for reference
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10))
    fig2.suptitle('Attitude Estimation Error (vs True)', fontsize=14, fontweight='bold')
    
    axes2[0].plot(time_steps, dead_reckoning_errors[0], 'b--', linewidth=1.5, alpha=0.7, label='Dead Reckoning Error')
    axes2[0].plot(time_steps, filtered_errors[0], 'r-', linewidth=1.5, alpha=0.7, label='MEKF Error')
    axes2[0].set_ylabel('Roll Error (radians)', fontweight='bold')
    axes2[0].legend(loc='upper right')
    axes2[0].grid(True, alpha=0.3)
    axes2[0].set_title('Roll Error')
    
    axes2[1].plot(time_steps, dead_reckoning_errors[1], 'b--', linewidth=1.5, alpha=0.7, label='Dead Reckoning Error')
    axes2[1].plot(time_steps, filtered_errors[1], 'r-', linewidth=1.5, alpha=0.7, label='MEKF Error')
    axes2[1].set_ylabel('Pitch Error (radians)', fontweight='bold')
    axes2[1].legend(loc='upper right')
    axes2[1].grid(True, alpha=0.3)
    axes2[1].set_title('Pitch Error')
    
    axes2[2].plot(time_steps, dead_reckoning_errors[2], 'b--', linewidth=1.5, alpha=0.7, label='Dead Reckoning Error')
    axes2[2].plot(time_steps, filtered_errors[2], 'r-', linewidth=1.5, alpha=0.7, label='MEKF Error')
    axes2[2].set_ylabel('Yaw Error (radians)', fontweight='bold')
    axes2[2].set_xlabel('Time steps (5ms each)', fontweight='bold')
    axes2[2].legend(loc='upper right')
    axes2[2].grid(True, alpha=0.3)
    axes2[2].set_title('Yaw Error')
    
    plt.tight_layout()
    plt.show()
