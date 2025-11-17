"""
Integrated EKF + MEKF Test Suite
================================

Tests the combined EKF and MEKF filters using simulated dynamics.
Generates ground truth trajectories and compares filter estimates.
"""

import sys
import os
import numpy as np
from pyquaternion import Quaternion
import math

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import EKF.EKF as EKF
from mekf.MEKF import MEKF
from data_processing.plotdata import plot_results


class SimulatedDynamics:
    """
    Simulates drone dynamics to generate ground truth trajectories
    and corresponding IMU measurements.
    """
    
    def __init__(self, dt=0.01, duration=10.0):
        """
        Initialize simulator.
        
        Args:
            dt: Time step
            duration: Total simulation time
        """
        self.dt = dt
        self.duration = duration
        self.t = np.arange(0, duration, dt)
        self.n_steps = len(self.t)
        
        # Ground truth trajectories (to be populated)
        self.position_truth = None      # (3, n_steps)
        self.velocity_truth = None      # (3, n_steps)
        self.acceleration_truth = None  # (3, n_steps)
        self.attitude_truth = None      # List of n_steps quaternions
        self.angular_velocity_truth = None  # (3, n_steps)
        
        # Sensor measurements
        self.accel_meas = None          # (3, n_steps) - with noise
        self.gyro_meas = None           # (3, n_steps) - with noise
        
        # Noise standard deviations
        self.accel_noise_std = 0.05     # m/s^2
        self.gyro_noise_std = 0.01      # rad/s
        self.gps_noise_std = 0.1        # m
        
    def generate_trajectory(self, trajectory_type='circular'):
        """
        Generate ground truth trajectory.
        
        Args:
            trajectory_type: 'linear', 'circular', 'helical'
        """
        if trajectory_type == 'circular':
            self._generate_circular_trajectory()
        elif trajectory_type == 'linear':
            self._generate_linear_trajectory()
        elif trajectory_type == 'helical':
            self._generate_helical_trajectory()
        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")
    
    def _generate_linear_trajectory(self):
        """Linear motion with constant velocity."""
        # Constant velocity motion
        v0 = np.array([10.0, 0.0, 0.0])  # 10 m/s in X direction
        
        self.position_truth = np.zeros((3, self.n_steps))
        self.velocity_truth = np.tile(v0[:, np.newaxis], (1, self.n_steps))
        self.acceleration_truth = np.zeros((3, self.n_steps))
        
        for i in range(1, self.n_steps):
            self.position_truth[:, i] = self.position_truth[:, i-1] + v0 * self.dt
    
    def _generate_circular_trajectory(self):
        """Circular motion in XY plane."""
        radius = 100.0  # 100 m radius
        period = 100.0  # 100 second period
        omega = 2 * np.pi / period  # Angular velocity
        
        self.position_truth = np.zeros((3, self.n_steps))
        self.velocity_truth = np.zeros((3, self.n_steps))
        self.acceleration_truth = np.zeros((3, self.n_steps))
        
        for i, t in enumerate(self.t):
            # Position
            self.position_truth[0, i] = radius * np.cos(omega * t)
            self.position_truth[1, i] = radius * np.sin(omega * t)
            self.position_truth[2, i] = 0.0
            
            # Velocity (tangential)
            self.velocity_truth[0, i] = -radius * omega * np.sin(omega * t)
            self.velocity_truth[1, i] = radius * omega * np.cos(omega * t)
            self.velocity_truth[2, i] = 0.0
            
            # Acceleration (centripetal)
            self.acceleration_truth[0, i] = -radius * omega**2 * np.cos(omega * t)
            self.acceleration_truth[1, i] = -radius * omega**2 * np.sin(omega * t)
            self.acceleration_truth[2, i] = 0.0
    
    def _generate_helical_trajectory(self):
        """Helical motion (circular in XY, rising in Z)."""
        radius = 50.0
        period = 50.0
        omega = 2 * np.pi / period
        vertical_velocity = 5.0  # m/s upward
        
        self.position_truth = np.zeros((3, self.n_steps))
        self.velocity_truth = np.zeros((3, self.n_steps))
        self.acceleration_truth = np.zeros((3, self.n_steps))
        
        for i, t in enumerate(self.t):
            # Position
            self.position_truth[0, i] = radius * np.cos(omega * t)
            self.position_truth[1, i] = radius * np.sin(omega * t)
            self.position_truth[2, i] = vertical_velocity * t
            
            # Velocity
            self.velocity_truth[0, i] = -radius * omega * np.sin(omega * t)
            self.velocity_truth[1, i] = radius * omega * np.cos(omega * t)
            self.velocity_truth[2, i] = vertical_velocity
            
            # Acceleration
            self.acceleration_truth[0, i] = -radius * omega**2 * np.cos(omega * t)
            self.acceleration_truth[1, i] = -radius * omega**2 * np.sin(omega * t)
            self.acceleration_truth[2, i] = 0.0
    
    def generate_attitude(self, attitude_type='yaw_only'):
        """
        Generate ground truth attitude (orientation).
        
        Args:
            attitude_type: 'static', 'yaw_only', 'rolling'
        """
        self.attitude_truth = []
        self.angular_velocity_truth = np.zeros((3, self.n_steps))
        
        if attitude_type == 'static':
            # No rotation - identity quaternion
            for i in range(self.n_steps):
                self.attitude_truth.append(Quaternion(axis=[1, 0, 0], angle=0))
        
        elif attitude_type == 'yaw_only':
            # Rotating around Z axis
            yaw_rate = 0.1  # rad/s
            for i, t in enumerate(self.t):
                angle = yaw_rate * t
                q = Quaternion(axis=[0, 0, 1], angle=angle)
                self.attitude_truth.append(q)
                self.angular_velocity_truth[:, i] = [0, 0, yaw_rate]
        
        elif attitude_type == 'rolling':
            # Rolling motion (rotation around X axis)
            roll_rate = 0.05  # rad/s
            for i, t in enumerate(self.t):
                angle = roll_rate * t
                q = Quaternion(axis=[1, 0, 0], angle=angle)
                self.attitude_truth.append(q)
                self.angular_velocity_truth[:, i] = [roll_rate, 0, 0]
        
        else:
            raise ValueError(f"Unknown attitude type: {attitude_type}")
    
    def calculate_imu_measurements(self, add_noise=True):
        """
        Calculate what IMU sensors should measure given true dynamics.
        
        IMU measures:
        - Accelerometer: Non-gravitational acceleration in body frame + gravity
        - Gyroscope: Angular velocity in body frame
        """
        self.accel_meas = np.zeros((3, self.n_steps))
        self.gyro_meas = np.zeros((3, self.n_steps))
        
        g = np.array([0, 0, 9.81])  # Gravity
        
        for i in range(self.n_steps):
            # Get true acceleration (without gravity)
            a_true = self.acceleration_truth[:, i]
            
            # Rotate to body frame (inverse of attitude quaternion)
            q_inv = self.attitude_truth[i].inverse
            a_body = q_inv.rotate(a_true + g)  # Add gravity
            
            # Gyroscope measures angular velocity in body frame
            omega_true = self.angular_velocity_truth[:, i]
            omega_body = q_inv.rotate(omega_true)
            
            self.accel_meas[:, i] = a_body
            self.gyro_meas[:, i] = omega_body
            
            # Add noise
            if add_noise:
                self.accel_meas[:, i] += np.random.normal(0, self.accel_noise_std, 3)
                self.gyro_meas[:, i] += np.random.normal(0, self.gyro_noise_std, 3)


class TestEKFMEKF:
    """Test suite for combined EKF + MEKF system."""
    
    def __init__(self, dt=0.01):
        self.dt = dt
        self.results = {}
    
    def test_linear_motion(self):
        """Test filters on linear motion."""
        print("\n" + "="*70)
        print("TEST 1: Linear Motion")
        print("="*70)
        
        # Generate trajectory
        sim = SimulatedDynamics(dt=self.dt, duration=10.0)
        sim.generate_trajectory('linear')
        sim.generate_attitude('static')
        sim.calculate_imu_measurements(add_noise=True)
        
        # Run filters
        self._run_filters(sim, "linear_motion")
    
    def test_circular_motion(self):
        """Test filters on circular motion."""
        print("\n" + "="*70)
        print("TEST 2: Circular Motion")
        print("="*70)
        
        # Generate trajectory
        sim = SimulatedDynamics(dt=self.dt, duration=50.0)
        sim.generate_trajectory('circular')
        sim.generate_attitude('yaw_only')
        sim.calculate_imu_measurements(add_noise=True)
        
        # Run filters
        self._run_filters(sim, "circular_motion")
    
    def test_helical_motion(self):
        """Test filters on helical motion with rolling."""
        print("\n" + "="*70)
        print("TEST 3: Helical Motion with Rolling")
        print("="*70)
        
        # Generate trajectory
        sim = SimulatedDynamics(dt=self.dt, duration=50.0)
        sim.generate_trajectory('helical')
        sim.generate_attitude('rolling')
        sim.calculate_imu_measurements(add_noise=True)
        
        # Run filters
        self._run_filters(sim, "helical_motion")
    
    def _run_filters(self, sim, test_name):
        """
        Run both EKF and MEKF on simulated data.
        
        Args:
            sim: SimulatedDynamics object
            test_name: Name of test for reporting
        """
        n_steps = sim.n_steps
        
        # ============ Initialize EKF ============
        A = EKF.build_A(self.dt)
        B = np.zeros((9, 3))
        C = EKF.build_C(1000, self.dt)  # m = 1000 kg
        Q = EKF.build_Q(0.1)
        R = EKF.build_R(0.1)
        
        x_k = np.zeros((9, 1))
        P_k = np.eye(9) * 1000
        
        ekf_estimates = np.zeros((9, n_steps))
        ekf_covariances = np.zeros((9, 9, n_steps))
        
        # ============ Initialize MEKF ============
        mekf = MEKF(Quaternion(axis=[1, 0, 0], angle=0),
                    1.0, 0.1, 0.1, 0.1, 0.1, 0.1)
        
        mekf_estimates = []
        mekf_covariances = np.zeros((15, 15, n_steps))
        
        # ============ Main Loop ============
        print(f"\nRunning filters on {n_steps} time steps...")
        
        for i in range(n_steps):
            # EKF measurement vector
            y_t = np.zeros((6, 1))
            y_t[0:3, 0] = sim.position_truth[:, i] + np.random.normal(0, sim.gps_noise_std, 3)
            y_t[3:6, 0] = sim.accel_meas[:, i]
            
            # Run EKF
            x_k, P_k = EKF.EKF_iteration(x_k, P_k, y_t, np.zeros((3, 1)), R, Q, A, B, C, 0.1, 0.1)
            ekf_estimates[:, i] = x_k.flatten()
            ekf_covariances[:, :, i] = P_k
            
            # Run MEKF
            mekf.update(sim.gyro_meas[:, i], sim.accel_meas[:, i], self.dt)
            mekf_estimates.append(mekf.estimate)
            mekf_covariances[:, :, i] = mekf.estimate_covariance
        
        print("✓ Filters executed successfully")
        
        # ============ Compute Errors ============
        print("\nComputing estimation errors...")
        
        # EKF Position Error
        ekf_pos_error = ekf_estimates[0:3, :] - sim.position_truth
        ekf_pos_rmse = np.sqrt(np.mean(np.sum(ekf_pos_error**2, axis=0)))
        ekf_pos_max_error = np.max(np.linalg.norm(ekf_pos_error, axis=0))
        
        # EKF Velocity Error
        ekf_vel_error = ekf_estimates[3:6, :] - sim.velocity_truth
        ekf_vel_rmse = np.sqrt(np.mean(np.sum(ekf_vel_error**2, axis=0)))
        
        # EKF Acceleration Error
        ekf_accel_error = ekf_estimates[6:9, :] - sim.acceleration_truth
        ekf_accel_rmse = np.sqrt(np.mean(np.sum(ekf_accel_error**2, axis=0)))
        
        # MEKF Attitude Error (quaternion distance)
        mekf_attitude_errors = []
        for i, q_est in enumerate(mekf_estimates):
            q_true = sim.attitude_truth[i]
            # Quaternion error: angle between true and estimated
            q_err = q_true.inverse * q_est
            angle_err = 2 * np.arccos(np.clip(abs(q_err.w), -1, 1))  # rad
            mekf_attitude_errors.append(np.degrees(angle_err))  # Convert to degrees
        
        mekf_attitude_rmse = np.sqrt(np.mean(np.array(mekf_attitude_errors)**2))
        mekf_attitude_max_error = np.max(mekf_attitude_errors)
        
        # ============ Print Results ============
        print("\n" + "-"*70)
        print("EKF RESULTS (Position Estimation)")
        print("-"*70)
        print(f"Position RMSE:     {ekf_pos_rmse:.4f} m")
        print(f"Position Max Error: {ekf_pos_max_error:.4f} m")
        print(f"Velocity RMSE:     {ekf_vel_rmse:.4f} m/s")
        print(f"Acceleration RMSE: {ekf_accel_rmse:.4f} m/s²")
        
        print("\n" + "-"*70)
        print("MEKF RESULTS (Attitude Estimation)")
        print("-"*70)
        print(f"Attitude RMSE:     {mekf_attitude_rmse:.4f} degrees")
        print(f"Attitude Max Error: {mekf_attitude_max_error:.4f} degrees")
        
        # Filter effectiveness
        print("\n" + "-"*70)
        print("FILTER EFFECTIVENESS")
        print("-"*70)
        
        # Compare to dead reckoning (no filter)
        dr_pos_error = ekf_estimates[0:3, -1] - sim.position_truth[:, -1]
        dr_pos_error_norm = np.linalg.norm(dr_pos_error)
        print(f"Final position error: {dr_pos_error_norm:.4f} m")
        
        # Store results
        self.results[test_name] = {
            'ekf_pos_rmse': ekf_pos_rmse,
            'ekf_pos_max_error': ekf_pos_max_error,
            'ekf_vel_rmse': ekf_vel_rmse,
            'ekf_accel_rmse': ekf_accel_rmse,
            'mekf_attitude_rmse': mekf_attitude_rmse,
            'mekf_attitude_max_error': mekf_attitude_max_error,
            'sim': sim,
            'ekf_estimates': ekf_estimates,
            'mekf_estimates': mekf_estimates,
        }
        
        # Success criteria
        success = (ekf_pos_rmse < 50.0 and  # Allow 50m error (loose for now)
                   mekf_attitude_rmse < 30.0)  # Allow 30 degrees (loose for now)
        
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"\n{status}")
    
    def print_summary(self):
        """Print summary of all tests."""
        print("\n\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        for test_name, results in self.results.items():
            print(f"\n{test_name.upper().replace('_', ' ')}:")
            print(f"  EKF Position RMSE:    {results['ekf_pos_rmse']:.4f} m")
            print(f"  MEKF Attitude RMSE:   {results['mekf_attitude_rmse']:.4f}°")
    
    def plot_test_results(self, test_name=None, save_dir='test_results'):
        """
        Plot results from a specific test or the first available test.
        
        Args:
            test_name: Name of test to plot (e.g., 'linear_motion')
            save_dir: Directory to save plots
        """
        if not self.results:
            print("No test results to plot!")
            return
        
        # Select which test to plot
        if test_name is None:
            test_name = list(self.results.keys())[0]
        
        if test_name not in self.results:
            print(f"Test '{test_name}' not found. Available tests: {list(self.results.keys())}")
            return
        
        results = self.results[test_name]
        ekf_estimates = results['ekf_estimates']
        
        # Convert to format expected by plot_results
        # plot_results expects: state as list of 3-element vectors, cov as list of 3x3 matrices
        state_list = [ekf_estimates[:3, i] for i in range(ekf_estimates.shape[1])]
        
        print(f"\nPlotting results for: {test_name.upper().replace('_', ' ')}")
        print("="*70)
        
        # Create covariance list (3x3 position covariances extracted from 9x9)
        cov_list = []
        # Note: We don't have the full covariance matrices in results, so use identity
        for i in range(len(state_list)):
            cov_list.append(np.eye(3) * 0.01)  # Small placeholder covariance
        
        try:
            plot_results(state_list, cov_list, save_dir=save_dir, show=True)
        except Exception as e:
            print(f"Error during plotting: {e}")
            print("Plots may not display correctly, but computation was successful.")


def main(plot=False):
    """
    Run all tests.
    
    Args:
        plot: If True, plot results from the linear motion test
    """
    print("\n" + "="*70)
    print("INTEGRATED EKF + MEKF TEST SUITE")
    print("="*70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create test suite
    test_suite = TestEKFMEKF(dt=0.01)
    
    # Run tests
    test_suite.test_linear_motion()
    test_suite.test_circular_motion()
    test_suite.test_helical_motion()
    
    # Print summary
    test_suite.print_summary()
    
    # Optionally plot results
    if plot:
        print("\n" + "="*70)
        print("PLOTTING RESULTS")
        print("="*70)
        test_suite.plot_test_results('linear_motion')
    
    print("\n" + "="*70)
    print("Tests completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main(plot=True)  # Set to True to generate plots

