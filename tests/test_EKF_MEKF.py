"""
Simulated Data Generator for EKF + MEKF System
===============================================

Generates simulated ground truth trajectories and corresponding IMU measurements.
Creates CSV files for input to main.py and saves ground truth data for later
comparison with filter estimates.

Workflow:
  1. This script generates CSV files (simulated_data_*.csv)
  2. Run main.py to process CSV files and generate estimates
  3. Run tests/compare_results.py to compare estimates with ground truth
"""

import sys
import os
import numpy as np
from pyquaternion import Quaternion

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


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
        self.gps_meas = None            # (3, n_steps) - position with GPS noise
        
        # Noise standard deviations
        self.accel_noise_std = 0.05     # m/s^2
        self.gyro_noise_std = 0.01      # rad/s
        self.gps_noise_std = 1.0        # m (1 meter GPS noise)
        
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
    
    def save_to_csv(self, filename, test_name='linear_motion'):
        """
        Save simulated data to CSV file for use with main.py.
        
        Args:
            filename: Output CSV filename
            test_name: Name of test to extract data from
        """
        import csv
        import os
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header row matching data_processing.data_processing expectations
            writer.writerow(['UTC', 'Lat', 'Lon', 'Alt', 'ECEF_X', 'ECEF_Y', 'ECEF_Z', 
                           'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
            
            # Write data rows
            for i in range(self.n_steps):
                utc_time = self.t[i]
                # For ECEF coordinates, use the simulated GPS positions (noisy)
                ecef_x, ecef_y, ecef_z = self.gps_meas[:, i]
                # Use placeholder lat/lon (not used in tests)
                lat, lon, alt = 0, 0, 0
                # Accelerometer measurements
                ax, ay, az = self.accel_meas[:, i]
                # Gyroscope measurements
                gx, gy, gz = self.gyro_meas[:, i]
                
                writer.writerow([utc_time, lat, lon, alt, ecef_x, ecef_y, ecef_z,
                               ax, ay, az, gx, gy, gz])
        
        print(f"✓ Saved simulated data to: {filename}")
    
    def calculate_imu_measurements(self, add_noise=True):
        """
        Calculate what IMU sensors should measure given true dynamics.
        
        IMU measures:
        - Accelerometer: Non-gravitational acceleration in body frame + gravity
        - Gyroscope: Angular velocity in body frame
        - GPS: Position measurements with noise
        """
        self.accel_meas = np.zeros((3, self.n_steps))
        self.gyro_meas = np.zeros((3, self.n_steps))
        self.gps_meas = np.zeros((3, self.n_steps))
        
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
            
            # GPS measures position in inertial frame
            self.gps_meas[:, i] = self.position_truth[:, i]
            
            self.accel_meas[:, i] = a_body
            self.gyro_meas[:, i] = omega_body
            
            # Add noise
            if add_noise:
                self.accel_meas[:, i] += np.random.normal(0, self.accel_noise_std, 3)
                self.gyro_meas[:, i] += np.random.normal(0, self.gyro_noise_std, 3)
                self.gps_meas[:, i] += np.random.normal(0, self.gps_noise_std, 3)


class TestEKFMEKF:
    """Simulated data generator for EKF + MEKF system."""
    
    def __init__(self, dt=0.01):
        self.dt = dt
        self.simulations = {}
    
    def generate_linear_motion(self):
        """Generate linear motion simulated data."""
        print("\n" + "="*70)
        print("GENERATING: Linear Motion")
        print("="*70)
        
        # Generate trajectory
        sim = SimulatedDynamics(dt=self.dt, duration=10.0)
        sim.generate_trajectory('linear')
        sim.generate_attitude('static')
        sim.calculate_imu_measurements(add_noise=True)
        
        self.simulations['linear_motion'] = sim
        return sim
    
    def generate_circular_motion(self):
        """Generate circular motion simulated data."""
        print("\n" + "="*70)
        print("GENERATING: Circular Motion")
        print("="*70)
        
        # Generate trajectory
        sim = SimulatedDynamics(dt=self.dt, duration=50.0)
        sim.generate_trajectory('circular')
        sim.generate_attitude('yaw_only')
        sim.calculate_imu_measurements(add_noise=True)
        
        self.simulations['circular_motion'] = sim
        return sim
    
    def generate_helical_motion(self):
        """Generate helical motion simulated data."""
        print("\n" + "="*70)
        print("GENERATING: Helical Motion with Rolling")
        print("="*70)
        
        # Generate trajectory
        sim = SimulatedDynamics(dt=self.dt, duration=50.0)
        sim.generate_trajectory('helical')
        sim.generate_attitude('rolling')
        sim.calculate_imu_measurements(add_noise=True)
        
        self.simulations['helical_motion'] = sim
        return sim
    
    def save_ground_truth(self, test_name, output_dir='simulated_ground_truth'):
        """
        Save ground truth data to numpy files for later comparison with estimates.
        
        Args:
            test_name: Name of test (e.g., 'linear_motion')
            output_dir: Directory to save ground truth files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if test_name not in self.simulations:
            print(f"Test '{test_name}' not found in simulations!")
            return
        
        sim = self.simulations[test_name]
        
        # Save ground truth data
        np.save(f'{output_dir}/{test_name}_position_truth.npy', sim.position_truth)
        np.save(f'{output_dir}/{test_name}_velocity_truth.npy', sim.velocity_truth)
        np.save(f'{output_dir}/{test_name}_acceleration_truth.npy', sim.acceleration_truth)
        np.save(f'{output_dir}/{test_name}_angular_velocity_truth.npy', sim.angular_velocity_truth)
        
        # Save attitude truth as quaternion list (convert to numpy for saving)
        quat_array = np.array([[q.w, q.x, q.y, q.z] for q in sim.attitude_truth])
        np.save(f'{output_dir}/{test_name}_attitude_truth.npy', quat_array)
        
        # Save time vector
        np.save(f'{output_dir}/{test_name}_time.npy', sim.t)
        
        print(f"✓ Saved ground truth data for {test_name} to {output_dir}/")


def main():
    """
    Generate simulated data for all test scenarios and save to CSV files.
    Ground truth data is also saved for later comparison.
    Then runs main.py and compare_results.py automatically.
    """
    print("\n" + "="*70)
    print("SIMULATED DATA GENERATION")
    print("="*70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create data generator
    data_gen = TestEKFMEKF(dt=0.01)
    
    # Generate data for each scenario
    print("\nGenerating simulated data for three scenarios:")
    data_gen.generate_linear_motion()
    data_gen.generate_circular_motion()
    data_gen.generate_helical_motion()
    
    # Save CSV files for use with main.py
    print("\n" + "="*70)
    print("SAVING SIMULATED DATA TO CSV FILES")
    print("="*70)
    
    for test_name in ['linear_motion', 'circular_motion', 'helical_motion']:
        sim = data_gen.simulations[test_name]
        csv_filename = f'simulated_data_{test_name}.csv'
        sim.save_to_csv(csv_filename, test_name)
    
    # Save ground truth data for later comparison
    print("\n" + "="*70)
    print("SAVING GROUND TRUTH DATA")
    print("="*70)
    
    for test_name in ['linear_motion', 'circular_motion', 'helical_motion']:
        data_gen.save_ground_truth(test_name, output_dir='simulated_ground_truth')
    
    print("\n" + "="*70)
    print("DATA GENERATION COMPLETED!")
    print("="*70)
    
    # ============ STEP 1: RUN FILTERS ON CSV DATA ============
    print("\n\n" + "="*70)
    print("STEP 1: RUNNING EKF/MEKF FILTERS ON SIMULATED DATA")
    print("="*70)
    
    import subprocess
    
    for test_name in ['linear_motion', 'circular_motion', 'helical_motion']:
        csv_filename = f'simulated_data_{test_name}.csv'
        print(f"\nProcessing {test_name}...")
        result = subprocess.run(['python', 'main.py', csv_filename], 
                              capture_output=False, text=True)
        if result.returncode != 0:
            print(f"✗ Error processing {test_name}")
        else:
            print(f"✓ Completed {test_name}")
    
    # ============ STEP 2: COMPARE ESTIMATES WITH GROUND TRUTH ============
    print("\n\n" + "="*70)
    print("STEP 2: COMPARING ESTIMATES WITH GROUND TRUTH")
    print("="*70)
    
    for test_name in ['linear_motion', 'circular_motion', 'helical_motion']:
        print(f"\nGenerating comparison plots for {test_name}...")
        result = subprocess.run(['python', 'tests/compare_results.py', test_name], 
                              capture_output=False, text=True)
        if result.returncode != 0:
            print(f"✗ Error comparing {test_name}")
        else:
            print(f"✓ Completed comparison for {test_name}")
    
    print("\n" + "="*70)
    print("ALL STEPS COMPLETED!")
    print("="*70)
    print("\nGenerated files:")
    print("  - CSV data: simulated_data_*.csv")
    print("  - Ground truth: simulated_ground_truth/")
    print("  - Estimates: results/")
    print("  - Plots: test_results/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

