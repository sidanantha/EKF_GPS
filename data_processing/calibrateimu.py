import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tabulate import tabulate

# Import helper functions from plotcsvdata
import sys
sys.path.insert(0, str(Path(__file__).parent))
from plotcsvdata import read_csv_data, get_time_array


# Define baselines for each IMU sensor
BASELINES = {
    'ax': 0.0,
    'ay': 0.0,
    'az': -9.81,
    'gx': 0.0,
    'gy': 0.0,
    'gz': 0.0,
}

# IMU column names
IMU_COLUMNS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']


def calibrate_imu(csv_file_path, output_dir='./imu_calibration'):
    """
    Calibrate IMU data by comparing to baselines.
    
    Creates plots with baseline lines and computes error statistics.
    
    Args:
        csv_file_path: Path to the CSV file with IMU data
        output_dir: Directory to save plots (default: field_data/imu_calibration)
    """
    
    # Read CSV data - try with and without skiprows
    try:
        df = read_csv_data(csv_file_path)
        if 'UTC' not in df.columns:
            # Try reading without skiprows
            df = pd.read_csv(csv_file_path)
    except:
        df = pd.read_csv(csv_file_path)
    
    time = get_time_array(df)
    
    # Get filename for display
    filename = Path(csv_file_path).stem
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Store error statistics
    error_stats = {}
    
    print(f"\n{'='*60}")
    print(f"Calibrating IMU Data: {filename}")
    print(f"{'='*60}")
    
    # Process each IMU column
    for col in IMU_COLUMNS:
        baseline = BASELINES[col]
        
        # Convert to numeric, coercing errors to NaN
        data = pd.to_numeric(df[col], errors='coerce').values
        
        # Remove NaN values for error calculation
        valid_mask = ~np.isnan(data)
        data_valid = data[valid_mask]
        
        # Skip if no valid data
        if len(data_valid) == 0:
            print(f"Warning: No valid data for {col}")
            error_stats[col] = {
                'mean_error': np.nan,
                'std_error': np.nan,
                'max_error': np.nan,
                'rms_error': np.nan,
                'baseline': baseline,
                'mean_data': np.nan,
                'std_data': np.nan,
            }
            continue
        
        # Calculate errors using only valid data
        errors = data_valid - baseline
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(np.abs(errors))
        rms_error = np.sqrt(np.mean(errors**2))
        
        error_stats[col] = {
            'mean_error': mean_error,
            'std_error': std_error,
            'max_error': max_error,
            'rms_error': rms_error,
            'baseline': baseline,
            'mean_data': np.mean(data_valid),
            'std_data': np.std(data_valid),
        }
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot raw data (handle both valid and invalid data)
        ax.plot(time, data, linewidth=1.5, label='Measured Data', color='blue')
        
        # Plot baseline as dashed green line
        ax.axhline(y=baseline, color='green', linestyle='--', linewidth=2.5, label=f'Baseline = {baseline}')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(col, fontsize=12)
        ax.set_title(f'{col} Calibration Data\n{filename}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'{col}_calibration.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_path}")
        
        plt.close(fig)
    
    # Print error statistics table
    print(f"\n{'='*100}")
    print("IMU CALIBRATION ERROR STATISTICS")
    print(f"{'='*100}\n")
    
    table_data = []
    for col in IMU_COLUMNS:
        stats = error_stats[col]
        table_data.append([
            col,
            f"{stats['baseline']:.6f}",
            f"{stats['mean_data']:.6f}",
            f"{stats['std_data']:.6f}",
            f"{stats['mean_error']:.6f}",
            f"{stats['std_error']:.6f}",
            f"{stats['rms_error']:.6f}",
            f"{stats['max_error']:.6f}",
        ])
    
    headers = [
        'Sensor',
        'Baseline',
        'Mean Data',
        'Std Data',
        'Mean Error',
        'Std Dev Error',
        'RMS Error',
        'Max |Error|'
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt='grid', floatfmt='.6f'))
    print(f"\n{'='*100}\n")
    
    return error_stats


def calibrate_gyro(csv_file_path, output_dir='./gyro_calibration'):
    """
    Calibrate Gyro data by comparing to baselines (all 0).
    
    Creates plots showing raw data, difference from baseline, and spread (±std deviation).
    
    Args:
        csv_file_path: Path to the CSV file with IMU data
        output_dir: Directory to save plots (default: field_data/gyro_calibration)
    """
    
    # Read CSV data
    try:
        df = read_csv_data(csv_file_path)
        if 'UTC' not in df.columns:
            df = pd.read_csv(csv_file_path)
    except:
        df = pd.read_csv(csv_file_path)
    
    time = get_time_array(df)
    
    # Get filename for display
    filename = Path(csv_file_path).stem
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Gyro columns and baselines
    gyro_columns = ['gx', 'gy', 'gz']
    gyro_baselines = {'gx': 0.0, 'gy': 0.0, 'gz': 0.0}
    
    # Store error statistics
    error_stats = {}
    
    print(f"\n{'='*60}")
    print(f"Calibrating Gyro Data: {filename}")
    print(f"{'='*60}")
    
    # Process each gyro column
    for col in gyro_columns:
        baseline = gyro_baselines[col]
        
        # Convert to numeric, coercing errors to NaN
        data = pd.to_numeric(df[col], errors='coerce').values
        
        # Remove NaN values for error calculation
        valid_mask = ~np.isnan(data)
        data_valid = data[valid_mask]
        
        # Skip if no valid data
        if len(data_valid) == 0:
            print(f"Warning: No valid data for {col}")
            error_stats[col] = {
                'mean_error': np.nan,
                'std_error': np.nan,
                'max_error': np.nan,
                'rms_error': np.nan,
                'baseline': baseline,
                'mean_data': np.nan,
                'std_data': np.nan,
            }
            continue
        
        # Calculate errors using only valid data
        errors = data_valid - baseline
        time_valid = time[valid_mask]  # Filter time array to match valid data
        
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(np.abs(errors))
        rms_error = np.sqrt(np.mean(errors**2))
        
        error_stats[col] = {
            'mean_error': mean_error,
            'std_error': std_error,
            'max_error': max_error,
            'rms_error': rms_error,
            'baseline': baseline,
            'mean_data': np.mean(data_valid),
            'std_data': np.std(data_valid),
        }
        
        # Create combined plot with raw data, baseline, and error spread
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # ===== TOP PLOT: Raw Data with Baseline and Spread Bands =====
        ax1.plot(time, data, linewidth=1.5, label='Measured Data', color='blue', alpha=0.8)
        
        # Plot baseline
        ax1.axhline(y=baseline, color='green', linestyle='--', linewidth=2.5, label=f'Baseline = {baseline}')
        
        # Plot spread (±std deviation bands)
        spread_upper = baseline + std_error
        spread_lower = baseline - std_error
        ax1.fill_between(time, spread_lower, spread_upper, color='green', alpha=0.2, label=f'±1σ Spread (±{std_error:.6f})')
        ax1.plot(time, np.full_like(time, spread_upper, dtype=float), color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        ax1.plot(time, np.full_like(time, spread_lower, dtype=float), color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        
        ax1.set_ylabel(col, fontsize=12, fontweight='bold')
        ax1.set_title(f'{col} Raw Data with Baseline and Spread', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10)
        
        # ===== BOTTOM PLOT: Difference from Baseline =====
        ax2.plot(time_valid, errors, linewidth=1.5, label='Error (Data - Baseline)', color='red', alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, label='Zero Error')
        
        # Plot spread bands around zero
        ax2.fill_between(time_valid, -std_error, std_error, color='red', alpha=0.2, label=f'±1σ Spread')
        ax2.plot(time_valid, np.full_like(time_valid, std_error, dtype=float), color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        ax2.plot(time_valid, np.full_like(time_valid, -std_error, dtype=float), color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        
        # Add 3σ bounds as dashed lines
        ax2.plot(time_valid, np.full_like(time_valid, 3*std_error, dtype=float), color='orange', linestyle='--', linewidth=1, alpha=0.5, label=f'±3σ')
        ax2.plot(time_valid, np.full_like(time_valid, -3*std_error, dtype=float), color='orange', linestyle='--', linewidth=1, alpha=0.5)
        
        ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax2.set_ylabel(f'{col} Error (rad/s)', fontsize=12, fontweight='bold')
        ax2.set_title(f'{col} Error from Baseline with Spread', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'{col}_calibration.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        
        plt.close(fig)
    
    # Print error statistics table
    print(f"\n{'='*100}")
    print("GYRO CALIBRATION ERROR STATISTICS")
    print(f"{'='*100}\n")
    
    table_data = []
    for col in gyro_columns:
        stats = error_stats[col]
        table_data.append([
            col,
            f"{stats['baseline']:.6f}",
            f"{stats['mean_data']:.6f}",
            f"{stats['std_data']:.6f}",
            f"{stats['mean_error']:.6f}",
            f"{stats['std_error']:.6f}",
            f"{stats['rms_error']:.6f}",
            f"{stats['max_error']:.6f}",
        ])
    
    headers = [
        'Sensor',
        'Baseline',
        'Mean Data',
        'Std Data',
        'Mean Error',
        'Std Dev Error',
        'RMS Error',
        'Max |Error|'
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt='grid', floatfmt='.6f'))
    print(f"\n{'='*100}\n")
    
    return error_stats


if __name__ == "__main__":
    # Get the path to the field_data directory
    current_dir = Path(__file__).parent.parent
    field_data_dir = current_dir / 'field_data'
    csv_file = field_data_dir / 'IMU_calibration_Loc1_run.csv'
    
    imu_output_dir = field_data_dir / 'imu_calibration'
    gyro_output_dir = field_data_dir / 'gyro_calibration'
    
    if csv_file.exists():
        print("Running IMU calibration...")
        calibrate_imu(str(csv_file), str(imu_output_dir))
        
        print("\nRunning Gyro calibration...")
        calibrate_gyro(str(csv_file), str(gyro_output_dir))
    else:
        print(f"Error: File not found: {csv_file}")

