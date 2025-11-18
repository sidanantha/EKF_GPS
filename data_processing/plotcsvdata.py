import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import argparse


def read_csv_data(file_path):
    """
    Read CSV data from the specified file.
    Skip the first row if it contains metadata.
    """
    df = pd.read_csv(file_path, skiprows=1)
    return df


def convert_time_to_seconds(time_str):
    """
    Convert time string in HH:MM:SS.mmm format to seconds.
    Returns None if the time string is 'NA'.
    """
    if pd.isna(time_str) or time_str == 'NA':
        return None
    
    try:
        parts = str(time_str).split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
    except:
        return None


def get_time_array(df):
    """
    Convert UTC column to time in seconds, handling NA values.
    """
    # Pre-allocate as float array to avoid object dtype issues
    time_seconds = np.full(len(df), np.nan, dtype=float)
    
    for i, time_val in enumerate(df['UTC']):
        sec = convert_time_to_seconds(time_val)
        if sec is not None:
            time_seconds[i] = sec
    
    # Find first valid time and use it as reference
    valid_mask = ~np.isnan(time_seconds)
    if not valid_mask.any():
        return np.arange(len(time_seconds))
    
    first_valid_idx = np.where(valid_mask)[0][0]
    first_valid_time = time_seconds[first_valid_idx]
    
    # Set all invalid times to the first valid time (or interpolate)
    time_seconds[~valid_mask] = first_valid_time
    
    # Make time relative to first valid time
    time_seconds = time_seconds - first_valid_time
    
    return time_seconds


def plot_gps_and_imu_data(csv_file_path, output_dir='./plots'):
    """
    Create two plots:
    Plot 1: 3x2 layout with GPS data (Lat, Long, Alt) and ECEF data
    Plot 2: 3x3 layout with accelerometer, gyro, and magnetometer data
    """
    
    # Read CSV data
    df = read_csv_data(csv_file_path)
    
    # Get time array
    time = get_time_array(df)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename for labeling
    filename = Path(csv_file_path).stem
    
    # ===== PLOT 1: GPS and ECEF Data (3x2) =====
    fig1, axes1 = plt.subplots(3, 2, figsize=(12, 10))
    fig1.suptitle(f'GPS and ECEF Data vs Time\n{filename}', fontsize=14, fontweight='bold')
    
    # Left column: GPS data (Lat, Long, Alt)
    axes1[0, 0].plot(time, df['Lat'], linewidth=1.5)
    axes1[0, 0].set_ylabel('Latitude (°)')
    axes1[0, 0].grid(True, alpha=0.3)
    axes1[0, 0].set_title('Latitude')
    
    axes1[1, 0].plot(time, df['Long'], linewidth=1.5, color='orange')
    axes1[1, 0].set_ylabel('Longitude (°)')
    axes1[1, 0].grid(True, alpha=0.3)
    axes1[1, 0].set_title('Longitude')
    
    axes1[2, 0].plot(time, df['Alt'], linewidth=1.5, color='green')
    axes1[2, 0].set_ylabel('Altitude (m)')
    axes1[2, 0].set_xlabel('Time (s)')
    axes1[2, 0].grid(True, alpha=0.3)
    axes1[2, 0].set_title('Altitude')
    
    # Right column: ECEF data (ECEF_X, ECEF_Y, ECEF_Z)
    axes1[0, 1].plot(time, df['ECEF_X'], linewidth=1.5, color='red')
    axes1[0, 1].set_ylabel('ECEF_X (m)')
    axes1[0, 1].grid(True, alpha=0.3)
    axes1[0, 1].set_title('ECEF X')
    
    axes1[1, 1].plot(time, df['ECEF_Y'], linewidth=1.5, color='purple')
    axes1[1, 1].set_ylabel('ECEF_Y (m)')
    axes1[1, 1].grid(True, alpha=0.3)
    axes1[1, 1].set_title('ECEF Y')
    
    axes1[2, 1].plot(time, df['ECEF_Z'], linewidth=1.5, color='brown')
    axes1[2, 1].set_ylabel('ECEF_Z (m)')
    axes1[2, 1].set_xlabel('Time (s)')
    axes1[2, 1].grid(True, alpha=0.3)
    axes1[2, 1].set_title('ECEF Z')
    
    plt.tight_layout()
    
    # Create subdirectory for this file's plots
    file_output_dir = os.path.join(output_dir, filename)
    os.makedirs(file_output_dir, exist_ok=True)
    
    plot1_path = os.path.join(file_output_dir, 'gps.png')
    fig1.savefig(plot1_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot1_path}")
    
    # ===== PLOT 2: Accelerometer, Gyro, Magnetometer Data (3x3) =====
    fig2, axes2 = plt.subplots(3, 3, figsize=(14, 10))
    fig2.suptitle(f'IMU Sensor Data vs Time\n{filename}', fontsize=14, fontweight='bold')
    
    # Column 1: Accelerometer (ax, ay, az)
    axes2[0, 0].plot(time, df['ax'], linewidth=1.5)
    axes2[0, 0].set_ylabel('ax (m/s²)')
    axes2[0, 0].grid(True, alpha=0.3)
    axes2[0, 0].set_title('Accelerometer X')
    
    axes2[1, 0].plot(time, df['ay'], linewidth=1.5, color='orange')
    axes2[1, 0].set_ylabel('ay (m/s²)')
    axes2[1, 0].grid(True, alpha=0.3)
    axes2[1, 0].set_title('Accelerometer Y')
    
    axes2[2, 0].plot(time, df['az'], linewidth=1.5, color='green')
    axes2[2, 0].set_ylabel('az (m/s²)')
    axes2[2, 0].set_xlabel('Time (s)')
    axes2[2, 0].grid(True, alpha=0.3)
    axes2[2, 0].set_title('Accelerometer Z')
    
    # Column 2: Gyroscope (gx, gy, gz)
    axes2[0, 1].plot(time, df['gx'], linewidth=1.5, color='red')
    axes2[0, 1].set_ylabel('gx (rad/s)')
    axes2[0, 1].grid(True, alpha=0.3)
    axes2[0, 1].set_title('Gyroscope X')
    
    axes2[1, 1].plot(time, df['gy'], linewidth=1.5, color='purple')
    axes2[1, 1].set_ylabel('gy (rad/s)')
    axes2[1, 1].grid(True, alpha=0.3)
    axes2[1, 1].set_title('Gyroscope Y')
    
    axes2[2, 1].plot(time, df['gz'], linewidth=1.5, color='brown')
    axes2[2, 1].set_ylabel('gz (rad/s)')
    axes2[2, 1].set_xlabel('Time (s)')
    axes2[2, 1].grid(True, alpha=0.3)
    axes2[2, 1].set_title('Gyroscope Z')
    
    # Column 3: Magnetometer (magx, magy, magz)
    axes2[0, 2].plot(time, df['magx'], linewidth=1.5, color='cyan')
    axes2[0, 2].set_ylabel('magx (µT)')
    axes2[0, 2].grid(True, alpha=0.3)
    axes2[0, 2].set_title('Magnetometer X')
    
    axes2[1, 2].plot(time, df['magy'], linewidth=1.5, color='magenta')
    axes2[1, 2].set_ylabel('magy (µT)')
    axes2[1, 2].grid(True, alpha=0.3)
    axes2[1, 2].set_title('Magnetometer Y')
    
    axes2[2, 2].plot(time, df['magz'], linewidth=1.5, color='lime')
    axes2[2, 2].set_ylabel('magz (µT)')
    axes2[2, 2].set_xlabel('Time (s)')
    axes2[2, 2].grid(True, alpha=0.3)
    axes2[2, 2].set_title('Magnetometer Z')
    
    plt.tight_layout()
    
    plot2_path = os.path.join(file_output_dir, 'imu.png')
    fig2.savefig(plot2_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot2_path}")
    
    plt.close(fig1)
    plt.close(fig2)


def plot_all_csv_files(field_data_dir, output_dir='./plots'):
    """
    Process all CSV files in the field_data directory.
    """
    csv_files = list(Path(field_data_dir).glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {field_data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s)")
    
    for csv_file in sorted(csv_files):
        print(f"\nProcessing: {csv_file.name}")
        try:
            plot_gps_and_imu_data(str(csv_file), output_dir)
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")


def generate_all_plots(field_data_dir=None):
    """
    Generate all plots for all CSV files in the field_data directory.
    This is a convenient wrapper function to generate all plots at once.
    
    Args:
        field_data_dir: Path to field_data directory. If None, uses default relative path.
    """
    if field_data_dir is None:
        current_dir = Path(__file__).parent.parent
        field_data_dir = current_dir / 'field_data'
    
    output_dir = Path(field_data_dir) / 'plots'
    
    print("=" * 60)
    print("GENERATING ALL PLOTS FOR FIELD DATA")
    print("=" * 60)
    plot_all_csv_files(str(field_data_dir), str(output_dir))
    print("=" * 60)
    print(f"All plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot GPS and IMU data from CSV files in the field_data directory.'
    )
    parser.add_argument(
        'file',
        nargs='?',
        default=None,
        help='Specific CSV file to process (absolute or relative path). If not provided, processes single file or use --all for all files.'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate plots for ALL CSV files in field_data/'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output directory for plots (default: field_data/plots/)'
    )
    
    args = parser.parse_args()
    
    # Get the path to the field_data directory
    current_dir = Path(__file__).parent.parent
    field_data_dir = current_dir / 'field_data'
    output_dir = Path(args.output) if args.output else field_data_dir / 'plots'
    
    if args.all:
        # Generate all plots
        generate_all_plots(str(field_data_dir))
    elif args.file:
        # Process specific file
        file_path = Path(args.file)
        
        # If relative path, resolve relative to current_dir or field_data_dir
        if not file_path.is_absolute():
            if not file_path.exists():
                # Try in field_data directory
                alt_path = field_data_dir / file_path.name
                if alt_path.exists():
                    file_path = alt_path
        
        if file_path.exists():
            print(f"Processing: {file_path.name}")
            try:
                plot_gps_and_imu_data(str(file_path), str(output_dir))
                print(f"Plots saved to: {output_dir}")
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
        else:
            print(f"Error: File not found: {file_path}")
    else:
        # Default: show help if no arguments provided
        parser.print_help()

