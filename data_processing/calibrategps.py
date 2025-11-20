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


# Define baselines for each GPS calibration file
BASELINES = {
    'GPS_Cal1.csv': {'lat': 37.4268460, 'long': -122.1730398},
    'GPS_Cal2.csv': {'lat': 37.4262003, 'long': -122.1765639},
    'GPS_Cal3.csv': {'lat': 37.4266911, 'long': -122.1737030},
}


def calibrate_gps(csv_file_path, output_dir='./gps_calibration'):
    """
    Calibrate GPS data by comparing to baselines.
    
    Creates plots with baseline lines and computes error statistics.
    
    Args:
        csv_file_path: Path to the CSV file with GPS data
        output_dir: Directory to save plots (default: field_data/gps_calibration)
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
    
    # Get filename for display and lookup baselines
    filename = Path(csv_file_path).name
    stem = Path(csv_file_path).stem
    
    # Get baseline for this file
    if filename not in BASELINES:
        print(f"Error: No baseline defined for {filename}")
        return None
    
    baseline = BASELINES[filename]
    baseline_lat = baseline['lat']
    baseline_long = baseline['long']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Store error statistics
    error_stats = {}
    
    print(f"\n{'='*60}")
    print(f"Calibrating GPS Data: {filename}")
    print(f"Baseline: Lat={baseline_lat}, Long={baseline_long}")
    print(f"{'='*60}")
    
    # Extract GPS data
    lat_data = pd.to_numeric(df['Lat'], errors='coerce').values
    long_data = pd.to_numeric(df['Long'], errors='coerce').values
    ecef_x_data = pd.to_numeric(df['ECEF_X'], errors='coerce').values
    ecef_y_data = pd.to_numeric(df['ECEF_Y'], errors='coerce').values
    ecef_z_data = pd.to_numeric(df['ECEF_Z'], errors='coerce').values
    
    # Remove NaN values for error calculation
    valid_mask_lat = ~np.isnan(lat_data)
    valid_mask_long = ~np.isnan(long_data)
    valid_mask_ecef = ~np.isnan(ecef_x_data) & ~np.isnan(ecef_y_data) & ~np.isnan(ecef_z_data)
    
    lat_valid = lat_data[valid_mask_lat]
    long_valid = long_data[valid_mask_long]
    ecef_x_valid = ecef_x_data[valid_mask_ecef]
    ecef_y_valid = ecef_y_data[valid_mask_ecef]
    ecef_z_valid = ecef_z_data[valid_mask_ecef]
    
    # Calculate errors for latitude
    if len(lat_valid) > 0:
        lat_errors = lat_valid - baseline_lat
        lat_mean_error = np.mean(lat_errors)
        lat_std_error = np.std(lat_errors)
        lat_max_error = np.max(np.abs(lat_errors))
        lat_rms_error = np.sqrt(np.mean(lat_errors**2))
        
        error_stats['Lat'] = {
            'mean_error': lat_mean_error,
            'std_error': lat_std_error,
            'max_error': lat_max_error,
            'rms_error': lat_rms_error,
            'baseline': baseline_lat,
            'mean_data': np.mean(lat_valid),
            'std_data': np.std(lat_valid),
        }
    else:
        error_stats['Lat'] = {
            'mean_error': np.nan,
            'std_error': np.nan,
            'max_error': np.nan,
            'rms_error': np.nan,
            'baseline': baseline_lat,
            'mean_data': np.nan,
            'std_data': np.nan,
        }
    
    # Calculate errors for longitude
    if len(long_valid) > 0:
        long_errors = long_valid - baseline_long
        long_mean_error = np.mean(long_errors)
        long_std_error = np.std(long_errors)
        long_max_error = np.max(np.abs(long_errors))
        long_rms_error = np.sqrt(np.mean(long_errors**2))
        
        error_stats['Long'] = {
            'mean_error': long_mean_error,
            'std_error': long_std_error,
            'max_error': long_max_error,
            'rms_error': long_rms_error,
            'baseline': baseline_long,
            'mean_data': np.mean(long_valid),
            'std_data': np.std(long_valid),
        }
    else:
        error_stats['Long'] = {
            'mean_error': np.nan,
            'std_error': np.nan,
            'max_error': np.nan,
            'rms_error': np.nan,
            'baseline': baseline_long,
            'mean_data': np.nan,
            'std_data': np.nan,
        }
    
    # Calculate statistics for ECEF coordinates
    if len(ecef_x_valid) > 0:
        error_stats['ECEF_X'] = {
            'mean_data': np.mean(ecef_x_valid),
            'std_data': np.std(ecef_x_valid),
        }
        error_stats['ECEF_Y'] = {
            'mean_data': np.mean(ecef_y_valid),
            'std_data': np.std(ecef_y_valid),
        }
        error_stats['ECEF_Z'] = {
            'mean_data': np.mean(ecef_z_valid),
            'std_data': np.std(ecef_z_valid),
        }
    else:
        error_stats['ECEF_X'] = {
            'mean_data': np.nan,
            'std_data': np.nan,
        }
        error_stats['ECEF_Y'] = {
            'mean_data': np.nan,
            'std_data': np.nan,
        }
        error_stats['ECEF_Z'] = {
            'mean_data': np.nan,
            'std_data': np.nan,
        }
    
    # Create 1x2 plot for Lat and Long
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'GPS Calibration Data\n{filename}', fontsize=14, fontweight='bold')
    
    # Plot Latitude
    axes[0].plot(time, lat_data, linewidth=1.5, label='Measured Data', color='blue')
    axes[0].axhline(y=baseline_lat, color='green', linestyle='--', linewidth=2.5, label=f'Baseline = {baseline_lat:.7f}')
    axes[0].set_xlabel('Time (s)', fontsize=11)
    axes[0].set_ylabel('Latitude (°)', fontsize=11)
    axes[0].set_title('Latitude Calibration', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].ticklabel_format(style='plain', axis='y', useOffset=False)
    
    # Plot Longitude
    axes[1].plot(time, long_data, linewidth=1.5, label='Measured Data', color='orange')
    axes[1].axhline(y=baseline_long, color='green', linestyle='--', linewidth=2.5, label=f'Baseline = {baseline_long:.7f}')
    axes[1].set_xlabel('Time (s)', fontsize=11)
    axes[1].set_ylabel('Longitude (°)', fontsize=11)
    axes[1].set_title('Longitude Calibration', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].ticklabel_format(style='plain', axis='y', useOffset=False)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{stem}_calibration.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    
    plt.close(fig)
    
    # Print error statistics table
    print(f"\n{'='*100}")
    print("GPS CALIBRATION ERROR STATISTICS")
    print(f"{'='*100}\n")
    
    table_data = []
    for sensor in ['Lat', 'Long']:
        stats = error_stats[sensor]
        table_data.append([
            sensor,
            f"{stats['baseline']:.7f}",
            f"{stats['mean_data']:.7f}",
            f"{stats['std_data']:.7f}",
            f"{stats['mean_error']:.7f}",
            f"{stats['std_error']:.7f}",
            f"{stats['rms_error']:.7f}",
            f"{stats['max_error']:.7f}",
        ])
    
    # Add ECEF coordinates
    for sensor in ['ECEF_X', 'ECEF_Y', 'ECEF_Z']:
        stats = error_stats[sensor]
        table_data.append([
            sensor,
            '-',
            f"{stats['mean_data']:.2f}",
            f"{stats['std_data']:.2f}",
            '-',
            '-',
            '-',
            '-',
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
    
    print(tabulate(table_data, headers=headers, tablefmt='grid', floatfmt='.7f'))
    print(f"\n{'='*100}\n")
    
    return error_stats


def calibrate_all_gps(field_data_dir, output_dir='./gps_calibration'):
    """
    Calibrate all GPS calibration CSV files and create a summary table.
    
    Args:
        field_data_dir: Path to field_data directory
        output_dir: Output directory for plots
    """
    print(f"\n{'='*60}")
    print("PROCESSING ALL GPS CALIBRATION FILES")
    print(f"{'='*60}")
    
    # Store results from all files
    all_results = {}
    
    # Process each calibration file
    for csv_filename in ['GPS_Cal1.csv', 'GPS_Cal2.csv', 'GPS_Cal3.csv']:
        csv_path = os.path.join(field_data_dir, csv_filename)
        
        if os.path.exists(csv_path):
            error_stats = calibrate_gps(csv_path, output_dir)
            all_results[csv_filename] = error_stats
        else:
            print(f"Warning: File not found: {csv_path}")
    
    # Create summary table averaging all three calibration files
    if len(all_results) > 0:
        print_gps_summary_table(all_results)


def print_gps_summary_table(all_results):
    """
    Print a summary table averaging statistics from all GPS calibration files.
    
    Args:
        all_results: Dictionary with filename keys and error_stats values
    """
    print(f"\n{'='*120}")
    print("GPS CALIBRATION SUMMARY - AVERAGE ACROSS ALL FILES")
    print(f"{'='*120}\n")
    
    # Calculate averages for Lat and Long
    lat_data_all = []
    long_data_all = []
    lat_error_all = []
    long_error_all = []
    lat_std_all = []
    long_std_all = []
    lat_rms_all = []
    long_rms_all = []
    lat_max_all = []
    long_max_all = []
    
    ecef_x_data_all = []
    ecef_y_data_all = []
    ecef_z_data_all = []
    ecef_x_std_all = []
    ecef_y_std_all = []
    ecef_z_std_all = []
    
    # Collect data from all files
    for filename, error_stats in all_results.items():
        if 'Lat' in error_stats:
            lat_stats = error_stats['Lat']
            lat_data_all.append(lat_stats['mean_data'])
            lat_error_all.append(lat_stats['mean_error'])
            lat_std_all.append(lat_stats['std_error'])
            lat_rms_all.append(lat_stats['rms_error'])
            lat_max_all.append(lat_stats['max_error'])
        
        if 'Long' in error_stats:
            long_stats = error_stats['Long']
            long_data_all.append(long_stats['mean_data'])
            long_error_all.append(long_stats['mean_error'])
            long_std_all.append(long_stats['std_error'])
            long_rms_all.append(long_stats['rms_error'])
            long_max_all.append(long_stats['max_error'])
        
        if 'ECEF_X' in error_stats:
            ecef_x_data_all.append(error_stats['ECEF_X']['mean_data'])
            ecef_x_std_all.append(error_stats['ECEF_X']['std_data'])
        
        if 'ECEF_Y' in error_stats:
            ecef_y_data_all.append(error_stats['ECEF_Y']['mean_data'])
            ecef_y_std_all.append(error_stats['ECEF_Y']['std_data'])
        
        if 'ECEF_Z' in error_stats:
            ecef_z_data_all.append(error_stats['ECEF_Z']['mean_data'])
            ecef_z_std_all.append(error_stats['ECEF_Z']['std_data'])
    
    # Build summary table
    table_data = []
    
    # Latitude average
    if lat_data_all:
        table_data.append([
            'Lat (avg)',
            '-',
            f"{np.mean(lat_data_all):.7f}",
            f"{np.std(lat_data_all):.7f}",
            f"{np.mean(lat_error_all):.7f}",
            f"{np.mean(lat_std_all):.7f}",
            f"{np.mean(lat_rms_all):.7f}",
            f"{np.mean(lat_max_all):.7f}",
        ])
    
    # Longitude average
    if long_data_all:
        table_data.append([
            'Long (avg)',
            '-',
            f"{np.mean(long_data_all):.7f}",
            f"{np.std(long_data_all):.7f}",
            f"{np.mean(long_error_all):.7f}",
            f"{np.mean(long_std_all):.7f}",
            f"{np.mean(long_rms_all):.7f}",
            f"{np.mean(long_max_all):.7f}",
        ])
    
    # ECEF averages
    if ecef_x_data_all:
        table_data.append([
            'ECEF_X (avg)',
            '-',
            f"{np.mean(ecef_x_data_all):.2f}",
            f"{np.mean(ecef_x_std_all):.2f}",
            '-',
            '-',
            '-',
            '-',
        ])
    
    if ecef_y_data_all:
        table_data.append([
            'ECEF_Y (avg)',
            '-',
            f"{np.mean(ecef_y_data_all):.2f}",
            f"{np.mean(ecef_y_std_all):.2f}",
            '-',
            '-',
            '-',
            '-',
        ])
    
    if ecef_z_data_all:
        table_data.append([
            'ECEF_Z (avg)',
            '-',
            f"{np.mean(ecef_z_data_all):.2f}",
            f"{np.mean(ecef_z_std_all):.2f}",
            '-',
            '-',
            '-',
            '-',
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
    
    print(tabulate(table_data, headers=headers, tablefmt='grid', floatfmt='.7f'))
    print(f"\n{'='*120}\n")


if __name__ == "__main__":
    # Get the path to the field_data directory
    current_dir = Path(__file__).parent.parent
    field_data_dir = current_dir / 'field_data'
    output_dir = field_data_dir / 'gps_calibration'
    
    # Calibrate all GPS files
    calibrate_all_gps(str(field_data_dir), str(output_dir))

