import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def read_csv_data(file_path):
    """
    Read CSV data from the specified file.
    Skip the first row if it contains metadata (PuTTY log header).
    """
    # Try reading with skiprows=1 first (for files with PuTTY log header)
    df = pd.read_csv(file_path, skiprows=1)
    
    # If UTC column is missing, try reading without skipping rows
    if 'UTC' not in df.columns:
        df = pd.read_csv(file_path)
    
    # If still no UTC column, raise an error
    if 'UTC' not in df.columns:
        raise ValueError(f"UTC column not found in {file_path}. Available columns: {df.columns.tolist()}")
    
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


def calculate_moving_average(data, window_size=10):
    """
    Calculate moving average of data over specified window size.
    Uses convolution with 'same' mode to maintain same length.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def plot_individual_columns_plotly(csv_file_path, output_dir='./plots', window_size=1, show_moving_avg=False):
    """
    Create interactive Plotly plots for each CSV column and save as HTML files.
    """
    # Read CSV data
    df = read_csv_data(csv_file_path)
    
    # Get time array
    time = get_time_array(df)
    
    # Get filename for organizing plots
    filename = Path(csv_file_path).stem
    file_output_dir = os.path.join(output_dir, filename)
    os.makedirs(file_output_dir, exist_ok=True)
    
    # Columns to plot (excluding UTC)
    columns_to_plot = [col for col in df.columns if col != 'UTC']
    
    for col in columns_to_plot:
        try:
            fig = go.Figure()
            
            # Add data trace
            fig.add_trace(go.Scatter(
                x=time,
                y=df[col],
                mode='lines',
                name='Data',
                line=dict(color='blue', width=1.5)
            ))
            
            # Add moving average line if enabled
            if show_moving_avg:
                moving_avg = calculate_moving_average(df[col].values, window_size=window_size)
                fig.add_trace(go.Scatter(
                    x=time,
                    y=moving_avg,
                    mode='lines',
                    name=f'Moving Avg ({window_size} samples)',
                    line=dict(color='red', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title=f'{col} vs Time<br>{filename}',
                xaxis_title='Time (s)',
                yaxis_title=col,
                hovermode='x unified',
                template='plotly_white',
                height=600,
                width=1000,
                yaxis=dict(tickformat='.6f')
            )
            
            plot_path = os.path.join(file_output_dir, f'{col}_interactive.html')
            fig.write_html(plot_path)
            print(f"Saved: {plot_path}")
            
        except Exception as e:
            print(f"Error plotting {col}: {e}")


def plot_individual_columns(csv_file_path, output_dir='./plots', window_size=1, show_moving_avg=False):
    """
    Create individual plots for each CSV column and save as PNG files.
    """
    # Read CSV data
    df = read_csv_data(csv_file_path)
    
    # Get time array
    time = get_time_array(df)
    
    # Get filename for organizing plots
    filename = Path(csv_file_path).stem
    file_output_dir = os.path.join(output_dir, filename)
    os.makedirs(file_output_dir, exist_ok=True)
    
    # Columns to plot (excluding UTC)
    columns_to_plot = [col for col in df.columns if col != 'UTC']
    
    for col in columns_to_plot:
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(time, df[col], linewidth=1.5, label='Data')
            
            # Add moving average line if enabled
            if show_moving_avg:
                moving_avg = calculate_moving_average(df[col].values, window_size=window_size)
                ax.plot(time, moving_avg, linewidth=2, color='red', linestyle='--', label=f'Moving Avg ({window_size} samples)')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(col)
            ax.set_title(f'{col} vs Time\n{filename}')
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style='plain', axis='y', useOffset=False)
            if show_moving_avg:
                ax.legend(loc='best')
            
            plt.tight_layout()
            
            plot_path = os.path.join(file_output_dir, f'{col}.png')
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {plot_path}")
            
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting {col}: {e}")


def plot_relative_ecef_data(csv_file_path, output_dir='./plots'):
    """
    Create plots for relative ECEF position (displacement from starting point).
    Plots relative X, Y, Z in a 1x3 layout.
    """
    # Read CSV data
    df = read_csv_data(csv_file_path)
    time = get_time_array(df)
    filename = Path(csv_file_path).stem
    file_output_dir = os.path.join(output_dir, filename)
    os.makedirs(file_output_dir, exist_ok=True)
    
    # Convert ECEF columns to numeric, handling NaN values
    ecef_x = pd.to_numeric(df['ECEF_X'], errors='coerce')
    ecef_y = pd.to_numeric(df['ECEF_Y'], errors='coerce')
    ecef_z = pd.to_numeric(df['ECEF_Z'], errors='coerce')
    
    # Get reference point (first valid values)
    valid_mask_x = ~np.isnan(ecef_x)
    if not valid_mask_x.any():
        print(f"Warning: No valid ECEF_X data in {filename}")
        return
    
    ref_x = ecef_x[valid_mask_x].iloc[0]
    ref_y = ecef_y[~np.isnan(ecef_y)].iloc[0] if (~np.isnan(ecef_y)).any() else 0
    ref_z = ecef_z[~np.isnan(ecef_z)].iloc[0] if (~np.isnan(ecef_z)).any() else 0
    
    # Calculate relative positions
    rel_x = ecef_x - ref_x
    rel_y = ecef_y - ref_y
    rel_z = ecef_z - ref_z
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Relative ECEF Position vs Time\n{filename}', fontsize=14, fontweight='bold')
    
    # Plot relative X
    axes[0].plot(time, rel_x, linewidth=1.5, color='red')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('ΔX (m)')
    axes[0].set_title('Relative X Position')
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(style='plain', axis='y', useOffset=False)
    
    # Plot relative Y
    axes[1].plot(time, rel_y, linewidth=1.5, color='green')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('ΔY (m)')
    axes[1].set_title('Relative Y Position')
    axes[1].grid(True, alpha=0.3)
    axes[1].ticklabel_format(style='plain', axis='y', useOffset=False)
    
    # Plot relative Z
    axes[2].plot(time, rel_z, linewidth=1.5, color='blue')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('ΔZ (m)')
    axes[2].set_title('Relative Z Position')
    axes[2].grid(True, alpha=0.3)
    axes[2].ticklabel_format(style='plain', axis='y', useOffset=False)
    
    plt.tight_layout()
    
    plot_path = os.path.join(file_output_dir, 'relative_ecef.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    
    plt.close(fig)
    
    



def plot_relative_ecef_data_plotly(csv_file_path, output_dir='./plots'):
    """
    Create interactive Plotly plots for relative ECEF position.
    """
    # Read CSV data
    df = read_csv_data(csv_file_path)
    time = get_time_array(df)
    filename = Path(csv_file_path).stem
    file_output_dir = os.path.join(output_dir, filename)
    os.makedirs(file_output_dir, exist_ok=True)
    
    # Convert ECEF columns to numeric, handling NaN values
    ecef_x = pd.to_numeric(df['ECEF_X'], errors='coerce')
    ecef_y = pd.to_numeric(df['ECEF_Y'], errors='coerce')
    ecef_z = pd.to_numeric(df['ECEF_Z'], errors='coerce')
    
    # Get reference point (first valid values)
    valid_mask_x = ~np.isnan(ecef_x)
    if not valid_mask_x.any():
        print(f"Warning: No valid ECEF_X data in {filename}")
        return
    
    ref_x = ecef_x[valid_mask_x].iloc[0]
    ref_y = ecef_y[~np.isnan(ecef_y)].iloc[0] if (~np.isnan(ecef_y)).any() else 0
    ref_z = ecef_z[~np.isnan(ecef_z)].iloc[0] if (~np.isnan(ecef_z)).any() else 0
    
    # Calculate relative positions
    rel_x = ecef_x - ref_x
    rel_y = ecef_y - ref_y
    rel_z = ecef_z - ref_z
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Relative X Position', 'Relative Y Position', 'Relative Z Position'),
        specs=[[{}, {}, {}]]
    )
    
    fig.add_trace(go.Scatter(x=time, y=rel_x, name='ΔX', line=dict(color='red', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=rel_y, name='ΔY', line=dict(color='green', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=time, y=rel_z, name='ΔZ', line=dict(color='blue', width=2)), row=1, col=3)
    
    fig.update_yaxes(title_text='ΔX (m)', row=1, col=1, tickformat='.1f')
    fig.update_yaxes(title_text='ΔY (m)', row=1, col=2, tickformat='.1f')
    fig.update_yaxes(title_text='ΔZ (m)', row=1, col=3, tickformat='.1f')
    fig.update_xaxes(title_text='Time (s)', row=1, col=1)
    fig.update_xaxes(title_text='Time (s)', row=1, col=2)
    fig.update_xaxes(title_text='Time (s)', row=1, col=3)
    
    fig.update_layout(
        title_text=f'Relative ECEF Position vs Time<br>{filename}',
        height=600,
        width=1400,
        showlegend=True,
        hovermode='x unified'
    )
    
    plot_path = os.path.join(file_output_dir, 'relative_ecef_interactive.html')
    fig.write_html(plot_path)
    print(f"Saved: {plot_path}")


def plot_relative_enu_data(csv_file_path, output_dir='./plots'):
    """
    Create plots for relative ENU position (displacement from starting point in ENU frame).
    Plots relative E, N, U in a 1x3 layout.
    Transforms from ECEF to ENU using reference lat/long from first data point.
    """
    # Read CSV data
    df = read_csv_data(csv_file_path)
    time = get_time_array(df)
    filename = Path(csv_file_path).stem
    file_output_dir = os.path.join(output_dir, filename)
    os.makedirs(file_output_dir, exist_ok=True)
    
    # Convert columns to numeric, handling NaN values
    ecef_x = pd.to_numeric(df['ECEF_X'], errors='coerce')
    ecef_y = pd.to_numeric(df['ECEF_Y'], errors='coerce')
    ecef_z = pd.to_numeric(df['ECEF_Z'], errors='coerce')
    lat = pd.to_numeric(df['Lat'], errors='coerce')
    lon = pd.to_numeric(df['Long'], errors='coerce')
    
    # Get reference point (first valid values)
    valid_mask_x = ~np.isnan(ecef_x)
    if not valid_mask_x.any():
        print(f"Warning: No valid ECEF_X data in {filename}")
        return
    
    ref_x = ecef_x[valid_mask_x].iloc[0]
    ref_y = ecef_y[~np.isnan(ecef_y)].iloc[0] if (~np.isnan(ecef_y)).any() else 0
    ref_z = ecef_z[~np.isnan(ecef_z)].iloc[0] if (~np.isnan(ecef_z)).any() else 0
    
    # Get reference lat/long
    ref_lat = lat[~np.isnan(lat)].iloc[0] if (~np.isnan(lat)).any() else 0
    ref_lon = lon[~np.isnan(lon)].iloc[0] if (~np.isnan(lon)).any() else 0
    
    # Convert to radians
    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)
    
    # Create DCM from ECEF to ENU using the transformation matrix
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    dcm_ecef_to_enu = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])
    
    # Calculate relative ECEF positions
    rel_ecef = np.array([
        ecef_x.values - ref_x,
        ecef_y.values - ref_y,
        ecef_z.values - ref_z
    ])
    
    # Transform to ENU (apply DCM to each position vector)
    rel_enu = dcm_ecef_to_enu @ rel_ecef
    
    rel_e = rel_enu[0]
    rel_n = rel_enu[1]
    rel_u = rel_enu[2]
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Relative ENU Position vs Time\n{filename}', fontsize=14, fontweight='bold')
    
    # Plot relative East
    axes[0].plot(time, rel_e, linewidth=1.5, color='red')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('ΔE (m)')
    axes[0].set_title('Relative East Position')
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(style='plain', axis='y', useOffset=False)
    
    # Plot relative North
    axes[1].plot(time, rel_n, linewidth=1.5, color='green')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('ΔN (m)')
    axes[1].set_title('Relative North Position')
    axes[1].grid(True, alpha=0.3)
    axes[1].ticklabel_format(style='plain', axis='y', useOffset=False)
    
    # Plot relative Up
    axes[2].plot(time, rel_u, linewidth=1.5, color='blue')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('ΔU (m)')
    axes[2].set_title('Relative Up Position')
    axes[2].grid(True, alpha=0.3)
    axes[2].ticklabel_format(style='plain', axis='y', useOffset=False)
    
    plt.tight_layout()
    
    plot_path = os.path.join(file_output_dir, 'relative_enu.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    
    plt.close(fig)


def plot_relative_enu_data_plotly(csv_file_path, output_dir='./plots'):
    """
    Create interactive Plotly plots for relative ENU position.
    Transforms from ECEF to ENU using reference lat/long from first data point.
    """
    # Read CSV data
    df = read_csv_data(csv_file_path)
    time = get_time_array(df)
    filename = Path(csv_file_path).stem
    file_output_dir = os.path.join(output_dir, filename)
    os.makedirs(file_output_dir, exist_ok=True)
    
    # Convert columns to numeric, handling NaN values
    ecef_x = pd.to_numeric(df['ECEF_X'], errors='coerce')
    ecef_y = pd.to_numeric(df['ECEF_Y'], errors='coerce')
    ecef_z = pd.to_numeric(df['ECEF_Z'], errors='coerce')
    lat = pd.to_numeric(df['Lat'], errors='coerce')
    lon = pd.to_numeric(df['Long'], errors='coerce')
    
    # Get reference point (first valid values)
    valid_mask_x = ~np.isnan(ecef_x)
    if not valid_mask_x.any():
        print(f"Warning: No valid ECEF_X data in {filename}")
        return
    
    ref_x = ecef_x[valid_mask_x].iloc[0]
    ref_y = ecef_y[~np.isnan(ecef_y)].iloc[0] if (~np.isnan(ecef_y)).any() else 0
    ref_z = ecef_z[~np.isnan(ecef_z)].iloc[0] if (~np.isnan(ecef_z)).any() else 0
    
    # Get reference lat/long
    ref_lat = lat[~np.isnan(lat)].iloc[0] if (~np.isnan(lat)).any() else 0
    ref_lon = lon[~np.isnan(lon)].iloc[0] if (~np.isnan(lon)).any() else 0
    
    # Convert to radians
    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)
    
    # Create DCM from ECEF to ENU
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    dcm_ecef_to_enu = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])
    
    # Calculate relative ECEF positions
    rel_ecef = np.array([
        ecef_x.values - ref_x,
        ecef_y.values - ref_y,
        ecef_z.values - ref_z
    ])
    
    # Transform to ENU
    rel_enu = dcm_ecef_to_enu @ rel_ecef
    
    rel_e = rel_enu[0]
    rel_n = rel_enu[1]
    rel_u = rel_enu[2]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Relative East Position', 'Relative North Position', 'Relative Up Position'),
        specs=[[{}, {}, {}]]
    )
    
    fig.add_trace(go.Scatter(x=time, y=rel_e, name='ΔE', line=dict(color='red', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=rel_n, name='ΔN', line=dict(color='green', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=time, y=rel_u, name='ΔU', line=dict(color='blue', width=2)), row=1, col=3)
    
    fig.update_yaxes(title_text='ΔE (m)', row=1, col=1, tickformat='.1f')
    fig.update_yaxes(title_text='ΔN (m)', row=1, col=2, tickformat='.1f')
    fig.update_yaxes(title_text='ΔU (m)', row=1, col=3, tickformat='.1f')
    fig.update_xaxes(title_text='Time (s)', row=1, col=1)
    fig.update_xaxes(title_text='Time (s)', row=1, col=2)
    fig.update_xaxes(title_text='Time (s)', row=1, col=3)
    
    fig.update_layout(
        title_text=f'Relative ENU Position vs Time<br>{filename}',
        height=600,
        width=1400,
        showlegend=True,
        hovermode='x unified'
    )
    
    plot_path = os.path.join(file_output_dir, 'relative_enu_interactive.html')
    fig.write_html(plot_path)
    print(f"Saved: {plot_path}")


def plot_gps_and_imu_data_plotly(csv_file_path, output_dir='./plots', window_size=1, show_moving_avg=False):
    """
    Create interactive Plotly plots for GPS/ECEF and IMU data.
    """
    # Read CSV data
    df = read_csv_data(csv_file_path)
    time = get_time_array(df)
    filename = Path(csv_file_path).stem
    file_output_dir = os.path.join(output_dir, filename)
    os.makedirs(file_output_dir, exist_ok=True)
    
    # ===== PLOTLY GPS and ECEF Plot =====
    fig_gps = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Latitude', 'ECEF X', 'Longitude', 'ECEF Y', 'Altitude', 'ECEF Z'),
        specs=[[{}, {}], [{}, {}], [{}, {}]]
    )
    
    # GPS data
    fig_gps.add_trace(go.Scatter(x=time, y=df['Lat'], name='Latitude', line=dict(color='blue')), row=1, col=1)
    fig_gps.add_trace(go.Scatter(x=time, y=df['Long'], name='Longitude', line=dict(color='orange')), row=2, col=1)
    fig_gps.add_trace(go.Scatter(x=time, y=df['Alt'], name='Altitude', line=dict(color='green')), row=3, col=1)
    
    # ECEF data
    fig_gps.add_trace(go.Scatter(x=time, y=df['ECEF_X'], name='ECEF X', line=dict(color='red')), row=1, col=2)
    fig_gps.add_trace(go.Scatter(x=time, y=df['ECEF_Y'], name='ECEF Y', line=dict(color='purple')), row=2, col=2)
    fig_gps.add_trace(go.Scatter(x=time, y=df['ECEF_Z'], name='ECEF Z', line=dict(color='brown')), row=3, col=2)
    
    fig_gps.update_yaxes(title_text='Lat (°)', row=1, col=1, tickformat='.6f')
    fig_gps.update_yaxes(title_text='Long (°)', row=2, col=1, tickformat='.6f')
    fig_gps.update_yaxes(title_text='Alt (m)', row=3, col=1, tickformat='.2f')
    fig_gps.update_yaxes(title_text='ECEF X (m)', row=1, col=2, tickformat='.1f')
    fig_gps.update_yaxes(title_text='ECEF Y (m)', row=2, col=2, tickformat='.1f')
    fig_gps.update_yaxes(title_text='ECEF Z (m)', row=3, col=2, tickformat='.1f')
    fig_gps.update_xaxes(title_text='Time (s)', row=3, col=1)
    fig_gps.update_xaxes(title_text='Time (s)', row=3, col=2)
    
    fig_gps.update_layout(
        title_text=f'GPS and ECEF Data vs Time<br>{filename}',
        height=1000,
        width=1200,
        showlegend=True,
        hovermode='x unified'
    )
    
    gps_path = os.path.join(file_output_dir, 'gps_interactive.html')
    fig_gps.write_html(gps_path)
    print(f"Saved: {gps_path}")
    
    # ===== PLOTLY IMU Plot =====
    fig_imu = make_subplots(
        rows=3, cols=3,
        subplot_titles=('Accel X', 'Gyro X', 'Mag X', 'Accel Y', 'Gyro Y', 'Mag Y', 'Accel Z', 'Gyro Z', 'Mag Z'),
        specs=[[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]]
    )
    
    # Accelerometer
    fig_imu.add_trace(go.Scatter(x=time, y=df['ax'], name='ax', line=dict(color='blue')), row=1, col=1)
    fig_imu.add_trace(go.Scatter(x=time, y=df['ay'], name='ay', line=dict(color='orange')), row=2, col=1)
    fig_imu.add_trace(go.Scatter(x=time, y=df['az'], name='az', line=dict(color='green')), row=3, col=1)
    
    # Gyroscope
    fig_imu.add_trace(go.Scatter(x=time, y=df['gx'], name='gx', line=dict(color='red')), row=1, col=2)
    fig_imu.add_trace(go.Scatter(x=time, y=df['gy'], name='gy', line=dict(color='purple')), row=2, col=2)
    fig_imu.add_trace(go.Scatter(x=time, y=df['gz'], name='gz', line=dict(color='brown')), row=3, col=2)
    
    # Magnetometer
    fig_imu.add_trace(go.Scatter(x=time, y=df['magx'], name='magx', line=dict(color='cyan')), row=1, col=3)
    fig_imu.add_trace(go.Scatter(x=time, y=df['magy'], name='magy', line=dict(color='magenta')), row=2, col=3)
    fig_imu.add_trace(go.Scatter(x=time, y=df['magz'], name='magz', line=dict(color='lime')), row=3, col=3)
    
    fig_imu.update_yaxes(title_text='ax (m/s²)', row=1, col=1)
    fig_imu.update_yaxes(title_text='ay (m/s²)', row=2, col=1)
    fig_imu.update_yaxes(title_text='az (m/s²)', row=3, col=1)
    fig_imu.update_yaxes(title_text='gx (rad/s)', row=1, col=2)
    fig_imu.update_yaxes(title_text='gy (rad/s)', row=2, col=2)
    fig_imu.update_yaxes(title_text='gz (rad/s)', row=3, col=2)
    fig_imu.update_yaxes(title_text='magx (µT)', row=1, col=3)
    fig_imu.update_yaxes(title_text='magy (µT)', row=2, col=3)
    fig_imu.update_yaxes(title_text='magz (µT)', row=3, col=3)
    fig_imu.update_xaxes(title_text='Time (s)', row=3, col=1)
    fig_imu.update_xaxes(title_text='Time (s)', row=3, col=2)
    fig_imu.update_xaxes(title_text='Time (s)', row=3, col=3)
    
    fig_imu.update_layout(
        title_text=f'IMU Sensor Data vs Time<br>{filename}',
        height=1000,
        width=1400,
        showlegend=True,
        hovermode='x unified'
    )
    
    imu_path = os.path.join(file_output_dir, 'imu_interactive.html')
    fig_imu.write_html(imu_path)
    print(f"Saved: {imu_path}")


def plot_gps_and_imu_data(csv_file_path, output_dir='./plots', window_size=1, show_moving_avg=False):
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
    
    # Calculate moving averages if enabled
    if show_moving_avg:
        lat_ma = calculate_moving_average(df['Lat'].values, window_size=window_size)
        long_ma = calculate_moving_average(df['Long'].values, window_size=window_size)
        alt_ma = calculate_moving_average(df['Alt'].values, window_size=window_size)
        ecef_x_ma = calculate_moving_average(df['ECEF_X'].values, window_size=window_size)
        ecef_y_ma = calculate_moving_average(df['ECEF_Y'].values, window_size=window_size)
        ecef_z_ma = calculate_moving_average(df['ECEF_Z'].values, window_size=window_size)
    
    # Left column: GPS data (Lat, Long, Alt)
    axes1[0, 0].plot(time, df['Lat'], linewidth=1.5, label='Data')
    if show_moving_avg:
        axes1[0, 0].plot(time, lat_ma, linewidth=2, color='red', linestyle='--', label='MA')
        axes1[0, 0].legend(fontsize=8)
    axes1[0, 0].set_ylabel('Latitude (°)')
    axes1[0, 0].grid(True, alpha=0.3)
    axes1[0, 0].set_title('Latitude')
    axes1[0, 0].ticklabel_format(style='plain', axis='y', useOffset=False)
    
    axes1[1, 0].plot(time, df['Long'], linewidth=1.5, color='orange', label='Data')
    if show_moving_avg:
        axes1[1, 0].plot(time, long_ma, linewidth=2, color='red', linestyle='--', label='MA')
        axes1[1, 0].legend(fontsize=8)
    axes1[1, 0].set_ylabel('Longitude (°)')
    axes1[1, 0].grid(True, alpha=0.3)
    axes1[1, 0].set_title('Longitude')
    axes1[1, 0].ticklabel_format(style='plain', axis='y', useOffset=False)
    
    axes1[2, 0].plot(time, df['Alt'], linewidth=1.5, color='green', label='Data')
    if show_moving_avg:
        axes1[2, 0].plot(time, alt_ma, linewidth=2, color='red', linestyle='--', label='MA')
        axes1[2, 0].legend(fontsize=8)
    axes1[2, 0].set_ylabel('Altitude (m)')
    axes1[2, 0].set_xlabel('Time (s)')
    axes1[2, 0].grid(True, alpha=0.3)
    axes1[2, 0].set_title('Altitude')
    axes1[2, 0].ticklabel_format(style='plain', axis='y', useOffset=False)
    
    # Right column: ECEF data (ECEF_X, ECEF_Y, ECEF_Z)
    axes1[0, 1].plot(time, df['ECEF_X'], linewidth=1.5, color='red', label='Data')
    if show_moving_avg:
        axes1[0, 1].plot(time, ecef_x_ma, linewidth=2, color='darkred', linestyle='--', label='MA')
        axes1[0, 1].legend(fontsize=8)
    axes1[0, 1].set_ylabel('ECEF_X (m)')
    axes1[0, 1].grid(True, alpha=0.3)
    axes1[0, 1].set_title('ECEF X')
    axes1[0, 1].ticklabel_format(style='plain', axis='y', useOffset=False)
    
    axes1[1, 1].plot(time, df['ECEF_Y'], linewidth=1.5, color='purple', label='Data')
    if show_moving_avg:
        axes1[1, 1].plot(time, ecef_y_ma, linewidth=2, color='red', linestyle='--', label='MA')
        axes1[1, 1].legend(fontsize=8)
    axes1[1, 1].set_ylabel('ECEF_Y (m)')
    axes1[1, 1].grid(True, alpha=0.3)
    axes1[1, 1].set_title('ECEF Y')
    axes1[1, 1].ticklabel_format(style='plain', axis='y', useOffset=False)
    
    axes1[2, 1].plot(time, df['ECEF_Z'], linewidth=1.5, color='brown', label='Data')
    if show_moving_avg:
        axes1[2, 1].plot(time, ecef_z_ma, linewidth=2, color='red', linestyle='--', label='MA')
        axes1[2, 1].legend(fontsize=8)
    axes1[2, 1].set_ylabel('ECEF_Z (m)')
    axes1[2, 1].set_xlabel('Time (s)')
    axes1[2, 1].grid(True, alpha=0.3)
    axes1[2, 1].set_title('ECEF Z')
    axes1[2, 1].ticklabel_format(style='plain', axis='y', useOffset=False)
    
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
    
    # Calculate moving averages for IMU data if enabled
    if show_moving_avg:
        ax_ma = calculate_moving_average(df['ax'].values, window_size=window_size)
        ay_ma = calculate_moving_average(df['ay'].values, window_size=window_size)
        az_ma = calculate_moving_average(df['az'].values, window_size=window_size)
        gx_ma = calculate_moving_average(df['gx'].values, window_size=window_size)
        gy_ma = calculate_moving_average(df['gy'].values, window_size=window_size)
        gz_ma = calculate_moving_average(df['gz'].values, window_size=window_size)
        magx_ma = calculate_moving_average(df['magx'].values, window_size=window_size)
        magy_ma = calculate_moving_average(df['magy'].values, window_size=window_size)
        magz_ma = calculate_moving_average(df['magz'].values, window_size=window_size)
    
    # Column 1: Accelerometer (ax, ay, az)
    axes2[0, 0].plot(time, df['ax'], linewidth=1.5, label='Data')
    if show_moving_avg:
        axes2[0, 0].plot(time, ax_ma, linewidth=2, color='red', linestyle='--', label='MA')
        axes2[0, 0].legend(fontsize=8)
    axes2[0, 0].set_ylabel('ax (m/s²)')
    axes2[0, 0].grid(True, alpha=0.3)
    axes2[0, 0].set_title('Accelerometer X')
    
    axes2[1, 0].plot(time, df['ay'], linewidth=1.5, color='orange', label='Data')
    if show_moving_avg:
        axes2[1, 0].plot(time, ay_ma, linewidth=2, color='red', linestyle='--', label='MA')
        axes2[1, 0].legend(fontsize=8)
    axes2[1, 0].set_ylabel('ay (m/s²)')
    axes2[1, 0].grid(True, alpha=0.3)
    axes2[1, 0].set_title('Accelerometer Y')
    
    axes2[2, 0].plot(time, df['az'], linewidth=1.5, color='green', label='Data')
    if show_moving_avg:
        axes2[2, 0].plot(time, az_ma, linewidth=2, color='red', linestyle='--', label='MA')
        axes2[2, 0].legend(fontsize=8)
    axes2[2, 0].set_ylabel('az (m/s²)')
    axes2[2, 0].set_xlabel('Time (s)')
    axes2[2, 0].grid(True, alpha=0.3)
    axes2[2, 0].set_title('Accelerometer Z')
    
    # Column 2: Gyroscope (gx, gy, gz)
    axes2[0, 1].plot(time, df['gx'], linewidth=1.5, color='red', label='Data')
    if show_moving_avg:
        axes2[0, 1].plot(time, gx_ma, linewidth=2, color='darkred', linestyle='--', label='MA')
        axes2[0, 1].legend(fontsize=8)
    axes2[0, 1].set_ylabel('gx (rad/s)')
    axes2[0, 1].grid(True, alpha=0.3)
    axes2[0, 1].set_title('Gyroscope X')
    
    axes2[1, 1].plot(time, df['gy'], linewidth=1.5, color='purple', label='Data')
    if show_moving_avg:
        axes2[1, 1].plot(time, gy_ma, linewidth=2, color='red', linestyle='--', label='MA')
        axes2[1, 1].legend(fontsize=8)
    axes2[1, 1].set_ylabel('gy (rad/s)')
    axes2[1, 1].grid(True, alpha=0.3)
    axes2[1, 1].set_title('Gyroscope Y')
    
    axes2[2, 1].plot(time, df['gz'], linewidth=1.5, color='brown', label='Data')
    if show_moving_avg:
        axes2[2, 1].plot(time, gz_ma, linewidth=2, color='red', linestyle='--', label='MA')
        axes2[2, 1].legend(fontsize=8)
    axes2[2, 1].set_ylabel('gz (rad/s)')
    axes2[2, 1].set_xlabel('Time (s)')
    axes2[2, 1].grid(True, alpha=0.3)
    axes2[2, 1].set_title('Gyroscope Z')
    
    # Column 3: Magnetometer (magx, magy, magz)
    axes2[0, 2].plot(time, df['magx'], linewidth=1.5, color='cyan', label='Data')
    if show_moving_avg:
        axes2[0, 2].plot(time, magx_ma, linewidth=2, color='red', linestyle='--', label='MA')
        axes2[0, 2].legend(fontsize=8)
    axes2[0, 2].set_ylabel('magx (µT)')
    axes2[0, 2].grid(True, alpha=0.3)
    axes2[0, 2].set_title('Magnetometer X')
    
    axes2[1, 2].plot(time, df['magy'], linewidth=1.5, color='magenta', label='Data')
    if show_moving_avg:
        axes2[1, 2].plot(time, magy_ma, linewidth=2, color='red', linestyle='--', label='MA')
        axes2[1, 2].legend(fontsize=8)
    axes2[1, 2].set_ylabel('magy (µT)')
    axes2[1, 2].grid(True, alpha=0.3)
    axes2[1, 2].set_title('Magnetometer Y')
    
    axes2[2, 2].plot(time, df['magz'], linewidth=1.5, color='lime', label='Data')
    if show_moving_avg:
        axes2[2, 2].plot(time, magz_ma, linewidth=2, color='red', linestyle='--', label='MA')
        axes2[2, 2].legend(fontsize=8)
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
    
    # Generate individual column plots
    print(f"Generating individual column plots for {filename}...")
    plot_individual_columns(csv_file_path, output_dir, window_size=window_size, show_moving_avg=show_moving_avg)
    
    # Generate relative ECEF plots
    print(f"Generating relative ECEF plots for {filename}...")
    plot_relative_ecef_data(csv_file_path, output_dir)
    
    # Generate relative ENU plots
    print(f"Generating relative ENU plots for {filename}...")
    plot_relative_enu_data(csv_file_path, output_dir)


def plot_all_data_with_plotly(csv_file_path, output_dir='./plots', window_size=1, show_moving_avg=False):
    """
    Generate all plots including interactive Plotly versions.
    """
    # Generate matplotlib plots
    plot_gps_and_imu_data(csv_file_path, output_dir, window_size=window_size, show_moving_avg=show_moving_avg)
    
    # Generate Plotly interactive plots
    print(f"\nGenerating interactive Plotly plots for {Path(csv_file_path).stem}...")
    plot_gps_and_imu_data_plotly(csv_file_path, output_dir, window_size=window_size, show_moving_avg=show_moving_avg)
    plot_individual_columns_plotly(csv_file_path, output_dir, window_size=window_size, show_moving_avg=show_moving_avg)
    plot_relative_ecef_data_plotly(csv_file_path, output_dir)
    plot_relative_enu_data_plotly(csv_file_path, output_dir)


def plot_all_csv_files(field_data_dir, output_dir='./plots', window_size=1, show_moving_avg=False, use_plotly=False):
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
            if use_plotly:
                plot_all_data_with_plotly(str(csv_file), output_dir, window_size=window_size, show_moving_avg=show_moving_avg)
            else:
                plot_gps_and_imu_data(str(csv_file), output_dir, window_size=window_size, show_moving_avg=show_moving_avg)
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")


def generate_all_plots(field_data_dir=None, window_size=1, show_moving_avg=False, use_plotly=False):
    """
    Generate all plots for all CSV files in the field_data directory.
    This is a convenient wrapper function to generate all plots at once.
    
    Args:
        field_data_dir: Path to field_data directory. If None, uses default relative path.
        window_size: Window size for moving average (default: 1)
        show_moving_avg: Whether to show moving average lines (default: False)
        use_plotly: Whether to generate interactive Plotly plots (default: False)
    """
    if field_data_dir is None:
        current_dir = Path(__file__).parent.parent
        field_data_dir = current_dir / 'field_data'
    
    output_dir = Path(field_data_dir) / 'plots'
    
    print("=" * 60)
    print("GENERATING ALL PLOTS FOR FIELD DATA")
    if show_moving_avg:
        print(f"Moving Average: ENABLED (Window Size: {window_size} samples)")
    else:
        print("Moving Average: DISABLED")
    if use_plotly:
        print("Interactive Plotly Plots: ENABLED")
    print("=" * 60)
    plot_all_csv_files(str(field_data_dir), str(output_dir), window_size=window_size, show_moving_avg=show_moving_avg, use_plotly=use_plotly)
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
    parser.add_argument(
        '--no-avg',
        action='store_false',
        dest='show_moving_avg',
        default=False,
        help='Disable moving average lines (default)'
    )
    parser.add_argument(
        '--avg',
        action='store_true',
        dest='show_moving_avg',
        help='Enable moving average lines'
    )
    parser.add_argument(
        '-w', '--window',
        type=int,
        default=100,
        help='Moving average window size in samples (default: 100, only used with --avg)'
    )
    parser.add_argument(
        '--plotly',
        action='store_true',
        help='Generate interactive Plotly HTML plots in addition to matplotlib plots'
    )
    
    args = parser.parse_args()
    
    # Get the path to the field_data directory
    current_dir = Path(__file__).parent.parent
    field_data_dir = current_dir / 'field_data'
    output_dir = Path(args.output) if args.output else field_data_dir / 'plots'
    
    if args.all:
        # Generate all plots
        generate_all_plots(str(field_data_dir), window_size=args.window, show_moving_avg=args.show_moving_avg, use_plotly=args.plotly)
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
                if args.plotly:
                    plot_all_data_with_plotly(str(file_path), str(output_dir), window_size=args.window, show_moving_avg=args.show_moving_avg)
                else:
                    plot_gps_and_imu_data(str(file_path), str(output_dir), window_size=args.window, show_moving_avg=args.show_moving_avg)
                print(f"Plots saved to: {output_dir}")
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
        else:
            print(f"Error: File not found: {file_path}")
    else:
        # Default: show help if no arguments provided
        parser.print_help()

