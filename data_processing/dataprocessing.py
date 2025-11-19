import pandas as pd
import numpy as np

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
    Returns None if the time string is 'NA' or invalid.
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


def data_processing(filepath):
    # Read CSV with robust header handling
    df = read_csv_data(filepath)
    
    # Convert time to seconds
    time = np.full(len(df), np.nan, dtype=float)
    for i, time_val in enumerate(df['UTC']):
        sec = convert_time_to_seconds(time_val)
        if sec is not None:
            time[i] = sec
    
    # Get data columns
    lat = df['Lat'].to_numpy()
    lon = df['Long'].to_numpy()
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
    
    # Remove rows with invalid time values
    valid_indices = ~np.isnan(time)
    time = time[valid_indices]
    lat = lat[valid_indices]
    lon = lon[valid_indices]
    alt = alt[valid_indices]
    x = x[valid_indices]
    y = y[valid_indices]
    z = z[valid_indices]
    ax = ax[valid_indices]
    ay = ay[valid_indices]
    az = az[valid_indices]
    gx = gx[valid_indices]
    gy = gy[valid_indices]
    gz = gz[valid_indices]
    
    starttime = time[0]
    endtime = time[-1]
    timestep = (endtime - starttime) / len(time)

    return time, starttime, endtime, timestep, lat, lon, alt, x, y, z, ax, ay, az, gx, gy, gz
