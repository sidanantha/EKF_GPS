import pandas as pd
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

file_path = '/content/drive/My Drive/test/test - Sheet1.csv'

def data_processing(filepath):
    df = pd.read_csv(filepath)
    time = df['UTC'].to_numpy()
    lat = df['Lat'].to_numpy()
    lon = df['Lon'].to_numpy()
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
    starttime = time[0]
    endtime = time[-1]
    timestep = (endtime - starttime) / len(time)

    return time, starttime, endtime, timestep, lat, lon, alt, x, y, z, ax, ay, az, gx, gy, gz
