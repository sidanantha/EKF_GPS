import csv
import pandas as pd


def data_processing(filepath):
    df = pd.read_csv(filepath)
    time = df['UTC'].to_numpy()
    lat = df['Lat'].to_numpy()
    long = df['Long'].to_numpy()
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

    return time, lat, long, alt, x, y, z, ax, ay, az, gx, gy, gz


