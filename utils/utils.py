# utils.py
# Utility functions for the EKF algorithm

import numpy as np


def ecef_to_lla(x, y, z):
    '''
    Convert ECEF coordinates to LLA coordinates
    Inputs:
        x: ECEF x coordinate, scalar
        y: ECEF y coordinate, scalar
        z: ECEF z coordinate, scalar
    Output:
        lat: Latitude, scalar
        long: Longitude, scalar
        alt: Altitude, scalar
    '''
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    long = np.arctan2(y, x)
    alt = np.sqrt(x**2 + y**2 + z**2) - 6378137
    return lat, long, alt


def relative_position(x_observer, x_drone):
    '''
    Calculate the relative position between the observer and the drone
    Inputs:
        x_observer: Observer position, 3x1 matrix
        x_drone: Drone position, 3x1 matrix
    Output:
        relative_position: Relative position, 3x1 matrix
    '''
    relative_position = x_drone - x_observer
    return relative_position


def ecef_to_enu(x_ecef, y_ecef, z_ecef, lat_ref, lon_ref):
    '''
    Convert ECEF coordinates to ENU coordinates
    Inputs:
        x: ECEF x coordinate, scalar
        y: ECEF y coordinate, scalar
        z: ECEF z coordinate, scalar
        lat: Latitude, scalar
        lon: Longitude, scalar
    Output:
    '''
    
    # Convert to radians
    lat_rad = np.radians(lat_ref)
    lon_rad = np.radians(lon_ref)
    
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
    
    # Create current ECEF position vector
    pos_ecef = np.array([x_ecef, y_ecef, z_ecef])
    
    # Convert ECEF position vector to ENU position vector
    pos_enu = dcm_ecef_to_enu @ pos_ecef
    
    return pos_enu


def compute_yaw(B_x_sensor, B_y_sensor, B_z_sensor, roll, pitch, R_sensor_to_body):
    '''
    Compute the yaw angle from the magnetic field vector and the roll and pitch angles
    Inputs:
        B_x_sensor: Magnetic field x component in sensor frame, scalar
        B_y_sensor: Magnetic field y component in sensor frame, scalar
        B_z_sensor: Magnetic field z component in sensor frame, scalar
        roll: Roll angle, scalar
        pitch: Pitch angle, scalar
        R_sensor_to_body: Rotation matrix from sensor frame to body frame, 3x3 matrix
    Output:
    '''
    
    # Compute rotation from body-fixed-frame to the ENU frame:
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(0), -np.sin(0), 0], [np.sin(0), np.cos(0), 0], [0, 0, 1]])
    R_body_to_enu = Rx @ Ry @ Rz
    
    # Magnetic field vector is in sensor frame
    # Convert from sensor to body frame:
    B_body = R_sensor_to_body @ np.array([B_x_sensor, B_y_sensor, B_z_sensor])
    # Convert from body to ENU frame:
    B_enu = R_body_to_enu @ B_body
    
    # Compute the yaw angle:
    yaw = np.arctan2(B_enu[1], B_enu[0])
    
    return yaw
    
    
def yaw_from_gps(GPS_ENU_prev, GPS_ENU_curr):
    '''
    Compute the yaw angle from the previous and current GPS ENU positions
    Inputs:
        GPS_ENU_prev: Previous GPS ENU position, 3x1 matrix
        GPS_ENU_curr: Current GPS ENU position, 3x1 matrix
    Output:
        yaw: Yaw angle, scalar
    '''
    
    # Compute the delta:
    delta_GPS_ENU = GPS_ENU_curr - GPS_ENU_prev
    # Compute the yaw angle:
    yaw = np.arctan2(delta_GPS_ENU[1], delta_GPS_ENU[0])
    
    # Because this is not perfect, add about 10 degrees of noise to the yaw angle:
    yaw += np.random.normal(0, np.radians(10))
    return yaw