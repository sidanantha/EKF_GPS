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