# =============================================================================
# File: orekit_helpers.py
# Author: Mia Tian
# Created: 4/2025
#
# Description: Helper functions for working with Orekit in Python.
# =============================================================================


from org.orekit.frames import Frame
from org.orekit.time import AbsoluteDate, TimeScalesFactory

from org.hipparchus.geometry.euclidean.threed import Vector3D

from math import degrees
import numpy as np
from datetime import datetime

def eci_to_geodetic(earth, date: AbsoluteDate, position: Vector3D, frame: Frame):
    """
    Converts Earth-Centered Inertial (ECI) coordinates to geodetic coordinates (latitude, longitude, altitude).
    Parameters:
    earth (object): The Earth model used for the transformation.
    date (AbsoluteDate): The date and time of the transformation.
    position (Vector3D): The position in ECI coordinates.
    frame (Frame): The reference frame of the ECI coordinates.
    Returns:
    tuple: A tuple containing the latitude (degrees), longitude (degrees), and altitude (meters).
    """

    geodetic_point = earth.transform(position, frame, date)
    lat = degrees(geodetic_point.getLatitude())
    lon = degrees(geodetic_point.getLongitude())
    alt = geodetic_point.getAltitude()

    return lat, lon, alt

def absolutedate_to_datetime(date: AbsoluteDate):
    """
    Convert an AbsoluteDate object to a Python datetime object.
    Args:
        date (AbsoluteDate): The AbsoluteDate object to be converted.
    Returns:
        datetime: A Python datetime object representing the same date and time as the input AbsoluteDate.
    """

    date_time = date.getComponents(TimeScalesFactory.getUTC())
    date = date_time.getDate()
    time = date_time.getTime()
    year = date.getYear()
    month = date.getMonth()
    day = date.getDay()
    hour = time.getHour()
    minute = time.getMinute()
    second = int(time.getSecond())

    return datetime(year, month, day, hour, minute, second)

def eci2ric(x0, x):
    """
    Convert a satellite state vector from Earth-Centered Inertial (ECI) coordinates to Radial, In-track, and Cross-track (RIC) coordinates.
    Parameters:
    x0 (numpy.ndarray): Reference state vector in ECI coordinates. The first three elements are the position components, and the next three elements are the velocity components.
    x (numpy.ndarray): Query state vector in ECI coordinates. The first three elements are the position components, and the next three elements are the velocity components.
    Returns:
    numpy.ndarray: State vector in RIC coordinates. The first three elements are the position components in the RIC frame, and the next three elements are the velocity components in the RIC frame.
    """

    # compute the radial unit vector
    r_unit = x0[0:3] / np.linalg.norm(x0[0:3])

    # compute the in-track unit vector aligned with the velocity of the reference state
    i_unit = x0[3:6] / np.linalg.norm(x0[3:6])

    # compute the cross-track unit vector
    c_unit = np.cross(r_unit, i_unit)

    # compute the state vector in the RTN frame
    x_diff = x - x0
    x_ric = np.array([np.dot(r_unit, x_diff[0:3]), np.dot(i_unit, x_diff[0:3]), np.dot(c_unit, x_diff[0:3]), np.dot(r_unit, x_diff[3:6]), np.dot(i_unit, x_diff[3:6]), np.dot(c_unit, x_diff[3:6])])

    return x_ric