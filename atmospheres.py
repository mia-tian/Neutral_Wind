from org.orekit.frames import Frame
from org.orekit.time import AbsoluteDate
from org.orekit.utils import PVCoordinates
from org.orekit.models.earth.atmosphere import DTM2000, HarrisPriester, JB2008, NRLMSISE00, Atmosphere, SimpleExponentialAtmosphere, PythonAtmosphere

from org.hipparchus.geometry.euclidean.threed import Vector3D

from netCDF4 import Dataset
import numpy as np
from numpy.linalg import norm
import os
from datetime import datetime, timedelta
from pymsis import msis


from orekit_helpers import absolutedate_to_datetime, eci_to_geodetic, eci2ric

data_path = '/Volumes/TOSHIBA EXT/Research/WAMIPE Data/'

def get_msis_density(earth, date: AbsoluteDate, position: Vector3D, frame: Frame):
    """
    Calculate the atmospheric density using the MSIS model.
    Parameters:
    earth (object): The Earth model object.
    date (AbsoluteDate): The date and time of the calculation.
    position (Vector3D): The position vector in the ECI frame.
    frame (Frame): The reference frame of the position vector.
    Returns:
    float: The atmospheric density at the given position and time.
    """

    lat, lon, alt = eci_to_geodetic(earth, date, position, frame)
    alt = alt * 1e-3 # convert to km

    date_time = absolutedate_to_datetime(date)
    date = np.datetime64(date_time)
    dates = np.array([date])
    lat = np.array([lat])
    lon = np.array([lon])
    alt = np.array([alt])

    data = msis.run(dates, lat, lon, alt, geomagnetic_activity=-1)

    return data[:,0].item()

def get_file(date, folder, file_header):
    """
    Constructs the file path for a netcdf WAM-IPE data file based on the given date and file header.
    Args:
        date (AbsoluteDate): The date for which the file is needed.
        file_header (str): The header string to be used in the file name.
    Returns:
        str: The constructed file path.
    """

    # Round the minute to the nearest 10 minutes. WAM-IPE data is every 10min
    dt = absolutedate_to_datetime(date)
    dt += timedelta(minutes=5)
    dt = dt.replace(minute=(dt.minute // 10) * 10, second=0, microsecond=0)
    year, month, day, hour, minute = dt.year, dt.month, dt.day, dt.hour, dt.minute

    # Construct the filename based on the date and time
    folder_name = f"wam10_fixed_height_{year}{month:02d}{day:02d}"
    file_name = f"{file_header}.wfs.t00z.wam10.{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}00.nc"
    file_path = f"{folder}/{file_name}"

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Date doesn't exist in file system.")
    
    return file_path

def get_wamipe_density(earth, date: AbsoluteDate, position: Vector3D, frame: Frame):
    """
    Retrieve the atmospheric density from the WAM-IPE model at a given position and time.
    Parameters:
    earth (object): The Earth model object.
    date (AbsoluteDate): The date and time for which the density is to be retrieved.
    position (Vector3D): The position in ECI coordinates.
    frame (Frame): The reference frame of the position.
    Returns:
    float: The atmospheric density at the specified position and time.
    """

    file = get_file(date, data_path + 'density', 'wam_fixed_height')
    ncfid_i = Dataset(file)

    height_list = ncfid_i.variables['hlevs'][:]
    lat_list = ncfid_i.variables['lat'][:]
    lon_list = ncfid_i.variables['lon'][:]
    density_values = ncfid_i.variables['den']

    lat, lon, alt = eci_to_geodetic(earth, date, position, frame)
    alt = alt * 1e-3 # convert to km

    # find the closest indices in the data
    closest_height_index = np.abs(height_list - alt).argmin()
    closest_lat_index = np.abs(lat_list - lat).argmin()
    closest_lon_index = np.abs(lon_list - lon).argmin()

    density = density_values[0, closest_height_index, closest_lat_index, closest_lon_index].item()

    return density

def get_wamipe_wind(earth, date: AbsoluteDate, position: Vector3D, frame: Frame):
    """
    Retrieves the neutral wind velocity from the WAM-IPE model at a given position and time.
    Parameters:
    earth (object): The Earth model object.
    date (AbsoluteDate): The date and time for which the wind data is required.
    position (Vector3D): The position in ECI coordinates.
    frame (Frame): The reference frame of the position.
    Returns:
    Vector3D: The neutral wind velocity vector in the spacecraft body frame at the specified position and time.
    """

    file = get_file(date, data_path + 'fixed_height_wind', 'fixed_height_wind')
    ncfid_i = Dataset(file)

    height_list = ncfid_i.variables['height'][:]
    lat_list = ncfid_i.variables['lat'][:]
    lon_list = ncfid_i.variables['lon'][:]
    u_neutral_list = ncfid_i.variables['u_neutral']
    v_neutral_list = ncfid_i.variables['v_neutral']
    w_neutral_list = ncfid_i.variables['w_neutral']

    lat, lon, alt = eci_to_geodetic(earth, date, position, frame)

    closest_height_index = np.abs(height_list - alt).argmin()
    closest_lat_index = np.abs(lat_list - lat).argmin()
    closest_lon_index = np.abs(lon_list - lon).argmin()

    u_neutral = u_neutral_list[closest_height_index, closest_lat_index, closest_lon_index].item()
    v_neutral = v_neutral_list[closest_height_index, closest_lat_index, closest_lon_index].item()
    w_neutral = w_neutral_list[closest_height_index, closest_lat_index, closest_lon_index].item()
    
    wind_velocity_uvw = Vector3D(u_neutral, v_neutral, w_neutral)
    return wind_velocity_uvw


def transform_wind_to_ecef(pos_ecef: Vector3D, wind_uvw: Vector3D):
    """
    Transforms the wind velocity from the UVW Frame to the ECEF frame.

    Parameters:
    pos_ecef (Vector3D): The position of the spacecraft in the body frame.
    wind_body (Vector3D): The wind velocity in the spacecraft body frame.

    Returns:
    Vector3D: The wind velocity in the ECEF frame.
    """
    # Convert pos_ecef and wind_body to numpy arrays
    pos_body = np.array([pos_ecef.getX(), pos_ecef.getY(), pos_ecef.getZ()])
    # U: Zonal, V: Meridional, W: Vertical
    wind_sc_body = np.array([wind_uvw.getZ(), wind_uvw.getX(), wind_uvw.getY()])

    # Normalize the position vector to get the radial direction (s/c body x-axis in ECEF)
    x_hat = pos_body / norm(pos_body)

    z_hat_ECEF = np.array([0, 0, 1])
    # Compute the s/c body y-axis in ECEF
    y_hat = np.cross(z_hat_ECEF, x_hat)
    y_hat /= norm(x_hat)

    # Compute the s/c body z-axis in ECEF
    z_hat = np.cross(x_hat, y_hat)

    # Construct the rotation matrix from the spacecraft body frame to the ECEF frame
    rotation_matrix = np.vstack([x_hat, y_hat, z_hat]).T

    # Transform the wind vector to the ECEF frame
    wind_ecef = np.dot(rotation_matrix, wind_sc_body)

    # Return the wind velocity in the ECEF frame as a Vector3D
    return Vector3D(float(wind_ecef[0]), float(wind_ecef[1]), float(wind_ecef[2]))

class CustomAtmosphere(PythonAtmosphere):
    """
    CustomAtmosphere is a custom implementation of the PythonAtmosphere class
    that uses the JB2008 atmospheric model to compute atmospheric density and
    velocity.

    Attributes:
        atm (JB2008): An instance of the JB2008 atmospheric model.
        earth (Body): The central body (Earth) for the atmospheric model.

    Methods:
        getDensity(date: AbsoluteDate, position: Vector3D, frame: Frame) -> float:
            Computes the atmospheric density at a given date, position, and frame.
        
        getVelocity(date: AbsoluteDate, position: Vector3D, frame: Frame) -> Vector3D:
            Computes the inertial velocity of atmosphere molecules at a given date,
            position, and frame. By default, the atmosphere is assumed to have a null
            velocity in the central body frame.
    """
    def __init__(self, cswl, sun, earth):
        super().__init__()
        self.earth = earth
    def getDensity(self, date: AbsoluteDate, position: Vector3D, frame: Frame):
        # TODO for partcipants: get the density from your model given date, position, and frame
        return get_wamipe_density(self.earth, date, position, frame)
        # return self.atm.getDensity(date, position, frame)
        # return get_msis_density(self.earth, date, position, frame)
    def getVelocity(self, date: AbsoluteDate, position: Vector3D, frame: Frame):
        '''
        Get the inertial velocity of atmosphere molecules.
        By default, atmosphere is supposed to have a null
        velocity in the central body frame.
        '''
        # get the transform from ECEF to the inertial frame
        ecef_to_eci = self.earth.getBodyFrame().getKinematicTransformTo(frame, date)
        # Inverse transform the position to ECEF
        pos_ecef = ecef_to_eci.getStaticInverse().transformPosition(position)
        # Create PVCoordinates object assuming zero velocity in body frame
        pv_body = PVCoordinates(pos_ecef, Vector3D.ZERO)
        # Transform the position/velocity (PV) coordinates to the given frame
        pvFrame = ecef_to_eci.transformOnlyPV(pv_body)
        # Return the velocity in the current frame
        return pvFrame.getVelocity()

class WindyCustomAtmosphere(PythonAtmosphere):
    """
    CustomAtmosphere is a custom implementation of the PythonAtmosphere class
    that uses the JB2008 atmospheric model to compute atmospheric density and
    velocity.

    Attributes:
        atm (JB2008): An instance of the JB2008 atmospheric model.
        earth (Body): The central body (Earth) for the atmospheric model.

    Methods:
        getDensity(date: AbsoluteDate, position: Vector3D, frame: Frame) -> float:
            Computes the atmospheric density at a given date, position, and frame.
        
        getVelocity(date: AbsoluteDate, position: Vector3D, frame: Frame) -> Vector3D:
            Computes the inertial velocity of atmosphere molecules at a given date,
            position, and frame. By default, the atmosphere is assumed to have a null
            velocity in the central body frame.
    """
    def __init__(self, cswl, sun, earth):
        super().__init__()
        # TODO for partcipants: initialize your atmospheric density model here
        self.earth = earth
        self.wind = []
        self.date = []
    def getDensity(self, date: AbsoluteDate, position: Vector3D, frame: Frame):
        return get_wamipe_density(self.earth, date, position, frame)
        # return self.atm.getDensity(date, position, frame)
        # return get_msis_density(self.earth, date, position, frame)
    def getVelocity(self, date: AbsoluteDate, position: Vector3D, frame: Frame):
        '''
        Get the inertial velocity of atmosphere molecules
        using WAM-IPE data.
        '''
        # get the transform from body frame (ECEF) to the inertial frame
        ecef_to_eci = self.earth.getBodyFrame().getKinematicTransformTo(frame, date)
        # Inverse transform the position to ECEF
        pos_ecef = ecef_to_eci.getStaticInverse().transformPosition(position)

        wind_uvw = get_wamipe_wind(self.earth, date, position, frame)

        wind_ECEF = transform_wind_to_ecef(pos_ecef, wind_uvw)
        # Create PVCoordinates object in body frame (ECEF)
        pv_body = PVCoordinates(pos_ecef, wind_ECEF)
        # Transform the position/velocity (PV) coordinates to the given frame
        pvFrame = ecef_to_eci.transformOnlyPV(pv_body)
        # Return the velocity in the current frame
        wind_ECI = pvFrame.getVelocity()
        return wind_ECI
    
    # def getVelocity(self, date: AbsoluteDate, position: Vector3D, frame: Frame):
    #     '''
    #     Get the inertial velocity of atmosphere molecules.
    #     By default, atmosphere is supposed to have a null
    #     velocity in ECEF.
    #     '''
    #     # get the transform from ECEF to the inertial frame
    #     ecef_to_eci = self.earth.getBodyFrame().getKinematicTransformTo(frame, date)
    #     # Inverse transform the position to ECEF
    #     pos_ecef = ecef_to_eci.getStaticInverse().transformPosition(position)

    #     # Compute the direction of velocity in the body frame (but equatorial)
    #     vel_ecef = Vector3D(-pos_ecef.getY(), pos_ecef.getX(), 0.0)

    #     # Reverse the direction of the velocity to get the wind velocity. Normalize to 1000.0 m/s
    #     wind = vel_ecef.negate().scalarMultiply(1000 / vel_ecef.getNorm())

    #     # Create PVCoordinates object with the wind velocity
    #     pv_body = PVCoordinates(pos_ecef, wind)

    #     # Transform the position/velocity (PV) coordinates to the given frame
    #     pvFrame = ecef_to_eci.transformOnlyPV(pv_body)

    #     # Return the velocity in the current frame
    #     return pvFrame.getVelocity()
  