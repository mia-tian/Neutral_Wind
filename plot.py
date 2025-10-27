# Initialize orekit and JVM
import orekit
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime

from org.orekit.orbits import KeplerianOrbit, EquinoctialOrbit, PositionAngleType, OrbitType, CartesianOrbit
from org.orekit.frames import FramesFactory, LOFType, Frame
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants, PVCoordinates
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.propagation import SpacecraftState
from org.orekit.bodies import OneAxisEllipsoid, CelestialBodyFactory, CelestialBody
from org.orekit.utils import IERSConventions
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction, OceanTides, SolidTides
from orekit import JArray_double, JArray
from java.util import ArrayList

from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient, RadiationSensitive
from org.orekit.models.earth.atmosphere.data import CssiSpaceWeatherData, JB2008SpaceEnvironmentData
from org.orekit.forces.drag import IsotropicDrag, DragForce
from org.orekit.models.earth.atmosphere import DTM2000, HarrisPriester, JB2008, NRLMSISE00, Atmosphere, SimpleExponentialAtmosphere, PythonAtmosphere

from org.hipparchus.geometry.euclidean.threed import Vector3D

from netCDF4 import Dataset
from math import radians, degrees, pi
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import time
import os
from datetime import datetime, timedelta
import matplotlib.ticker as ticker
from pymsis import msis
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from skyfield.api import load, wgs84


vm = orekit.initVM()
setup_orekit_curdir()

plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})

# ...rest of your code...
def absolutedate_to_datetime(date: AbsoluteDate):
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

def eci_to_geodetic(earth, date: AbsoluteDate, position: Vector3D, frame: Frame):
    geodetic_point = earth.transform(position, frame, date)
    lat = degrees(geodetic_point.getLatitude())
    lon = degrees(geodetic_point.getLongitude())
    alt = geodetic_point.getAltitude()

    return lat, lon, alt

def get_msis_density(earth, date: AbsoluteDate, position: Vector3D, frame: Frame):

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
    file = get_file(date, 'density', 'wam_fixed_height')
    ncfid_i = Dataset(file)

    height_list = ncfid_i.variables['hlevs'][:]
    lat_list = ncfid_i.variables['lat'][:]
    lon_list = ncfid_i.variables['lon'][:]
    density_values = ncfid_i.variables['den']

    lat, lon, alt = eci_to_geodetic(earth, date, position, frame)
    alt = alt * 1e-3 # convert to km

    closest_height_index = np.abs(height_list - alt).argmin()
    closest_lat_index = np.abs(lat_list - lat).argmin()
    closest_lon_index = np.abs(lon_list - lon).argmin()

    density = density_values[0, closest_height_index, closest_lat_index, closest_lon_index].item()

    return density

def get_wamipe_wind_lla(date: AbsoluteDate, alt, lat, lon):
    file = get_file(date, 'fixed_height_wind', 'fixed_height_wind')
    ncfid_i = Dataset(file)

    height_list = ncfid_i.variables['height'][:]
    lat_list = ncfid_i.variables['lat'][:]
    lon_list = ncfid_i.variables['lon'][:]
    u_neutral_list = ncfid_i.variables['u_neutral']
    v_neutral_list = ncfid_i.variables['v_neutral']
    w_neutral_list = ncfid_i.variables['w_neutral']

    closest_height_index = np.abs(height_list - alt).argmin()
    closest_lat_index = np.abs(lat_list - lat).argmin()
    closest_lon_index = np.abs(lon_list - lon).argmin()

    u_neutral = u_neutral_list[closest_height_index, closest_lat_index, closest_lon_index].item()
    v_neutral = v_neutral_list[closest_height_index, closest_lat_index, closest_lon_index].item()
    w_neutral = w_neutral_list[closest_height_index, closest_lat_index, closest_lon_index].item()

    wind_velocity = Vector3D(u_neutral, v_neutral, w_neutral)

    return wind_velocity

def get_wamipe_wind(earth, date: AbsoluteDate, position: Vector3D, frame: Frame):
    file = get_file(date, 'fixed_height_wind', 'fixed_height_wind')
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

    wind_velocity = Vector3D(u_neutral, v_neutral, w_neutral)

    return wind_velocity

def plot_wind():
    # Set the date
    year = 2024
    month = 5
    # day = 10
    days = [10, 11, 12]
    hour = 0
    minute = 10
    second = 0
    for day in days:
        date = AbsoluteDate(year, month, day, hour, minute, float(second), TimeScalesFactory.getUTC())
    
        # Set the frame
        inertial_frame = FramesFactory.getEME2000()
        earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, inertial_frame)

        # Create a grid of latitudes and longitudes
        latitudes = np.linspace(-90, 90, 45)
        longitudes = np.linspace(0, 360, 45)
        u_wind = np.zeros((len(latitudes), len(longitudes)))
        v_wind = np.zeros((len(latitudes), len(longitudes)))
        w_wind = np.zeros((len(latitudes), len(longitudes)))
        altitude = 400e3

        for i, lat in enumerate(latitudes):
            for j, lon in enumerate(longitudes):

                # Get the wind velocity
                wind_velocity = get_wamipe_wind_lla(date, altitude, lat, lon) 
                u_wind[i, j] = wind_velocity.getX()
                v_wind[i, j] = wind_velocity.getY()
                w_wind[i, j] = wind_velocity.getZ()


        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        fig.suptitle('Neutral Wind Velocity at 400 km Altitude on {}-{}-{}'.format(year, month, day))

        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        axes[0].set_title('U-direction Wind Velocity')
        axes[0].grid(True)

        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        axes[1].set_title('V-direction Wind Velocity')
        axes[1].grid(True)

        axes[2].set_xlabel('Longitude')
        axes[2].set_ylabel('Latitude')
        axes[2].set_title('W-direction Wind Velocity')
        axes[2].grid(True)

        fig.subplots_adjust(right=0.9, hspace=0.5)
        # Increase the number of levels for smoother color transitions
        levels = np.linspace(-1800, 1800, 171)
        norm = plt.Normalize(vmin=-1800, vmax=1800)
        c1 = axes[0].contourf(longitudes, latitudes, u_wind, levels=levels, cmap='seismic', norm=norm)
        axes[1].contourf(longitudes, latitudes, v_wind, levels=levels, cmap='seismic', norm=norm)
        axes[2].contourf(longitudes, latitudes, w_wind, levels=levels, cmap='seismic', norm=norm)

        # Use the same colorbar for all plots
        cbar = fig.colorbar(c1, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        ticks = np.linspace(-1800, 1800, 7, dtype=int)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks)
        cbar.set_label('Wind Velocity (m/s)')
        
    plt.show()

def animate_wind():

    # Create a grid of latitudes and longitudes
    latitudes = np.linspace(-90, 90, 45)
    longitudes = np.linspace(0, 360, 45)
    altitude = 400e3

    # Initialize arrays to store wind data for animation
    u_wind_frames = []
    v_wind_frames = []
    w_wind_frames = []
    dates = []

    # Start and end dates
    start_date = datetime(2024, 5, 10, 0, 10, 0)
    # end_date = datetime(2024, 5, 10, 0, 20, 0)
    end_date = datetime(2024, 5, 14, 0, 0, 0)
    delta = timedelta(minutes=10)

    current_date = start_date
    while current_date <= end_date:
        date = AbsoluteDate(current_date.year, current_date.month, current_date.day, current_date.hour, current_date.minute, float(current_date.second), TimeScalesFactory.getUTC())
        current_date += delta
        u_wind = np.zeros((len(latitudes), len(longitudes)))
        v_wind = np.zeros((len(latitudes), len(longitudes)))
        w_wind = np.zeros((len(latitudes), len(longitudes)))

        for i, lat in enumerate(latitudes):
            for j, lon in enumerate(longitudes):
                # Get the wind velocity
                wind_velocity = get_wamipe_wind_lla(date, altitude, lat, lon)
                u_wind[i, j] = wind_velocity.getX()
                v_wind[i, j] = wind_velocity.getY()
                w_wind[i, j] = wind_velocity.getZ()

        u_wind_frames.append(u_wind)
        v_wind_frames.append(v_wind)
        w_wind_frames.append(w_wind)
        dates.append(current_date)

    # Create the figure and axes
    fig, axes = plt.subplots(3, 1, figsize=(8, 11))

    for ax in axes:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xticks(np.linspace(0, 360, 9))
        ax.set_yticks(np.linspace(-90, 90, 5))
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_aspect('equal')  # Ensures 1 degree lat = 1 degree lon in plot space
        ax.grid(False)

    axes[0].set_title('U-direction (Zonal) Wind Velocity')
    axes[1].set_title('V-direction (Meridional) Wind Velocity')
    axes[2].set_title('W-direction (Radial) Wind Velocity')


    fig.subplots_adjust(left=0.1, right=0.85, top=0.91, hspace=0.4)

    # Initialize the plots
    levels = np.linspace(-2000, 2000, 801)
    norm = plt.Normalize(vmin=-2000, vmax=2000)
    # Create a custom colormap with a gray center
    colors = [
        (0.0, "#000012"),  # Very dark blue
        (0.25, "#0000FF"),  # Blue
        (0.5, "#F7F7F3"),  # Very light yellow (almost white)
        (0.75, "#FF0000"),  # Red
        (1.0, "#120000"),  # Very dark red
    ]
    custom_cmap = LinearSegmentedColormap.from_list("custom_seismic", colors)

    # Use TwoSlopeNorm to center the colormap at 0
    norm = TwoSlopeNorm(vmin=-2000, vcenter=0, vmax=2000)
    # levels = np.linspace(-2000, 2000, 171)
    # norm = plt.Normalize(vmin=-2000, vmax=2000)
    # custom_cmap = plt.cm.twilight_shifted

    u_plot = axes[0].contourf(longitudes, latitudes, u_wind_frames[0], levels=levels, cmap=custom_cmap, norm=norm)
    v_plot = axes[1].contourf(longitudes, latitudes, v_wind_frames[0], levels=levels, cmap=custom_cmap, norm=norm)
    w_plot = axes[2].contourf(longitudes, latitudes, w_wind_frames[0], levels=levels, cmap=custom_cmap, norm=norm)

    def add_day_night_shading(ax, date):
        ts = load.timescale()
        planets = load('de421.bsp')
        earth = planets['earth']

        time = ts.utc(date.year, date.month, date.day, date.hour, date.minute, date.second)
        sun = planets['sun']
        sun_position = earth.at(time).observe(sun)
        subsolar_point = wgs84.subpoint(sun_position)  # Get the subsolar point

        subsolar_lat = subsolar_point.latitude.degrees
        subsolar_lon = subsolar_point.longitude.degrees

        # Calculate the terminator line
        lon = np.linspace(0, 360, 360)
        lat = np.arctan(-np.cos(np.radians(lon - subsolar_lon)) / np.tan(np.radians(subsolar_lat)))
        lat = np.degrees(lat)

        # Shade the nighttime regions
        ax.fill_between(lon, -90, lat, color='black', alpha=0.1)  

    # Add shading to show night and day time
    add_day_night_shading(axes[0], dates[0])
    add_day_night_shading(axes[1], dates[0])
    add_day_night_shading(axes[2], dates[0])

    # Use the same colorbar for all plots
    cbar = fig.colorbar(u_plot, ax=axes, orientation='vertical', fraction=0.025, pad=0.1)
    ticks = np.linspace(-2000, 2000, 5, dtype=int)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks)
    cbar.set_label('Wind Velocity (m/s)')

    # Update function for animation
    def update(frame):
        for c in axes[0].collections:
            c.remove()
        for c in axes[1].collections:
            c.remove()
        for c in axes[2].collections:
            c.remove()

        axes[0].contourf(longitudes, latitudes, u_wind_frames[frame], levels=levels, cmap=custom_cmap, norm=norm)
        axes[1].contourf(longitudes, latitudes, v_wind_frames[frame], levels=levels, cmap=custom_cmap, norm=norm)
        axes[2].contourf(longitudes, latitudes, w_wind_frames[frame], levels=levels, cmap=custom_cmap, norm=norm)

        # Add day-night shading to each subplot
        add_day_night_shading(axes[0], dates[frame])
        add_day_night_shading(axes[1], dates[frame])
        add_day_night_shading(axes[2], dates[frame])
    
        fig.suptitle(f'Neutral Wind Velocity at 400 km Altitude on {dates[frame]}', fontsize=18)
        return

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(dates), repeat=False)
    ani.save('figures/neutral_wind_animation.mp4', writer='ffmpeg', fps=25)
    # Show the animation
    plt.show()
    return

def animate_wind_compare_altitudes():
    # Create a grid of latitudes and longitudes
    latitudes = np.linspace(-90, 90, 45)
    longitudes = np.linspace(0, 360, 45)
    altitudes = [400e3, 600e3]

    # Initialize arrays to store wind data for animation
    wind_frames = {alt: {'u': [], 'v': [], 'w': []} for alt in altitudes}
    dates = []

    # Start and end dates
    start_date = datetime(2024, 5, 10, 0, 10, 0)
    # end_date = datetime(2024, 5, 10, 0, 30, 0)
    end_date = datetime(2024, 5, 14, 0, 0, 0)
    delta = timedelta(minutes=10)

    current_date = start_date
    while current_date <= end_date:
        date = AbsoluteDate(current_date.year, current_date.month, current_date.day, current_date.hour, current_date.minute, float(current_date.second), TimeScalesFactory.getUTC())
        current_date += delta
        for alt in altitudes:
            u_wind = np.zeros((len(latitudes), len(longitudes)))
            v_wind = np.zeros((len(latitudes), len(longitudes)))
            w_wind = np.zeros((len(latitudes), len(longitudes)))
            for i, lat in enumerate(latitudes):
                for j, lon in enumerate(longitudes):
                    wind_velocity = get_wamipe_wind_lla(date, alt, lat, lon)
                    u_wind[i, j] = wind_velocity.getX()
                    v_wind[i, j] = wind_velocity.getY()
                    w_wind[i, j] = wind_velocity.getZ()
            wind_frames[alt]['u'].append(u_wind)
            wind_frames[alt]['v'].append(v_wind)
            wind_frames[alt]['w'].append(w_wind)
        dates.append(current_date)

    # Create the figure and axes: 2 columns (altitudes), 3 rows (wind components)
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig.suptitle('Neutral Wind Velocity Comparison at 400 km and 600 km Altitudes')

    # Add column headers for 400 km and 600 km
    axes[0, 0].annotate("400 km Altitude", xy=(0.5, 1.18), xycoords='axes fraction', ha='center', va='bottom', fontsize=24, fontweight='bold')
    axes[0, 1].annotate("600 km Altitude", xy=(0.5, 1.18), xycoords='axes fraction', ha='center', va='bottom', fontsize=24, fontweight='bold')

    for col, alt in enumerate(altitudes):
        for row in range(3):
            ax = axes[row, col]
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_xticks(np.linspace(0, 360, 9))
            ax.set_yticks(np.linspace(-90, 90, 5))
            ax.set_xlim(0, 360)
            ax.set_ylim(-90, 90)
            ax.set_aspect('equal')
            ax.grid(False)
            if row == 0:
                ax.set_title(f'Zonal Wind')
            elif row == 1:
                ax.set_title(f'Meridional Wind')
            else:
                ax.set_title(f'Radial Wind')

    fig.subplots_adjust(left=0.07, right=0.92, top=0.86, hspace=0.5, wspace=0.2)

    levels = np.linspace(-2000, 2000, 801)
    colors = [
        (0.0, "#000012"),
        (0.25, "#0000FF"),
        (0.5, "#F7F7F3"),
        (0.75, "#FF0000"),
        (1.0, "#120000"),
    ]
    custom_cmap = LinearSegmentedColormap.from_list("custom_seismic", colors)
    norm = TwoSlopeNorm(vmin=-2000, vcenter=0, vmax=2000)

    # Initial plots
    plots = []
    for col, alt in enumerate(altitudes):
        plots.append([
            axes[0, col].contourf(longitudes, latitudes, wind_frames[alt]['u'][0], levels=levels, cmap=custom_cmap, norm=norm),
            axes[1, col].contourf(longitudes, latitudes, wind_frames[alt]['v'][0], levels=levels, cmap=custom_cmap, norm=norm),
            axes[2, col].contourf(longitudes, latitudes, wind_frames[alt]['w'][0], levels=levels, cmap=custom_cmap, norm=norm)
        ])

    def add_day_night_shading(ax, date):
        ts = load.timescale()
        planets = load('de421.bsp')
        earth = planets['earth']
        time = ts.utc(date.year, date.month, date.day, date.hour, date.minute, date.second)
        sun = planets['sun']
        sun_position = earth.at(time).observe(sun)
        subsolar_point = wgs84.subpoint(sun_position)
        subsolar_lat = subsolar_point.latitude.degrees
        subsolar_lon = subsolar_point.longitude.degrees
        lon = np.linspace(0, 360, 360)
        lat = np.arctan(-np.cos(np.radians(lon - subsolar_lon)) / np.tan(np.radians(subsolar_lat)))
        lat = np.degrees(lat)
        ax.fill_between(lon, -90, lat, color='black', alpha=0.1)

    for col in range(2):
        for row in range(3):
            add_day_night_shading(axes[row, col], dates[0])

    # Shared colorbar
    cbar = fig.colorbar(plots[0][0], ax=axes, orientation='vertical', fraction=0.025, pad=0.04)
    ticks = np.linspace(-2000, 2000, 5, dtype=int)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks)
    cbar.set_label('Wind Velocity (m/s)')

    def update(frame):
        for col, alt in enumerate(altitudes):
            for row, comp in enumerate(['u', 'v', 'w']):
                ax = axes[row, col]
                for c in ax.collections:
                    c.remove()
                ax.contourf(longitudes, latitudes, wind_frames[alt][comp][frame], levels=levels, cmap=custom_cmap, norm=norm)
                add_day_night_shading(ax, dates[frame])
        fig.suptitle(f'Neutral Wind Velocity at 400 km and 600 km Altitudes on {dates[frame]}', fontsize=24)
        return

    ani = animation.FuncAnimation(fig, update, frames=len(dates), repeat=False)
    ani.save('figures/neutral_wind_compare_altitudes.mp4', writer='ffmpeg', fps=25)
    plt.show()
    return

# compare the density values from MSIS and WAM-IPE
def compare_densities():
    # Set the date
    year = 2024
    month = 5
    day = 10
    hour = 0
    minute = 10
    second = 0
    start_date = datetime(year, month, 10, hour, minute, second)
    end_date = datetime(year, month, 12, hour, minute, second)
    delta = timedelta(minutes=10)

    dates = []
    msis_densities = []
    wamipe_densities = []

    current_date = start_date
    while current_date <= end_date:
        try:
            date = AbsoluteDate(current_date.year, current_date.month, current_date.day, current_date.hour, current_date.minute, float(current_date.second), TimeScalesFactory.getUTC())
            dates.append(current_date)

            # Set the position of the spacecraft
            x = 400.0e3 + Constants.WGS84_EARTH_EQUATORIAL_RADIUS
            y = 0.0
            z = 0.0
            position = Vector3D(x, y, z)

            # Set the frame
            inertial_frame = FramesFactory.getEME2000()
            earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, inertial_frame)

            # Get the density values
            msis_density = get_msis_density(earth, date, position, inertial_frame)
            wamipe_density = get_wamipe_density(earth, date, position, inertial_frame)

            msis_densities.append(msis_density)
            wamipe_densities.append(wamipe_density)
        except Exception as e:
            print(f"Error processing date {current_date}: {e}")

        current_date += delta

    # Plot the densities
    plt.figure(figsize=(10, 5))
    plt.plot(dates, msis_densities, label='MSIS Density')
    plt.plot(dates, wamipe_densities, label='WAM-IPE Density')
    plt.xlabel('Date')
    plt.ylabel('Density (kg/m^3)')
    plt.title('Density Comparison from March 10 to March 13')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# compare_densities()
# plot_wind()
animate_wind()
# animate_wind_compare_altitudes()