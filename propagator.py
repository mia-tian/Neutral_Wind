# Initialize orekit and JVM
import orekit
from orekit.pyhelpers import setup_orekit_curdir

from org.orekit.frames import FramesFactory
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants, PVCoordinates
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator, ClassicalRungeKuttaIntegrator
from org.orekit.propagation import SpacecraftState
from org.orekit.bodies import OneAxisEllipsoid, CelestialBodyFactory, CelestialBody
from org.orekit.utils import IERSConventions
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction, OceanTides, SolidTides
from java.util import ArrayList

from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient, RadiationSensitive
from org.orekit.models.earth.atmosphere.data import CssiSpaceWeatherData, JB2008SpaceEnvironmentData
from org.orekit.forces.drag import IsotropicDrag, DragForce


import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.ticker as ticker
from tqdm import tqdm
import datetime

from orekit_helpers import absolutedate_to_datetime, eci2ric



vm = orekit.initVM()
setup_orekit_curdir()

plt.rcParams.update({'font.size': 12, 'font.family': 'Arial'})

# =============================================================================
# Constants/Basics
# =============================================================================
r_Earth = Constants.IERS2010_EARTH_EQUATORIAL_RADIUS #m
ecef    = FramesFactory.getITRF(IERSConventions.IERS_2010, True) # International Terrestrial Reference Frame, earth fixed
inertialFrame = FramesFactory.getEME2000()
earth = OneAxisEllipsoid(r_Earth,
                         Constants.IERS2010_EARTH_FLATTENING,
                         ecef)
mu = Constants.IERS2010_EARTH_MU #m^3/s^2
utc = TimeScalesFactory.getUTC()
sun = CelestialBodyFactory.getSun()
moon = CelestialBodyFactory.getMoon()
deg = np.pi / 180 # Degrees-Radians conversion

def prop_orbit(initial_orbit, duration, CustomAtmosphereClass, plot=True, **kwargs):
    """
    Propagates the orbit of a satellite over a given duration using a high-fidelity numerical propagator.
    Parameters:
    initial_orbit (Orbit): The initial orbit of the satellite.
    duration (float): The duration for which to propagate the orbit, in seconds.
    CustomAtmosphereClass (class): A custom atmosphere model class to be used for drag force calculations.
    Returns:
    tuple: A tuple containing:
        - states (list of SpacecraftState): The list of spacecraft states at each propagation step.
    """

    # Satellite Parameters
    satellite_mass = kwargs.get('satellite_mass', 260.0) 
    crossSection = kwargs.get('cross_section', 3.0) # In m^2
    srpArea = kwargs.get('srp_area', 5.0) # In m^2
    dragCoeff = kwargs.get('drag_coeff', 2.2)

    # Perturbation Parameters
    degree = kwargs.get('degree', 70) # Degree of the gravity field
    torder = kwargs.get('torder', 70) # Order of the gravity field
    cr = 1.0

    # Propagation time steps calculation
    initialDate = initial_orbit.getDate()
    output_step = kwargs.get('output_step', 45.0) # Step in seconds 
    tspan = [initialDate.shiftedBy(float(dt)) for dt in np.linspace(0, duration, int(duration / output_step))]

    # Integrator settings
    minStep = 1e-6
    maxstep = 1e4
    initStep = 1.0
    fixed_step = 60.0
    positionTolerance = 1e-4

    sun = CelestialBodyFactory.getSun()
    moon = CelestialBodyFactory.getMoon()

    satmodel = IsotropicDrag(crossSection, dragCoeff) # Cross sectional area and the drag coefficient

    initialOrbit = initial_orbit
    orbitType = initialOrbit.getType()
    initialState = SpacecraftState(initialOrbit, satellite_mass)
    tol = NumericalPropagator.tolerances(positionTolerance, initialOrbit, orbitType)

    # integrator = DormandPrince853Integrator(minStep, maxstep, JArray_double.cast_(tol[0]), JArray_double.cast_(tol[1]))
    # integrator.setInitialStepSize(initStep)
    integrator = ClassicalRungeKuttaIntegrator(fixed_step)

    propagator_num = NumericalPropagator(integrator)
    propagator_num.setOrbitType(orbitType)
    propagator_num.setInitialState(initialState)

    # Add Solar Radiation Pressure
    if kwargs.get('srp', True):
        spacecraft = IsotropicRadiationSingleCoefficient(srpArea, cr)
        srpProvider = SolarRadiationPressure(sun, earth, spacecraft)
        propagator_num.addForceModel(srpProvider)

    # Add Gravity Force
    gravityProvider = GravityFieldFactory.getConstantNormalizedProvider(degree, torder, initialDate)
    gravityForce = HolmesFeatherstoneAttractionModel(earth.getBodyFrame(), gravityProvider)
    propagator_num.addForceModel(gravityForce)

    # Add Solid Tides
    if kwargs.get('solid_tides', True):
        solidTidesBodies = ArrayList().of_(CelestialBody)
        solidTidesBodies.add(sun)
        solidTidesBodies.add(moon)
        solidTidesBodies = solidTidesBodies.toArray()
        solidTides = SolidTides(earth.getBodyFrame(), 
                                gravityProvider.getAe(), gravityProvider.getMu(),
                                gravityProvider.getTideSystem(), 
                                IERSConventions.IERS_2010,
                                TimeScalesFactory.getUT1(IERSConventions.IERS_2010, True), 
                                solidTidesBodies)
        propagator_num.addForceModel(solidTides)

    # Add Third Body Attractions
    if kwargs.get('third_body_attraction', True):
        propagator_num.addForceModel(ThirdBodyAttraction(sun))
        propagator_num.addForceModel(ThirdBodyAttraction(moon)) 

    # Add Custom Drag Force
    if kwargs.get('drag', True):
        # cswl = JB2008SpaceEnvironmentData("SOLFSMY.TXT", "DTCFILE.TXT")
        cswl = None
        atmosphere = CustomAtmosphereClass(cswl, sun, earth)
        dragForce = DragForce(atmosphere, satmodel)
        propagator_num.addForceModel(dragForce)

    # Add NRLMSIS Drag Force
    # nrlmsis_parameters = NRLMSISE00InputParameters()  # Assuming you have a way to initialize this
    # nrlmsis_atmosphere = NRLMSISE00(nrlmsis_parameters, sun, earth)
    # nrlmsis_dragForce = DragForce(nrlmsis_atmosphere, satmodel)
    # propagator_num.addForceModel(nrlmsis_dragForce)

    states = [initialState]
    tic = time.time()
    progress_bar = tqdm(total=len(tspan) - 1, desc="Propagating Orbit", unit="step")
    for i1 in range(len(tspan) - 1):
        states.append(propagator_num.propagate(tspan[i1], tspan[i1 + 1]))
        progress_bar.update(1)  # Update progress bar
    toc = time.time()
    progress_bar.close()  # Close progress bar
    print('Propagation time:', toc - tic, 'seconds')

    if (plot):
        

        # Retrieve wind vectors from the propagator's force models if available
        wind = []
        dates = []
        for state in states:
            pv = state.getPVCoordinates()
            position = pv.getPosition()
            date = state.getDate()
            frame = state.getFrame()
            velocity = pv.getVelocity()
            wind_eci = atmosphere.getVelocity(date, position, frame)

            # Convert wind_eci to ECEF coordinates
            eci_to_ecef = frame.getKinematicTransformTo(ecef, date)
            pv_ecef = eci_to_ecef.transformOnlyPV(pv)
            pos_ecef = pv_ecef.getPosition()
            vel_ecef = pv_ecef.getVelocity()
            pv_wind = PVCoordinates(position, wind_eci)
            wind_ecef = eci_to_ecef.transformOnlyPV(pv_wind).getVelocity()


            ######## Convert wind vector to RIC frame ########
            # compute the radial unit vector
            r_unit = pos_ecef.toArray() / np.linalg.norm(pos_ecef.toArray())

            # compute the in-track unit vector aligned with the velocity of the reference state
            i_unit = vel_ecef.toArray() / np.linalg.norm(vel_ecef.toArray())

            # compute the cross-track unit vector
            c_unit = np.cross(r_unit, i_unit)

            # compute the state vector in the RTN frame
            wind_ric = np.array([np.dot(r_unit, wind_ecef.toArray()), np.dot(i_unit, wind_ecef.toArray()), np.dot(c_unit, wind_ecef.toArray())])
            ###################################################

            wind.append(wind_ric)
            dates.append(absolutedate_to_datetime(date))

        wind = np.array(wind)
        dates = np.array(dates)

        # compute running average of wind over a time window (minutes)
        running_minutes = kwargs.get('running_average_minutes', 90)
        window_size = max(1, int(round(running_minutes * 60.0 / output_step))) # number of output steps
        kernel = np.ones(window_size) / window_size
        wind_avg = np.vstack([np.convolve(wind[:, i], kernel, mode='same') for i in range(wind.shape[1])]).T
        avg_label = f'{running_minutes:g}-min Running Average'


        fig, ax = plt.subplots(3, 1, figsize=(10, 8))

        gray_color = "#e0e0e0b0"
        line_color = "#ff7878"
        avg_color = "#A83B3B"

        ax[0].axhline(0, color=gray_color, linestyle='-', linewidth=1, zorder=0)
        ax[0].plot(dates, wind[:, 0], label='Radial', linewidth=1, color=line_color, alpha=0.6)
        ax[0].plot(dates, wind_avg[:, 0], label=avg_label, linewidth=1, color=avg_color, alpha=0.6)
        ax[0].set_ylabel('Radial [m/s]', fontsize=14)
        ax[0].legend()

        ax[1].axhline(0, color=gray_color, linestyle='-', linewidth=1, zorder=0)
        ax[1].plot(dates, wind[:, 1], label='In-Track', linewidth=1, color=line_color, alpha=0.6)
        ax[1].plot(dates, wind_avg[:, 1], label=avg_label, linewidth=1, color=avg_color, alpha=0.6)
        ax[1].set_ylabel('In-Track [m/s]', fontsize=14)
        ax[1].legend()

        ax[2].axhline(0, color=gray_color, linestyle='-', linewidth=1, zorder=0)
        ax[2].plot(dates, wind[:, 2], label='Cross-Track', linewidth=1, color=line_color, alpha=0.6)
        ax[2].plot(dates, wind_avg[:, 2], label=avg_label, linewidth=1, color=avg_color, alpha=0.6)
        ax[2].set_ylabel('Cross-Track [m/s]', fontsize=14)
        ax[2].set_xlabel('Time [days]', fontsize=14)
        ax[2].legend()
        
        for axis in ax:
            axis.set_ylim(-750, 750)
            axis.set_yticks([-500, 0, 500])
            start_tick = datetime.datetime(2024, 5, 10, 0, 0, 0)
            end_tick = datetime.datetime(2024, 5, 14, 0, 0, 0)
            tick_dates = [start_tick + datetime.timedelta(days=i) 
                          for i in range((end_tick - start_tick).days + 1)]
            axis.set_xticks(tick_dates)
            axis.xaxis.set_major_formatter(ticker.FixedFormatter(
                [dt.strftime("%-m/%-d") for dt in tick_dates]
            ))
            plt.setp(axis.xaxis.get_majorticklabels(), rotation=45, ha='right')
            

        fig.suptitle('Wind Velocity in RIC Coordinates', fontsize=18)
        plt.tight_layout()
        plt.savefig('figures/wind_vectors.png', dpi=300)
        plt.show()
        plt.close()

    
    return states