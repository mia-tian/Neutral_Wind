# Initialize orekit and JVM
import orekit
from orekit.pyhelpers import setup_orekit_curdir

from org.orekit.orbits import KeplerianOrbit, EquinoctialOrbit, PositionAngleType, OrbitType, CartesianOrbit
from org.orekit.frames import FramesFactory
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants
from org.orekit.bodies import OneAxisEllipsoid
from org.orekit.utils import IERSConventions

from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime
import os
import traceback

from orekit_helpers import absolutedate_to_datetime, eci2ric
from propagator import prop_orbit


vm = orekit.initVM()
setup_orekit_curdir()

# Import atmosphere modules after the JVM is initialized to avoid loading
# conflicting native libraries (e.g. HDF5) before the JVM starts. Loading
# HDF5 (via netCDF4/h5py) before the JVM can cause SIGSEGVs in libhdf5
# when both Python and the JVM load different HDF5 versions into the same
# process. Moving this import after initVM reduces that risk.
from atmospheres import CustomAtmosphere, WindyCustomAtmosphere


plt.rcParams.update({'font.size': 12, 'font.family': 'Arial'})


# =============================================================================
#% Constants/Basics
# =============================================================================
r_Earth = Constants.IERS2010_EARTH_EQUATORIAL_RADIUS #m
ecef    = FramesFactory.getITRF(IERSConventions.IERS_2010, True) # International Terrestrial Reference Frame, earth fixed
inertialFrame = FramesFactory.getEME2000()
earth = OneAxisEllipsoid(r_Earth,
                         Constants.IERS2010_EARTH_FLATTENING,
                         ecef)
mu = Constants.IERS2010_EARTH_MU #m^3/s^2

utc = TimeScalesFactory.getUTC()

deg_to_rad = pi / 180

def main():
    # analyze()
    analyze_multiple_orbits()

def analyze():

    rp0 = r_Earth + 400 * 1e3 # perigee radius (m)
    ra0 = r_Earth + 400 * 1e3 # apogee radius (m)

    a0 = (rp0 + ra0) / 2
    e0 = (ra0 - rp0) / (ra0 + rp0)
    w0 = 0.000000 * deg_to_rad # argument of perigee
    i0 = 0 * deg_to_rad # inclination
    ra0 = 0 * deg_to_rad # right ascension of ascending node
    M0 = 0 * deg_to_rad # true anomaly
    initialOrbit_keplerian = KeplerianOrbit(a0, e0, i0, w0, ra0, M0, PositionAngleType.TRUE, inertialFrame, AbsoluteDate(2024, 5, 10, 0, 10, 00.000, TimeScalesFactory.getUTC()), mu)
    initialOrbit = EquinoctialOrbit(initialOrbit_keplerian)
    duration = 3.99 * 86400.0 # 4 day in seconds
    # duration = .5 * 86400.0 # some amount of days in seconds

    output_step = 45.0 # seconds


    print("Atmosphere")
    # states = prop_orbit(initialOrbit, duration, CustomAtmosphere, degree=2, torder=2,srp= False,solid_tides=False, third_body_attraction=False, plot=False, output_step=300.0)
    states = prop_orbit(initialOrbit, duration, CustomAtmosphere, plot=False, output_step=output_step)
    posvel = [state.getPVCoordinates() for state in states]
    poss = [state.getPosition() for state in posvel]
    vels = [state.getVelocity() for state in posvel]
    px = [pos.getX() * 1e-3 for pos in poss]
    py = [pos.getY() * 1e-3 for pos in poss]
    pz = [pos.getZ() * 1e-3 for pos in poss]
    vx = [vel.getX() * 1e-3 for vel in vels]
    vy = [vel.getY() * 1e-3 for vel in vels]
    vz = [vel.getZ() * 1e-3 for vel in vels]
    print("Final Pos [km]:", np.linalg.norm([px[-1], py[-1], pz[-1]]))
    print("Final Pos [km]:", px[-1], py[-1], pz[-1])
    print("Final Vel [km]:", np.linalg.norm([vx[-1], vy[-1], vz[-1]]))

    total_distance = 0.0
    for i in range(1, len(poss)):
        total_distance += poss[i].distance(poss[i - 1])

    print("Total Distance Traveled [km]:", total_distance * 1e-3)


    print()

    print("Windy Atmosphere")
    # states_windy = prop_orbit(initialOrbit, duration, WindyCustomAtmosphere, degree=2, torder=2,srp= False,solid_tides=False, third_body_attraction=False, plot=False, output_step=300.0)
    states_windy = prop_orbit(initialOrbit, duration, WindyCustomAtmosphere, output_step=output_step)
    posvel_w = [state.getPVCoordinates() for state in states_windy]
    poss_w = [state.getPosition() for state in posvel_w]
    vels_w = [state.getVelocity() for state in posvel_w]
    px_w = [pos.getX() * 1e-3 for pos in poss_w]
    py_w = [pos.getY() * 1e-3 for pos in poss_w]
    pz_w = [pos.getZ() * 1e-3 for pos in poss_w]
    vx_w = [vel.getX() * 1e-3 for vel in vels_w]
    vy_w = [vel.getY() * 1e-3 for vel in vels_w]
    vz_w = [vel.getZ() * 1e-3 for vel in vels_w]
    print("Final Pos [km]:", np.linalg.norm([px_w[-1], py_w[-1], pz_w[-1]]))
    print("Final Pos [km]:", px_w[-1], py_w[-1], pz_w[-1])
    print("Final Vel [km]:", np.linalg.norm([vx_w[-1], vy_w[-1], vz_w[-1]]))

    total_distance = 0.0
    for i in range(1, len(poss_w)):
        total_distance += poss_w[i].distance(poss_w[i - 1])

    print("Total Distance Traveled [km]:", total_distance * 1e-3)

    print()

    print('Final ECI Pos Difference [km]', px_w[-1]-px[-1], py_w[-1]-py[-1], pz_w[-1]-pz[-1])

    # Convert the states to RIC coordinates
    ric_states = []
    for i, state in enumerate(states):
        windy_state = states_windy[i]
        x0 = np.concatenate((state.getPVCoordinates().getPosition().toArray(), state.getPVCoordinates().getVelocity().toArray()))
        x = np.concatenate((windy_state.getPVCoordinates().getPosition().toArray(), windy_state.getPVCoordinates().getVelocity().toArray()))
        ric_state = eci2ric(x0, x)
        ric_states.append(ric_state)


    # Print the final RIC coordinates
    print("Final RIC Pos Diff [km]:", ric_states[-1][:3])
    print("Final RIC Vel Diff [km/s]:", ric_states[-1][3:])

    # Plot the RIC coordinates
    ric_states = np.array(ric_states)
    dates = [state.getDate() for state in states]
    dates = [absolutedate_to_datetime(date).strftime("%m/%d, %H:%M") for date in dates]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Define colors and labels for RIC components
    ric_colors = {
        'Radial': "#92ecee",
        'In-Track': "#6bb2ff",
        'Cross-Track': "#BD95F1"
    }
    ric_labels = ['Radial', 'In-Track', 'Cross-Track']

    gray_color = "#e0e0e0b0"

    # Plot position
    ax1.axhline(0, color=gray_color, linestyle='-', linewidth=1)
    for i, label in enumerate(ric_labels):
        ax1.plot(dates, ric_states[:, i] * 1e-3, label=label, linewidth=2, color=ric_colors[label])
    ax1.set_ylabel('Position [km]', fontsize=14)
    ax1.legend(fontsize=14)

    # Plot velocity
    ax2.axhline(0, color=gray_color, linestyle='-', linewidth=1)
    for i, label in enumerate(ric_labels):
        ax2.plot(dates, ric_states[:, i+3] * 1e-3, label=label, linewidth=2, color=ric_colors[label])
    ax2.set_ylabel('Velocity [km/s]', fontsize=14)
    ax2.set_xlabel('Time [days]', fontsize=14)
    ax2.legend(fontsize=14)

    # Set x-axis tick labels to show only some of the dates
    for axis in (ax1, ax2):
        # Set ticks at the beginning of each day from 5/10 to 5/14 (no hours/minutes)
        start_tick = datetime.datetime(2024, 5, 10, 0, 0, 0)
        end_tick = datetime.datetime(2024, 5, 14, 0, 0, 0)
        tick_dates = [start_tick + datetime.timedelta(days=i)
                        for i in range((end_tick - start_tick).days + 1)]
        tick_labels = [dt.strftime("%-m/%-d") for dt in tick_dates]

        # The plotted x-values are strings in `dates` (e.g. '05/10, 00:10'),
        # so build integer tick positions by finding the first index for each day.
        tick_positions = []
        for dt in tick_dates:
            dt_str = dt.strftime("%m/%d")  # zero-padded to match beginning of strings in `dates`
            pos = next((i for i, s in enumerate(dates) if s.startswith(dt_str)), None)
            if pos is not None:
                tick_positions.append(pos)

        if tick_positions:
            axis.set_xticks(tick_positions)
            axis.xaxis.set_major_formatter(ticker.FixedFormatter(tick_labels[:len(tick_positions)]))
        plt.setp(axis.xaxis.get_majorticklabels(), rotation=45, ha='right')

    fig.suptitle('Impact of Neutral Wind on Satellite Orbit', fontsize=18)

    plt.tight_layout()
    plt.savefig('figures/wind_effect_ric.png', dpi=300)
    plt.show()
    plt.close()

    # Find the propagated state closest to epoch 2024-05-10 21:20:00 UTC and compute true anomaly
    target_date = AbsoluteDate(2024, 5, 10, 21, 20, 0.000, utc)

    # find index of closest state in `states`
    idx_closest = min(range(len(states)), key=lambda i: abs(states[i].getDate().durationFrom(target_date)))
    closest_state = states[idx_closest]

    # optionally check time difference
    time_diff_s = target_date.durationFrom(closest_state.getDate())
    print("Closest state time:", absolutedate_to_datetime(closest_state.getDate()).strftime("%Y-%m-%d %H:%M:%S"), "UTC")
    print("Time difference to target (s):", time_diff_s)

    # get position (m) and print (km)
    pv = closest_state.getPVCoordinates()
    pos = pv.getPosition()
    print("Position at epoch [km]:", pos.getX() * 1e-3, pos.getY() * 1e-3, pos.getZ() * 1e-3)

    # build a Keplerian orbit from the state's PV and compute true anomaly
    kepler_from_state = KeplerianOrbit(pv, inertialFrame, closest_state.getDate(), mu)
    true_anom_rad = kepler_from_state.getTrueAnomaly()
    true_anom_deg = (true_anom_rad * 180.0 / pi) % 360.0

    print("True anomaly at", absolutedate_to_datetime(target_date).strftime("%Y-%m-%d %H:%M:%S"), "UTC:", f"{true_anom_deg:.6f} deg")

    # compute geodetic latitude, longitude and altitude from ECEF position (pos in meters)
    x = pos.getX()
    y = pos.getY()
    z = pos.getZ()

    p = np.hypot(x, y)
    a = r_Earth
    f = Constants.IERS2010_EARTH_FLATTENING
    e2 = 2.0 * f - f * f

    lat = np.arctan2(z, p * (1.0 - f))
    alt = 0.0
    for _ in range(10):
        N = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - N
        lat_new = np.arctan2(z, p * (1.0 - e2 * N / (N + alt)))
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new

    lon = np.arctan2(y, x)

    lat_deg = lat * 180.0 / pi
    lon_deg = lon * 180.0 / pi

    print("Geodetic latitude (deg):", f"{lat_deg:.6f}")
    print("Geodetic longitude (deg):", f"{lon_deg:.6f}")
    print("Ellipsoidal altitude (m):", f"{alt:.3f}")

    return

def analyze_multiple_orbits():
    # parameter grids (examples — adjust as needed)
    alt_km_vals = [400.0]          # altitude above mean equator radius (km)
    e_vals = [0.0]                  # eccentricity
    i_deg_vals = [0.0]               # inclination (deg)
    raan_deg_vals = [0.0]            # RAAN (deg)
    argp_deg_vals = [0.0]                # argument of perigee (deg)
    M_deg_vals = np.arange(0.0, 360.0, 5.0)  # true anomaly (deg)

    # propagation settings (tunable)
    duration = 3.99 * 86400.0
    output_step = 300.0 # seconds

    # result file (ensure directory exists)
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "multiple_orbit_runs.txt")

    # open file for append so partial results are preserved on error; flush+fsync each write
    with open(results_path, "w", encoding="utf-8") as fout:
        def flush_to_disk():
            try:
                fout.flush()
                os.fsync(fout.fileno())
            except Exception:
                pass

        # header
        header = (
            "Run timestamp: {}\n"
            "Columns: altitude_km, ecc, inc_deg, raan_deg, argp_deg, trueanom_deg | "
            "RIC_pos_km( R,I,C ), RIC_vel_kmps( R,I,C ) | status\n"
        ).format(datetime.datetime.utcnow().isoformat())
        fout.write(header)
        flush_to_disk()

        run_index = 0
        for alt_km in alt_km_vals:
            for ecc in e_vals:
                for inc_deg in i_deg_vals:
                    for raan_deg in raan_deg_vals:
                        for argp_deg in argp_deg_vals:
                            for M_deg in M_deg_vals:
                                run_index += 1
                                try:
                                    # print orbital elements for this run
                                    print(
                                        f"RUN {run_index}: alt_km={alt_km}, ecc={ecc}, inc_deg={inc_deg}, "
                                        f"raan_deg={raan_deg}, argp_deg={argp_deg}, M_deg={M_deg}"
                                    )
                                    # build initial Keplerian orbit for this run
                                    a_m = (r_Earth + alt_km * 1e3)  # semi-major ~ radius for circular-like cases
                                    inc_rad = inc_deg * deg_to_rad
                                    raan_rad = raan_deg * deg_to_rad
                                    argp_rad = argp_deg * deg_to_rad
                                    M_rad = M_deg * deg_to_rad

                                    epoch = AbsoluteDate(2024, 5, 10, 0, 10, 00.000, utc)

                                    # Ensure native Python floats are passed to the Java wrapper
                                    ko = KeplerianOrbit(float(a_m), float(ecc), float(inc_rad), float(argp_rad), float(raan_rad), float(M_rad),
                                                            PositionAngleType.TRUE, inertialFrame, epoch, float(mu))
                                    initial_orbit = EquinoctialOrbit(ko)

                                    # propagate in baseline atmosphere
                                    try:
                                        states_base = prop_orbit(initial_orbit, duration, CustomAtmosphere, degree=2, torder=2,srp= False, solid_tides=False, third_body_attraction=False, plot=False, output_step=300.0)
                                        if not states_base:
                                            raise RuntimeError("prop_orbit returned empty result for baseline atmosphere")
                                    except Exception as e_base:
                                        # log error for baseline and skip to next combination
                                        msg = (
                                            f"RUN {run_index}: alt_km={alt_km}, ecc={ecc}, inc_deg={inc_deg}, "
                                            f"raan_deg={raan_deg}, argp_deg={argp_deg}, M_deg={M_deg} | "
                                            f"ERROR during baseline propagation: {repr(e_base)}\n"
                                            f"{traceback.format_exc()}\n"
                                        )
                                        fout.write(msg)
                                        flush_to_disk()
                                        continue

                                    # propagate in windy atmosphere
                                    try:
                                        states_windy = prop_orbit(initial_orbit, duration, WindyCustomAtmosphere, degree=2, torder=2,srp= False,solid_tides=False, third_body_attraction=False, plot=False, output_step=300.0)
                                        if not states_windy:
                                            raise RuntimeError("prop_orbit returned empty result for windy atmosphere")
                                    except Exception as e_w:
                                        msg = (
                                            f"RUN {run_index}: alt_km={alt_km}, ecc={ecc}, inc_deg={inc_deg}, "
                                            f"raan_deg={raan_deg}, argp_deg={argp_deg}, M_deg={M_deg} | "
                                            f"ERROR during windy propagation: {repr(e_w)}\n"
                                            f"{traceback.format_exc()}\n"
                                        )
                                        fout.write(msg)
                                        flush_to_disk()
                                        continue

                                    # extract final states (PVCoordinates)
                                    final_base = states_base[-1].getPVCoordinates()
                                    final_windy = states_windy[-1].getPVCoordinates()

                                    # compute RIC difference between final states (ECI -> RIC)
                                    x0 = (
                                        final_base.getPosition().toArray()
                                        + final_base.getVelocity().toArray()
                                    )
                                    x = (
                                        final_windy.getPosition().toArray()
                                        + final_windy.getVelocity().toArray()
                                    )
                                    ric = eci2ric(np.array(x0), np.array(x))  # units: meters and m/s
                                    ric_pos_km = ric[:3] * 1e-3
                                    ric_vel_kmps = ric[3:] * 1e-3

                                    # print ric pos for this run
                                    print(
                                        f"RUN {run_index} RIC Pos Diff [km]: {ric_pos_km[0]:.6f}, {ric_pos_km[1]:.6f}, {ric_pos_km[2]:.6f}"
                                    )
                                    print()

                                    # write a single-line summary: starting elements + RIC deviations
                                    line = (
                                        f"RUN {run_index}: alt_km={alt_km}, ecc={ecc}, inc_deg={inc_deg}, "
                                        f"raan_deg={raan_deg}, argp_deg={argp_deg}, M_deg={M_deg} | "
                                        f"RIC_pos_km=({ric_pos_km[0]:.6f},{ric_pos_km[1]:.6f},{ric_pos_km[2]:.6f}) | "
                                        f"RIC_vel_kmps=({ric_vel_kmps[0]:.6e},{ric_vel_kmps[1]:.6e},{ric_vel_kmps[2]:.6e}) | "
                                        f"STATUS=OK\n"
                                    )
                                    fout.write(line)
                                    flush_to_disk()

                                except Exception as e_general:
                                    # unexpected error for this combination: log and continue
                                    msg = (
                                        f"RUN {run_index}: alt_km={alt_km}, ecc={ecc}, inc_deg={inc_deg}, "
                                        f"raan_deg={raan_deg}, argp_deg={argp_deg}, M_deg={M_deg} | "
                                        f"UNEXPECTED ERROR: {repr(e_general)}\n"
                                        f"{traceback.format_exc()}\n"
                                    )
                                    fout.write(msg)
                                    flush_to_disk()
                                    continue

        # final note
        fout.write("All runs finished (or skipped on error). End of log.\n")
        flush_to_disk()

    # return the path so caller can find the file
    return results_path
    
if __name__ == "__main__":
    main()
