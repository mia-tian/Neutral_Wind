# Will Parker
#   12:54 PM
import numpy as np
from pymsis import msis
import matplotlib.pyplot as plt


dates = np.arange(np.datetime64("2024-01-01T00:00"), np.datetime64("2024-12-30T00:00"), np.timedelta64(30, "m"))
lat = np.linspace(-60,60,1)
lon = np.linspace(-180,180,1)
alt = np.arange(6,1000,1)
# geomagnetic_activity=-1 is a storm-time run
data = msis.run(dates, lon, lat, alt, geomagnetic_activity=-1)

# Plot the data

# Total mass density over time
plt.plot(dates, data[:, 0, 0, 0, 0])
plt.show()