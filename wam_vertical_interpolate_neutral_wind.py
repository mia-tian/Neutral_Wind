from scipy.interpolate import interp1d
from netCDF4 import Dataset
import numpy as np
from glob import glob
from os.path import basename
from itertools import product
import matplotlib.pyplot as plt

#######################################
HEIGHTS = np.arange(0, 1001, 10)*1000

INPUT_PATH = 'raw_wind'
FILE_GLOB = 'w?s.t??z.wam10.*.nc'

OUTPUT_PATH = 'fixed_height_wind'
OUTPUT_FORMAT = 'fixed_height_wind.{}'
#######################################
DIMENSIONS = ['x01', 'x02', 'x03']
COPY_VARIABLES_1D = ['lon', 'lat']
COPY_VARIABLES_3D = ['O_Density', 'O2_Density', 'N2_Density']
COPY_VARIABLES_3D_MOLAR_MASS = [15.999, 31.998, 28.014]
NEW_VARS = ['u_neutral', 'v_neutral', 'w_neutral']
NEW_VAR_LONGNAME = ['eastward_wind_neutral', 'northward_wind_neutral', 'upward_wind_neutral']
NEW_VAR_DTYPE = ['f4']*3
NEW_VAR_UNITS = ['m s-1']*3
AVOGADRO = 6.02214076e23
#######################################

def create_output(file):
    ncfid_i = Dataset(file)

    outfile = '{}/{}'.format(OUTPUT_PATH, OUTPUT_FORMAT.format(basename(file)))

    ncfid_o = Dataset(outfile, 'w')

    for d, v in zip(DIMENSIONS, COPY_VARIABLES_1D):
        dim = ncfid_i.dimensions[d]
        ncfid_o.createDimension(v, dim.size)

    ncfid_o.createDimension('height', len(HEIGHTS))
    varo = ncfid_o.createVariable('height', 'f4', ('height',))
    varo[:] = HEIGHTS

    for v in COPY_VARIABLES_1D:
        vari = ncfid_i.variables[v]
        varo = ncfid_o.createVariable(v, vari.datatype, (v,))
        varo[:] = vari[:]

    for v, ln, dtype, unit in zip(NEW_VARS, NEW_VAR_LONGNAME, NEW_VAR_DTYPE, NEW_VAR_UNITS):
        varo = ncfid_o.createVariable(v, dtype, ('height', 'lat', 'lon',))
        varo.long_name = ln
        varo.units = unit

    return ncfid_o

def convert(file):
    ncfid_i = Dataset(file)
    ncfid_o = create_output(file)

    height = ncfid_i.variables['height'][:]
    heights, lats, lons = height.shape

    for var in NEW_VARS: # u_neutral, v_neutral, w_neutral
        v = ncfid_i.variables[var][:] 
        for i,j in product(range(lats), range(lons)):
            f = interp1d(height[:,i,j], v[:,i,j], bounds_error=False, fill_value=(v[0,i,j],v[-1,i,j]), kind='linear')
            ncfid_o.variables[var][:,i,j] = f(HEIGHTS)

    ncfid_i.close()

def main():
    files = glob('{}/{}'.format(INPUT_PATH, FILE_GLOB))

    for file in files:
        print('File Converted.')
        convert(file)

if __name__ == '__main__':
    main()
