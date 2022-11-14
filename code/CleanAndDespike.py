"""
Script for cleaning and despiking raw vector data from Los Pen for IG experiment, methods paper
Saves files used in DissipationCalc.py, DespikeToleranceStep2, MakeSpectrumCalcPlot, and WhiteNoiseEval.py
@author: Duncan Wheeler
"""

#before running, make sure that functionPath points to where Functions.py is located, dataPath points to where the data files are located,
#plotPath points to where the plots are located, and tolerancePath points to where the tolerance test data is located.

functionPath = "/data/tidalpower/LosPen/IGexperiment2020/MethodsPaper/code"
dataPath = '/data/tidalpower/LosPen/IGexperiment2020/MethodsPaper/data'
plotPath = '/data/tidalpower/LosPen/IGexperiment2020/MethodsPaper/plots'
tolerancePath = '/data/tidalpower/LosPen/IGexperiment2020/MethodsPaper/toleranceTests'

#region setup
print('importing libraries')

# always import python debugger in case an error occurs!!!
import pdb

# Add code base to be able to import Functions library
import sys
sys.path.append(functionPath)
import Functions as fn

# import other important librarys
import numpy as np
import xarray as xr

#load met data
with xr.open_dataset(dataPath+'/met.nc') as ds:
    BPressure = ds.BarometricPressure

#endregion


#region vector 12412  

print('processing vector 12412')

print('loading data')

#load raw data
data = xr.open_dataset(dataPath + '/vec12412raw.nc')

#set times for pressure correction
Pdate1 = np.datetime64('2020-02-10T12:14:00')

Pdate2 = np.datetime64('2020-02-10T12:25:00')

#set sections of known bad data
badSections = []

#set variable for reversing direction after rotation if needed
reverse = False

#process data
data_new = fn.ProcessVec(data,BPressure,Pdate1,Pdate2,badSections,reverse)

print('saving data')

#save data
data_new.to_netcdf(dataPath+'/vec12412despiked.nc')

#endregion


#region vector 12414

print('processing vector 12414')

print('loading data')

#load raw data
with xr.open_dataset(dataPath + '/vec12414raw.nc') as ds:
    data = ds.load()

#set times for pressure correction
Pdate1 = np.datetime64('2020-03-03T21:00:00')

Pdate2 = np.datetime64('2020-03-04T05:00:00')

#set sections of known bad data
badSections = [[np.datetime64('2020-02-12T03:00:00'),np.datetime64('2020-02-12T05:00:00')],\
    [np.datetime64('2020-02-12T14:00:00'),np.datetime64('2020-02-12T15:00:00')],\
        [np.datetime64('2020-02-12T15:00:00'),np.datetime64('2020-02-12T16:00:00')],\
            [np.datetime64('2020-02-13T04:00:00'),np.datetime64('2020-02-13T05:00:00')]]

#set variable for reversing direction after rotation if needed
reverse = True

#process data
data_new = fn.ProcessVec(data,BPressure,Pdate1,Pdate2,badSections,reverse)

#save data
print('Saving vector 12414')
data_new.to_netcdf(dataPath+'/vec12414despiked.nc')

#endregion


#region vector 8155

print('processing vector 8155')

print('loading data')

#load raw data
with xr.open_dataset(dataPath + '/vec8155raw.nc') as ds:
    data = ds.load()

#set times for pressure correction
Pdate1 = None

Pdate2 = None

#set sections of known bad data
badSections = [[np.datetime64('2020-02-07T05:00:00'),np.datetime64('2020-02-07T06:00:00')]] #burst num 118

#set variable for reversing direction after rotation if needed
reverse = False

#process data
data_new = fn.ProcessVec(data,BPressure,Pdate1,Pdate2,badSections,reverse)

#save data
print('Saving vector 8155')
data_new.to_netcdf(dataPath+'/vec8155despiked.nc')

#endregion
