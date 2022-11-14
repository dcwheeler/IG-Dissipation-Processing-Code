"""
Script for running the despiking step of the despike tolerance tests
Saves the files to be used in DespikeToleranceTest2.py
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

#region define Process Cases

#set tolerance test parameters for despiking
lpHigh = 1/10
lpLow = 1/30
expSizeHigh = .02
expSizeLow = 0.005
expEndHigh = 1
expEndLow = 0.9

#define process function to run on each vector
def Process(data,BPressure,Pdate1,Pdate2,badSections,reverse,vecnum):
    #runs despiking with corresponding tolerance test parameter and then saves the result of each test
    #data, BPressure,Pdate1,Pdate2,badSections, reverse, and vecnum are all specific to the vector being processed

    print('lowpass high test')
    data_new = fn.ProcessVec(data,BPressure,Pdate1,Pdate2,badSections,reverse,lp=lpHigh)
    data_new.to_netcdf(tolerancePath+'/vec'+vecnum+'despiked_lp_{:.2f}.nc'.format(lpHigh))

    print('lowpass low test')
    data_new = fn.ProcessVec(data,BPressure,Pdate1,Pdate2,badSections,reverse,lp=lpLow)
    data_new.to_netcdf(tolerancePath+'/vec'+vecnum+'despiked_lp_{:.2f}.nc'.format(lpLow))

    print('expand size high test')
    data_new = fn.ProcessVec(data,BPressure,Pdate1,Pdate2,badSections,reverse,expSize=expSizeHigh)
    data_new.to_netcdf(tolerancePath+'/vec'+vecnum+'despiked_expSize_{:.2f}.nc'.format(expSizeHigh))

    print('expand size low test')
    data_new = fn.ProcessVec(data,BPressure,Pdate1,Pdate2,badSections,reverse,expSize=expSizeLow)
    data_new.to_netcdf(tolerancePath+'/vec'+vecnum+'despiked_expSize_{:.2f}.nc'.format(expSizeLow))

    print('expand end high test')
    data_new = fn.ProcessVec(data,BPressure,Pdate1,Pdate2,badSections,reverse,expEnd=expEndHigh)
    data_new.to_netcdf(tolerancePath+'/vec'+vecnum+'despiked_expEnd_{:.2f}.nc'.format(expEndHigh))

    print('expand end low test')
    data_new = fn.ProcessVec(data,BPressure,Pdate1,Pdate2,badSections,reverse,expEnd=expEndLow)
    data_new.to_netcdf(tolerancePath+'/vec'+vecnum+'despiked_expEnd_{:.2f}.nc'.format(expEndLow))

#further define despiking tolerance test cases for the focused expansion step size test
expSizeSteps = [0.006,0.007,0.008,0.009,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019]

#define expansion step size tolerance test function to run on each vector
def Process2(data,BPressure,Pdate1,Pdate2,badSections,reverse,vecnum):
    #runs despiking with corresponding expansion step size parameter and then saves the result of each test
    #data, BPressure,Pdate1,Pdate2,badSections, reverse, and vecnum are all specific to the vector being processed
    for i in np.arange(np.size(expSizeSteps)):
        print('test '+str(i))
        data_new = fn.ProcessVec(data,BPressure,Pdate1,Pdate2,badSections,reverse,expSize=expSizeSteps[i])
        data_new.to_netcdf(tolerancePath+'/vec'+vecnum+'despiked_expSize_{:.3f}.nc'.format(expSizeSteps[i]))

#endregion

#region vector 12412  

print('processing vector 12412')

print('loading data')

#load raw data
with xr.open_dataset(dataPath + '/vec12412raw.nc') as ds:
    data = ds.load()

#set times for pressure correction
Pdate1 = np.datetime64('2020-02-10T12:14:00')

Pdate2 = np.datetime64('2020-02-10T12:25:00')

#set sections of known bad data
badSections = []

#set variable for reversing direction after rotation if needed
reverse = False

#run tolerance tests
Process(data,BPressure,Pdate1,Pdate2,badSections,reverse,'12412')

#run expansion step size tests
Process2(data,BPressure,Pdate1,Pdate2,badSections,reverse,'12412')

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

#run tolerance tests
Process(data,BPressure,Pdate1,Pdate2,badSections,reverse,'12414')

#run expansion step size tests
Process2(data,BPressure,Pdate1,Pdate2,badSections,reverse,'12414')

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

#run tolerance tests
Process(data,BPressure,Pdate1,Pdate2,badSections,reverse,'8155')

#run expansion step size tests
Process2(data,BPressure,Pdate1,Pdate2,badSections,reverse,'8155')

#endregion
