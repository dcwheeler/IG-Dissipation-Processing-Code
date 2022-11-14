"""
Code for calculating dissipation values
Saves final results
Saves file used in MakeDissipationFitPlot.py, ToleranceEval.py, and WhiteNoiseEval.py
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
import xarray as xr

#endregion

#region load data
print('loading data')

v1 = xr.open_dataset(dataPath + '/vec12412despiked.nc')

v2 = xr.open_dataset(dataPath + '/vec12414despiked.nc')

v3 = xr.open_dataset(dataPath + '/vec8155despiked.nc')

#endregion

#run epCalc function on each instrument
print('v12412')
v1Results = fn.epCalc(v1,'12412')
print('v12414')
v2Results = fn.epCalc(v2,'12414')
print('v8155')
v3Results = fn.epCalc(v3,'8155')

#compine results into one datastructure
print('combining')
results = v1Results.merge(v2Results).merge(v3Results)

#save metadata notes
results.fitType12412.attrs['notes'] = '0 = not a long enough spectrum to even try, 1 = no good fits,\
                    2 = no fits that satisfy their own disspation derived cutoffs, 3 = good fits'

results.fitType12414.attrs['notes'] = '0 = not a long enough spectrum to even try, 1 = no good fits,\
                    2 = no fits that satisfy their own disspation derived cutoffs, 3 = good fits'

results.fitType8155.attrs['notes'] = '0 = not a long enough spectrum to even try, 1 = no good fits,\
                    2 = no fits that satisfy their own disspation derived cutoffs, 3 = good fits'

#save results
print('saving')
results.to_netcdf(dataPath+'/dissipations.nc') 







