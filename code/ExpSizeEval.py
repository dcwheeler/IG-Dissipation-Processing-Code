"""
Code for evaluating the detailed expansion step size tolerance tests
creates an xarray dataset (toleranceResults) and a figure to look at
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
import glob
import matplotlib.pyplot as plt

plt.ion()

#endregion

#region load data
print('loading data')

#load final dissipation results
results = xr.open_dataset(dataPath+'/dissipations.nc')

#store total number of dissipation values
epNum = np.sum(~np.isnan(results.ep12412))+np.sum(~np.isnan(results.ep12414))+np.sum(~np.isnan(results.ep8155))

#store typical variable values
standard = {'binSize':50, 'genScale':1/2, 'highLowSep':1/5, 'highSpectrumRemoval':4, 'kdeBW':1, \
    'lowWavenumberRemoval':2, 'minDataPoints':10, 'minWavenumberSpan':2.5, \
        'peakProminence':.8, 'segSize':10, 'slopeConfidence':0.95, 'turbVarCutoff':5, \
            'unorigionalCutoff':0.01, 'waveCutoff':1.025, 'expEnd':0.95, 'expSize':0.01, 'lp':1/20}

#pull all of the expansion step size tolerance test dissipation result files
files = glob.glob(tolerancePath+'/expSizeTests/dissipations*')

#endregion

#region calculate changes

#create placeholder arrays
variables = ['']*np.size(files)
values = np.empty(np.size(files))
epGained = np.empty(np.size(files))
epLost = np.empty(np.size(files))
epChangeFrac = np.empty((np.size(files),results.ep12412.size+results.ep12414.size+results.ep8155.size))
numChangeLarge = np.empty(np.size(files))
maxChange = np.empty(values.size)
avChange = np.empty(values.size)
avAbsChange = np.empty(values.size)

#loop through each tolerance test
for i in np.arange(values.size):

    #store file name
    file = files[i]

    #load data
    test = xr.open_dataset(file)

    #store variable name
    variables[i] = file.split('_')[-2]

    #store variable value
    values[i] = float(file.split('_')[-1][:-3])

    #calculate and store total number of dissipation values calculated in the
    #tolerance test, but not in the original algorithm
    epGained[i] = np.sum(np.logical_and(np.isnan(results.ep12412),~np.isnan(test.ep12412)))+\
        np.sum(np.logical_and(np.isnan(results.ep12414),~np.isnan(test.ep12414)))+\
            np.sum(np.logical_and(np.isnan(results.ep8155),~np.isnan(test.ep8155)))
    
    #calculate and store total number of dissipation values calculated in the
    #original algorithm, but not in the tolerance test
    epLost[i] = np.sum(np.logical_and(~np.isnan(results.ep12412),np.isnan(test.ep12412)))+\
        np.sum(np.logical_and(~np.isnan(results.ep12414),np.isnan(test.ep12414)))+\
            np.sum(np.logical_and(~np.isnan(results.ep8155),np.isnan(test.ep8155)))

    #calculate the fractional change in dissipation value between original algorithm and tolerance test
    epChangeFrac[i,:] = np.concatenate((((test.ep12412-results.ep12412)/results.ep12412).values,\
        ((test.ep12414-results.ep12414)/results.ep12414).values,((test.ep8155-results.ep8155)/results.ep8155).values))
    
    #calculate how many of these changes were bigger than the typical uncertainty
    numChangeLarge[i] = np.sum(np.abs(epChangeFrac[i,:])>0.163)

    #calculate the largest change
    maxChange[i] = np.nanmax(np.abs(epChangeFrac[i,:]))

    #calculate the average change
    avChange[i] = np.nanmean(epChangeFrac[i,:])

    #calculate the average of the absolute value of the change
    avAbsChange[i] = np.nanmean(np.abs(epChangeFrac[i,:]))

#endregion

#region store results

#store results in an xarray dataset
toleranceResults = xr.Dataset()

#store coordinates
toleranceResults.coords['variable'] = np.unique(variables)
toleranceResults.coords['test'] = values[np.argsort(values)]
toleranceResults.coords['burst'] = np.arange(epChangeFrac.shape[-1])

#initialize arrays of the correct shape
epChangeFinal = np.empty((toleranceResults.variable.size,toleranceResults.test.size,toleranceResults.burst.size))
numChangeFinal = np.empty((toleranceResults.variable.size,toleranceResults.test.size))
maxChangeFinal = np.empty(numChangeFinal.shape)
avChangeFinal = np.empty(numChangeFinal.shape)
avAbsChangeFinal = np.empty(numChangeFinal.shape)
valuesFinal = np.empty(numChangeFinal.shape)
gainedFinal = np.empty(numChangeFinal.shape)
lostFinal = np.empty(numChangeFinal.shape)

#populate the arrays
for i in np.arange(toleranceResults.variable.size):
    loc =  np.array([j for j,e in enumerate(variables) if e == toleranceResults.variable.values[i]])
    loc = loc[np.argsort(values[loc])]
    valuesFinal[i,:] = values[loc]
    epChangeFinal[i,:,:] = epChangeFrac[loc,:]
    numChangeFinal[i,:] = numChangeLarge[loc]
    maxChangeFinal[i,:] = maxChange[loc]
    avChangeFinal[i,:] = avChange[loc]
    avAbsChangeFinal[i,:] = avAbsChange[loc]
    gainedFinal[i,:] = epGained[loc]
    lostFinal[i,:] = epLost[loc]

#store arrays in xarray dataset
toleranceResults['epChangeFrac'] = (('variable','test','burst'),epChangeFinal)
toleranceResults['newValue'] = (('variable','test'),valuesFinal)
toleranceResults['numChangeLarge'] = (('variable','test'), numChangeFinal)
toleranceResults['maxChange'] = (('variable','test'), maxChangeFinal)
toleranceResults['avChange'] = (('variable','test'), avChangeFinal)
toleranceResults['avAbsChange'] = (('variable','test'), avAbsChangeFinal)
toleranceResults['epGained'] = (('variable','test'), gainedFinal)
toleranceResults['epLost'] = (('variable','test'), lostFinal)
toleranceResults['standardValue'] = ('variable',[standard[var] for var in toleranceResults.variable.values])

#endregion

#region make plot

#pull the tolerance test values and insert original algorithm case for plotting
expSize = np.insert(toleranceResults.test.values,5,0.01)

#do the same for the average change in dissipation value
avEr = np.insert(toleranceResults.avChange[0,:].values,5,0)

#fit a line to get an idea of the slope
fit = np.polyfit(expSize,avEr,1)

#make plot
plt.plot(expSize,avEr)
plt.plot(expSize,fit[-1]+expSize*fit[0])
plt.axvline(0.01)
plt.axhline(0)

plt.xlabel('Expansion Step Size')
plt.ylabel('Average Dissipation % Change from 0.01 Case')

#endregion