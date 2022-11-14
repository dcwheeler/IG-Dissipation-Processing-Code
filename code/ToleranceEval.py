"""
Code for testing spectra calculation with semi-idealized model
creates an xarray dataset (toleranceResults) to look at
does not save a new file because this runs pretty fast
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

# import other important librarys
import numpy as np
import xarray as xr
import glob

#endregion

#region load data
print('loading data')

results = xr.open_dataset(dataPath+'/dissipations.nc')

#store total number of dissipation values
epNum = np.sum(~np.isnan(results.ep12412))+np.sum(~np.isnan(results.ep12414))+np.sum(~np.isnan(results.ep8155))

#define standard variable values
standard = {'binSize':50, 'genScale':1/2, 'highLowSep':1/5, 'highSpectrumRemoval':4, 'kdeBW':1, \
    'lowWavenumberRemoval':2, 'minDataPoints':10, 'minWavenumberSpan':2.5, \
        'peakProminence':.8, 'segSize':10, 'slopeConfidence':0.95, 'turbVarCutoff':5, \
            'unorigionalCutoff':0.01, 'waveCutoff':1.025, 'expEnd':0.95, 'expSize':0.01, 'lp':1/20}

#endregion

#region calculate test changes

#find all tolerance test files
files = glob.glob(tolerancePath+'/dissipations*')

#initialize arrays
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

    #grab file
    file = files[i]

    #load data
    test = xr.open_dataset(file)

    #store variable being tested
    variables[i] = file.split('_')[-2]

    #store test value
    values[i] = float(file.split('_')[-1][:-3])

    #calculate how many dissipation values in the tolerance test were nan originally
    epGained[i] = np.sum(np.logical_and(np.isnan(results.ep12412),~np.isnan(test.ep12412)))+\
        np.sum(np.logical_and(np.isnan(results.ep12414),~np.isnan(test.ep12414)))+\
            np.sum(np.logical_and(np.isnan(results.ep8155),~np.isnan(test.ep8155)))
    
    #calculate how many original dissipation values became nan in the tolerance test
    epLost[i] = np.sum(np.logical_and(~np.isnan(results.ep12412),np.isnan(test.ep12412)))+\
        np.sum(np.logical_and(~np.isnan(results.ep12414),np.isnan(test.ep12414)))+\
            np.sum(np.logical_and(~np.isnan(results.ep8155),np.isnan(test.ep8155)))

    #calculate fractional change in dissipation from original to the tolerance test
    epChangeFrac[i,:] = np.concatenate((((test.ep12412-results.ep12412)/results.ep12412).values,\
        ((test.ep12414-results.ep12414)/results.ep12414).values,((test.ep8155-results.ep8155)/results.ep8155).values))

    #calculate how many of these changes are greater than the 16.3% uncertainty
    numChangeLarge[i] = np.sum(np.abs(epChangeFrac[i,:])>0.163)

    #calculate maximum fractional change
    maxChange[i] = np.nanmax(np.abs(epChangeFrac[i,:]))

    #calculate average fractional change
    avChange[i] = np.nanmean(epChangeFrac[i,:])

    #calculate the average of the absolute value of the fractional change
    avAbsChange[i] = np.nanmean(np.abs(epChangeFrac[i,:]))

#endregion

#region create xarray dataset

epOrig = np.concatenate((results.ep12412.values,results.ep12414.values,results.ep8155.values))

#put results in an xarray dataset for easier viewing
toleranceResults = xr.Dataset()

#store coordinates
toleranceResults.coords['variable'] = np.unique(variables)
toleranceResults.coords['test'] = ['low','high']
toleranceResults.coords['burst'] = np.arange(epChangeFrac.shape[-1])

#initialize arrays of the proper shape
epChangeFinal = np.empty((toleranceResults.variable.size,toleranceResults.test.size,toleranceResults.burst.size))
numChangeFinal = np.empty((toleranceResults.variable.size,toleranceResults.test.size))
maxChangeFinal = np.empty(numChangeFinal.shape)
avChangeFinal = np.empty(numChangeFinal.shape)
avAbsChangeFinal = np.empty(numChangeFinal.shape)
valuesFinal = np.empty(numChangeFinal.shape)
gainedFinal = np.empty(numChangeFinal.shape)
lostFinal = np.empty(numChangeFinal.shape)


#loop through each variable being tested
for i in np.arange(toleranceResults.variable.size):
    
    #grab locations of corresponding tolerance tests
    loc =  np.array([j for j,e in enumerate(variables) if e == toleranceResults.variable.values[i]])

    #sort into low and high tests
    loc = loc[np.argsort(values[loc])]

    #store results
    valuesFinal[i,:] = values[loc]
    epChangeFinal[i,:,:] = epChangeFrac[loc,:]
    numChangeFinal[i,:] = numChangeLarge[loc]
    maxChangeFinal[i,:] = maxChange[loc]
    avChangeFinal[i,:] = avChange[loc]
    avAbsChangeFinal[i,:] = avAbsChange[loc]
    gainedFinal[i,:] = epGained[loc]
    lostFinal[i,:] = epLost[loc]

#save to dataset
toleranceResults['epChangeFrac'] = (('variable','test','burst'),epChangeFinal)
toleranceResults['newValue'] = (('variable','test'),valuesFinal)
toleranceResults['numChangeLarge'] = (('variable','test'), numChangeFinal)
toleranceResults['maxChange'] = (('variable','test'), maxChangeFinal)
toleranceResults['avChange'] = (('variable','test'), avChangeFinal)
toleranceResults['avAbsChange'] = (('variable','test'), avAbsChangeFinal)
toleranceResults['epGained'] = (('variable','test'), gainedFinal)
toleranceResults['epLost'] = (('variable','test'), lostFinal)
toleranceResults['standardValue'] = ('variable',[standard[var] for var in toleranceResults.variable.values])
toleranceResults['epOrig'] = ('burst',epOrig)

ind = np.logical_and(~np.isnan(toleranceResults.epOrig.values),\
    np.isnan(toleranceResults.sel(variable='highSpectrumRemoval').\
        sel(test='high').epChangeFrac.values))