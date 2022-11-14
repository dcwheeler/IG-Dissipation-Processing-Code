"""
Code for calculating dissipations for the tolerance tests on the despiking algorithm
and for compare the results of the despike tolerance tests compared to unaltered despiking algorithm
Saves some files for ExpSizeEval.py and ToleranceEval.py and also creates arrays for intermediate evaluation
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

#endregion

#region dissipation calc function

def epCombine(varString, segSize = 10, highLowSep = 1/5, turbVarCutoff = 5, waveCutoff = 1.025, unorigionalCutoff = 0.01,\
    lowWavenumberRemoval = 2, highSpectrumRemoval = 4, binSize = 50, genScale = 1/2, minDataPoints = 10,\
        minWavenumberSpan = 2.5, slopeConfidence = .95, peakProminence = 0.8, kdeBW=1):
        
    v1 = xr.open_dataset(tolerancePath+'/vec12412despiked_' + varString)
    v2 = xr.open_dataset(tolerancePath+'/vec12414despiked_' + varString)
    v3 = xr.open_dataset(tolerancePath+'/vec8155despiked_' + varString)

    print('v12412')
    v1Results = fn.epCalc(v1,'12412',segSize=segSize,highLowSep=highLowSep,turbVarCutoff=turbVarCutoff,waveCutoff=waveCutoff,\
        unorigionalCutoff=unorigionalCutoff,lowWavenumberRemoval=lowWavenumberRemoval,highSpectrumRemoval=highSpectrumRemoval,\
            binSize=binSize,genScale=genScale,minDataPoints=minDataPoints,minWavenumberSpan=minWavenumberSpan,\
                slopeConfidence=slopeConfidence,peakProminence=peakProminence,kdeBW=kdeBW)
    print('v12414')
    v2Results = fn.epCalc(v2,'12414',segSize=segSize,highLowSep=highLowSep,turbVarCutoff=turbVarCutoff,waveCutoff=waveCutoff,\
        unorigionalCutoff=unorigionalCutoff,lowWavenumberRemoval=lowWavenumberRemoval,highSpectrumRemoval=highSpectrumRemoval,\
            binSize=binSize,genScale=genScale,minDataPoints=minDataPoints,minWavenumberSpan=minWavenumberSpan,\
                slopeConfidence=slopeConfidence,peakProminence=peakProminence,kdeBW=kdeBW)
    print('v8155')
    v3Results = fn.epCalc(v3,'8155',segSize=segSize,highLowSep=highLowSep,turbVarCutoff=turbVarCutoff,waveCutoff=waveCutoff,\
        unorigionalCutoff=unorigionalCutoff,lowWavenumberRemoval=lowWavenumberRemoval,highSpectrumRemoval=highSpectrumRemoval,\
            binSize=binSize,genScale=genScale,minDataPoints=minDataPoints,minWavenumberSpan=minWavenumberSpan,\
                slopeConfidence=slopeConfidence,peakProminence=peakProminence,kdeBW=kdeBW)

    print('combining')
    results = v1Results.merge(v2Results).merge(v3Results)

    return results

#endregion

#region get file endings

#find all tolerance test despiked files
files = glob.glob(tolerancePath+'/vec*')

#pull the endings from these files to determine all tolerance test cases
endings = np.unique([''.join(np.insert(file.split('_')[-2:],1,'_')) for file in files])

#endregion

#region calculate dissipations

#loop through tolerance test cases
for ending in endings:
    #calculate corresponding dissipations for the test case
    results = epCombine(ending)
    #save results to corresponding location
    splitEnding = ending.split('_')
    #put the extra expansion size tests in their own folder
    if splitEnding[0] == 'expSize' and (splitEnding[1] != '0.005.nc' or splitEnding[1] != '0.02.nc'):
        results.to_netcdf(tolerancePath+'/expSizeTests/dissipations_'+ending)
    #otherwise store everything in the toleranceTests folder
    else:
        results.to_netcdf(tolerancePath+'/dissipations_'+ending)

#endregion

#after dissipation values are saved, compare the despiking results directly

#region despike comparison

#load original despiked data to compare for the tolerance tests
v1 = xr.open_dataset(dataPath + '/vec12412despiked.nc')
v2 = xr.open_dataset(dataPath + '/vec12414despiked.nc')
v3 = xr.open_dataset(dataPath + '/vec8155despiked.nc')

#calculate burst means
"""
Note, this part and a corresponding part in the for loop below take a while, so they are commented
uncomment them if you care about the mean values

def burstMeans(vec):
    meansP = np.array([np.nanmean(vec.Primary.values[vec.bNum30.values==i]) for i in vec.burst30.values])
    meansUp = np.array([np.nanmean(vec.Up.values[vec.bNum30.values==i]) for i in vec.burst30.values])
    return (meansP,meansUp)

(v1P,v1Up) = burstMeans(v1)
(v2P,v2Up) = burstMeans(v2)
(v3P,v3Up) = burstMeans(v3)
"""

#get total number of original data points (not identified as spikes) for all of the normal data set
totalOriginal = (np.sum(v1.UpOrig) + np.sum(v2.UpOrig) + np.sum(v3.UpOrig)).values

#get location for spikes in normal data set
v1Spikes = np.logical_and(~v1.UpOrig,~np.isnan(v1.Up))
v2Spikes = np.logical_and(~v2.UpOrig,~np.isnan(v2.Up))
v3Spikes = np.logical_and(~v3.UpOrig,~np.isnan(v3.Up))

#get total number of spikes in normal data set
totalSpikes = (np.sum(v1Spikes) + np.sum(v2Spikes) + np.sum(v3Spikes)).values

#create placeholder arrays
lostOriginal = np.empty(endings.size)
gainedOriginal = np.empty(endings.size)

lostSpikes = np.empty(endings.size)
gainedSpikes = np.empty(endings.size)

changeSpikes = np.empty(endings.size)

meanChangedP1 = np.empty((endings.size,v1.burst30.values.size))
meanChangedP2 = np.empty((endings.size,v2.burst30.values.size))
meanChangedP3 = np.empty((endings.size,v3.burst30.values.size))

meanChangedUp1 = np.empty((endings.size,v1.burst30.values.size))
meanChangedUp2 = np.empty((endings.size,v2.burst30.values.size))
meanChangedUp3 = np.empty((endings.size,v3.burst30.values.size))

#loop through each tolerance test case again to calculate comparisons
for i,ending in zip(np.arange(endings.size),endings):

    #load tolerance despike data
    v1new = xr.open_dataset(tolerancePath+'/vec12412despiked_' + ending)
    v2new = xr.open_dataset(tolerancePath+'/vec12414despiked_' + ending)
    v3new = xr.open_dataset(tolerancePath+'/vec8155despiked_' + ending)

    #get location of tolerance test spikes
    v1newSpikes = np.logical_and(~v1new.UpOrig,~np.isnan(v1new.Up))
    v2newSpikes = np.logical_and(~v2new.UpOrig,~np.isnan(v2new.Up))
    v3newSpikes = np.logical_and(~v3new.UpOrig,~np.isnan(v3new.Up))

    #calculate how many more spikes are detected with the tolerance test
    changeSpikes[i] = np.sum(v1newSpikes)+np.sum(v2newSpikes)+np.sum(v3newSpikes)-totalSpikes

    #calculate how many original data points become non-original in tolerance test
    lostOriginal[i] = np.sum(np.logical_and(v1.UpOrig,~v1new.UpOrig)) + \
        np.sum(np.logical_and(v2.UpOrig,~v2new.UpOrig)) + \
            np.sum(np.logical_and(v3.UpOrig,~v3new.UpOrig))

    #calculate how non-original points become original points in tolerance test
    gainedOriginal[i] = np.sum(np.logical_and(~v1.UpOrig,v1new.UpOrig)) + \
        np.sum(np.logical_and(~v2.UpOrig,v2new.UpOrig)) + \
            np.sum(np.logical_and(~v3.UpOrig,v3new.UpOrig))

    #calculate how many spikes are no longer spikes in tolerance test
    lostSpikes[i] = np.sum(np.logical_and(v1Spikes,~v1newSpikes)) + \
        np.sum(np.logical_and(v2Spikes,~v2newSpikes)) + \
            np.sum(np.logical_and(v3Spikes,~v3newSpikes))

    #calculate how many spikes in the tolerance test were not spikes originally
    gainedSpikes[i] = np.sum(np.logical_and(~v1Spikes,v1newSpikes)) + \
        np.sum(np.logical_and(~v2Spikes,v2newSpikes)) + \
            np.sum(np.logical_and(~v3Spikes,v3newSpikes))

    #calculate new means
    (v1Pnew,v1UpNew) = burstMeans(v1new)
    (v2Pnew,v2UpNew) = burstMeans(v2new)
    (v3Pnew,v3UpNew) = burstMeans(v3new)

    #calculate change in means
    """
    Note, this is the part that takes a while, so it is commented out
    uncomment these lines and the corresponding ones before the for loop if you care about mean values

    meanChangedP1[i,:] = (v1P-v1Pnew)/v1P
    meanChangedP2[i,:] = (v2P-v2Pnew)/v2P
    meanChangedP3[i,:] = (v3P-v3Pnew)/v3P
    
    meanChangedUp1[i,:] = (v1Up-v1UpNew)/v1Up
    meanChangedUp2[i,:] = (v2Up-v2UpNew)/v2Up
    meanChangedUp3[i,:] = (v3Up-v3UpNew)/v3Up
    """

#endregion

#note, the despike comparison results are not saved because they are quick to calculate. these arrays are just for reference