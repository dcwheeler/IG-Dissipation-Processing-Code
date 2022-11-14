"""
Code for running tolerance tests on the spectrum calculation and dissipation fitting algorithms
saves output files to data/toleranceTests/dissipations_{variable name}_{variable value}.nc
these files are used by ToleranceEval.py
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

#region dissipation calc function

def epCombine(segSize = 10, highLowSep = 1/5, turbVarCutoff = 5, waveCutoff = 1.025, unorigionalCutoff = 0.01,\
    lowWavenumberRemoval = 2, highSpectrumRemoval = 4, binSize = 50, genScale = 1/2, minDataPoints = 10,\
        minWavenumberSpan = 2.5, slopeConfidence = .95, peakProminence = 0.8, kdeBW=1, correct=True):

    #runsdissipation calc with corresponding values on each of the 3 vectors, and combines the results
        
    print('v12412')
    v1Results = fn.epCalc(v1,'12412',segSize=segSize,highLowSep=highLowSep,turbVarCutoff=turbVarCutoff,waveCutoff=waveCutoff,\
        unorigionalCutoff=unorigionalCutoff,lowWavenumberRemoval=lowWavenumberRemoval,highSpectrumRemoval=highSpectrumRemoval,\
            binSize=binSize,genScale=genScale,minDataPoints=minDataPoints,minWavenumberSpan=minWavenumberSpan,\
                slopeConfidence=slopeConfidence,peakProminence=peakProminence,kdeBW=kdeBW,correct=correct)
    print('v12414')
    v2Results = fn.epCalc(v2,'12414',segSize=segSize,highLowSep=highLowSep,turbVarCutoff=turbVarCutoff,waveCutoff=waveCutoff,\
        unorigionalCutoff=unorigionalCutoff,lowWavenumberRemoval=lowWavenumberRemoval,highSpectrumRemoval=highSpectrumRemoval,\
            binSize=binSize,genScale=genScale,minDataPoints=minDataPoints,minWavenumberSpan=minWavenumberSpan,\
                slopeConfidence=slopeConfidence,peakProminence=peakProminence,kdeBW=kdeBW,correct=correct)
    print('v8155')
    v3Results = fn.epCalc(v3,'8155',segSize=segSize,highLowSep=highLowSep,turbVarCutoff=turbVarCutoff,waveCutoff=waveCutoff,\
        unorigionalCutoff=unorigionalCutoff,lowWavenumberRemoval=lowWavenumberRemoval,highSpectrumRemoval=highSpectrumRemoval,\
            binSize=binSize,genScale=genScale,minDataPoints=minDataPoints,minWavenumberSpan=minWavenumberSpan,\
                slopeConfidence=slopeConfidence,peakProminence=peakProminence,kdeBW=kdeBW,correct=correct)

    print('combining')
    results = v1Results.merge(v2Results).merge(v3Results)

    return results

#endregion

"""
The code below runs the function above for all of the different tolerance test cases and saves the results in a file
It takes a long time to run, so I grouped them into 3 sets that succesfully ran in parallel on my computer.
"""

#region review round 2
results = epCombine(correct=False)
results.to_netcdf(tolerancePath+'/dissipations_noRosmanCorrection.nc')
#endregion

#region set 1
segSizeHigh = 15
results = epCombine(segSize = segSizeHigh)
results.to_netcdf(tolerancePath+'/dissipations_segSize_{:.2f}.nc'.format(segSizeHigh))

segSizeLow = 5
results = epCombine(segSize = segSizeLow)
results.to_netcdf(tolerancePath+'/dissipations_segSize_{:.2f}.nc'.format(segSizeLow))

highLowSepHigh = 2/5
results = epCombine(highLowSep=highLowSepHigh)
results.to_netcdf(tolerancePath+'/dissipations_highLowSep_{:.2f}.nc'.format(highLowSepHigh))

highLowSepLow = 1/10
results = epCombine(highLowSep=highLowSepLow)
results.to_netcdf(tolerancePath+'/dissipations_highLowSep_{:.2f}nc'.format(highLowSepLow))

turbVarCutoffHigh = 10
results = epCombine(turbVarCutoff=turbVarCutoffHigh)
results.to_netcdf(tolerancePath+'/dissipations_turbVarCutoff_{:.2f}.nc'.format(turbVarCutoffHigh))

turbVarCutoffLow = 2.5
results = epCombine(turbVarCutoff=turbVarCutoffLow)
results.to_netcdf(tolerancePath+'/dissipations_turbVarCutoff_{:.2f}.nc'.format(turbVarCutoffLow))

waveCutoffHigh = 1.2
results = epCombine(waveCutoff=waveCutoffHigh)
results.to_netcdf(tolerancePath+'/dissipations_waveCutoff_{:.2f}.nc'.format(waveCutoffHigh))

waveCutoffLow = 0.8
results = epCombine(waveCutoff=waveCutoffLow)
results.to_netcdf(tolerancePath+'/dissipations_waveCutoff_{:.2f}.nc'.format(waveCutoffLow))

lowWavenumberRemovalHigh = 4
results = epCombine(lowWavenumberRemoval=lowWavenumberRemovalHigh)
results.to_netcdf(tolerancePath+'/dissipations_lowWavenumberRemoval_{:.2f}.nc'.format(lowWavenumberRemovalHigh))

lowWavenumberRemovalLow = 0
results = epCombine(lowWavenumberRemoval=lowWavenumberRemovalLow)
results.to_netcdf(tolerancePath+'/dissipations_lowWavenumberRemoval_{:.2f}.nc'.format(lowWavenumberRemovalLow))

#endregion

#region set 2

highSpectrumRemovalHigh = 8
results = epCombine(highSpectrumRemoval=highSpectrumRemovalHigh)
results.to_netcdf(tolerancePath+'/dissipations_highSpectrumRemoval_{:.2f}.nc'.format(highSpectrumRemovalHigh))

highSpectrumRemovalLow = 2
results = epCombine(highSpectrumRemoval=highSpectrumRemovalLow)
results.to_netcdf(tolerancePath+'/dissipations_highSpectrumRemoval_{:.2f}.nc'.format(highSpectrumRemovalLow))

binSizeHigh = 75
results = epCombine(binSize=binSizeHigh)
results.to_netcdf(tolerancePath+'/dissipations_binSize_{:.2f}.nc'.format(binSizeHigh))

binSizeLow = 25
results = epCombine(binSize=binSizeLow)
results.to_netcdf(tolerancePath+'/dissipations_binSize_{:.2f}.nc'.format(binSizeLow))

genScaleHigh = 1
results = epCombine(genScale=genScaleHigh)
results.to_netcdf(tolerancePath+'/dissipations_genScale_{:.2f}.nc'.format(genScaleHigh))

genScaleLow = 1/4
results = epCombine(genScale=genScaleLow)
results.to_netcdf(tolerancePath+'/dissipations_genScale_{:.2f}.nc'.format(genScaleLow))

minDataPointsHigh = 15
results = epCombine(minDataPoints=minDataPointsHigh)
results.to_netcdf(tolerancePath+'/dissipations_minDataPoints_{:.2f}.nc'.format(minDataPointsHigh))

minDataPointsLow = 5
results = epCombine(minDataPoints=minDataPointsLow)
results.to_netcdf(tolerancePath+'/dissipations_minDataPoints_{:.2f}.nc'.format(minDataPointsLow))

#endregion

#region set 3

minWavenumberSpanHigh = 5
results = epCombine(minWavenumberSpan=minWavenumberSpanHigh)
results.to_netcdf(tolerancePath+'/dissipations_minWavenumberSpan_{:.2f}.nc'.format(minWavenumberSpanHigh))

minWavenumberSpanLow = 1.25
results = epCombine(minWavenumberSpan=minWavenumberSpanLow)
results.to_netcdf(tolerancePath+'/dissipations_minWavenumberSpan_{:.2f}.nc'.format(minWavenumberSpanLow))

slopeConfidenceHigh = .99
results = epCombine(slopeConfidence=slopeConfidenceHigh)
results.to_netcdf(tolerancePath+'/dissipations_slopeConfidence_{:.2f}.nc'.format(slopeConfidenceHigh))

slopeConfidenceLow = 0.9
results = epCombine(slopeConfidence=slopeConfidenceLow)
results.to_netcdf(tolerancePath+'/dissipations_slopeConfidence_{:.2f}.nc'.format(slopeConfidenceLow))

peakProminenceHigh = 0.9
results = epCombine(peakProminence=peakProminenceHigh)
results.to_netcdf(tolerancePath+'/dissipations_peakProminence_{:.2f}.nc'.format(peakProminenceHigh))

peakProminenceLow = 0.7
results = epCombine(peakProminence=peakProminenceLow)
results.to_netcdf(tolerancePath+'/dissipations_peakProminence_{:.2f}.nc'.format(peakProminenceLow))

kdeBWHigh = 1.2
results = epCombine(kdeBW=kdeBWHigh)
results.to_netcdf(tolerancePath+'/dissipations_kdeBW_{:.2f}.nc'.format(kdeBWHigh))

kdeBWLow = 0.8
results = epCombine(kdeBW=kdeBWLow)
results.to_netcdf(tolerancePath+'/dissipations_kdeBW_{:.2f}.nc'.format(kdeBWLow))

unorigionalCutoffHigh = .02
results = epCombine(unorigionalCutoff=unorigionalCutoffHigh)
results.to_netcdf(tolerancePath+'/dissipations_unorigionalCutoff_{:.2f}.nc'.format(unorigionalCutoffHigh))

unorigionalCutoffLow = 0.005
results = epCombine(unorigionalCutoff=unorigionalCutoffLow)
results.to_netcdf(tolerancePath+'/dissipations_unorigionalCutoff_{:.3f}.nc'.format(unorigionalCutoffLow))

#endregion
