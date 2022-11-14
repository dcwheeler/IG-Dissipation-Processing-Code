"""
Code for testing how subtracting flat white noise value from spectra
affects final dissipation values
If the spectrum plotting portion is run, saves to plots/whiteNoiseTest_{vector number}_{spectrum number}.png
saves dissipation values to /data/dissipationsNoiseRemovedTest_10-9.nc
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
from iapws import iapws95

#endregion

#region load data
print('loading data')

v1 = xr.open_dataset(dataPath + '/vec12412despiked.nc')

v2 = xr.open_dataset(dataPath + '/vec12414despiked.nc')

v3 = xr.open_dataset(dataPath + '/vec8155despiked.nc')

results = xr.open_dataset(dataPath + '/dissipations.nc')

#endregion

"""
This is just for creating a lot of plots. So, I am leaving it commented out
here for later use if wanted

#region plot full spectra (without the high wavenumbers removed)
def SpectraCalc(vec,name,segSize = 10, highLowSep = 1/5, turbVarCutoff = 5, waveCutoff = 1.025, unorigionalCutoff = 0.01,\
    lowWavenumberRemoval = 2, highSpectrumRemoval = 0, binSize = 50, genScale = 1/2, minDataPoints = 10,\
        minWavenumberSpan = 2.5, slopeConfidence = .95, peakProminence = 0.8, kdeBW=1):

    #Calculates the spectra for each burst of vector vec, without removing the lowest wavenumbers
    #So we can look at the white noise level off if it exists

    #pull burst numbers and sampling frequency
    bnum = vec.burst30.values
    hz = vec.attrs['Sampling Rate (Hz)']

    #pull time of each burst
    t = vec.avTime30.values

    #initialize variables
    temp = np.empty(bnum.size)*np.nan
    pressure = np.empty(bnum.size)*np.nan
    nu = np.empty(bnum.size)*np.nan

    uAv = np.empty(bnum.size)*np.nan
    uStd = np.empty(bnum.size)*np.nan

    ep = np.empty(bnum.size)*np.nan
    er = np.empty(bnum.size)*np.nan
    pv = np.empty(bnum.size)*np.nan
    kept = np.empty(bnum.size)*np.nan
    fitType = np.empty(bnum.size)*np.nan
    peaksThrown = np.empty(bnum.size)*np.nan
    UrmsSW = np.empty(bnum.size)*np.nan
    UrmsIG = np.empty(bnum.size)*np.nan

    k = [np.empty(0)]*bnum.size
    s = [np.empty(0)]*bnum.size
    dof = [np.empty(0)]*bnum.size
    epDist = [np.empty(0)]*bnum.size
    xfit = [np.empty(0)]*bnum.size
    yfit = [np.empty(0)]*bnum.size

    #Interpolate temperature to full time frequency
    tempAll = vec.Temp.interp(time_sen=vec.time)

    #loop through each burst
    for i in np.arange(bnum.size):

        print('burst ' + str(i+1) + ' of ' + str(bnum.size) + ', v' + name)

        #pull indices for the current burst
        ind = vec.bNum30.values == bnum[i]

        #get average temperature and pressure for this burst
        temp[i] = np.nanmean(tempAll.values[ind])
        pressure[i] = np.nanmean(vec.Pressure.values[ind])

        #get average velocity and velocity variance for thi burst
        uAv[i] = np.nanmean(vec.Primary.values[ind])
        uStd[i] = np.nanstd(vec.Primary.values[ind])

        #calculate average dynamic viscosity for this burst
        nuTemp = temp[i]+273.15
        nuPress = pressure[i]/100 + 0.101325
        nu[i] = iapws95.IAPWS95_PT(nuPress,nuTemp).nu

        #eliminate nans from each velocity component
        w = fn.nan_interp(vec.Up.values[ind])
        u = fn.nan_interp(vec.Primary.values[ind])
        v = fn.nan_interp(vec.Secondary.values[ind])

        #Calculate IG and SS root mean square velocities (making sure we don't have only nans first to avoid an error)
        if np.isnan(u)[0]:
            UrmsSW[i] = np.nan
            UrmsIG[i] = np.nan
        else:
            UrmsIG[i] = fn.bandVrms(u,hz,0.04,0.004)
            UrmsSW[i] = fn.bandVrms(u,hz,0.2,0.04)

        #store which vertical velocity points are original data for this burst
        orig = vec.UpOrig.values[ind]

        #calculate spectrum
        (k[i],s[i],dof[i],_,_,keptTemp) = fn.waveCorrectedSpectrum(w,u,v,orig,seconds=segSize,hz=hz,fullOut=True,useW=False,\
            highLowSep=highLowSep, turbVarCutoff = turbVarCutoff, waveCutoff = waveCutoff, unorigionalCutoff = unorigionalCutoff,\
                lowWavenumberRemoval = lowWavenumberRemoval, highSpectrumRemoval = highSpectrumRemoval, binSize = binSize)

    return k,s

def SpectraView(k,s,index,vec):
    #plots and saves the spectra stored in k and s at index i
    #vec is just for labelling
    plt.ioff()
    for i in np.where(index)[0]:
        print(vec+' '+str(i))
        plt.loglog(k[i],s[i])
        plt.savefig(dataPath + '/plots/whiteNoiseTest/whiteNoiseTest_'+vec+'_{0:d}.png'.format(i))
        plt.close()

#calculate spectra for each vector
(k1,s1) = SpectraCalc(v1,'12412')
(k2,s2) = SpectraCalc(v1,'12414')
(k3,s3) = SpectraCalc(v1,'8155')

#store location of non-nan spectra
i1 =  ~np.isnan(results.ep12412.values[0:np.size(k1)])
i2 =  ~np.isnan(results.ep12414.values[0:np.size(k2)])
i3 =  ~np.isnan(results.ep8155.values[0:np.size(k3)])   

#save plots of spectra
SpectraView(k1,s1,i1,'12412')
SpectraView(k2,s2,i2,'12414')
SpectraView(k3,s3,i3,'8155')
#endregion

"""

#region dissipation change calc

#calculate dissipations with flate noise subtracted from spectrum
print('v12412')
v1Results = fn.epCalc(v1,'12412',whiteNoiseRemove = 10**-9)
print('v12414')
v2Results = fn.epCalc(v2,'12414',whiteNoiseRemove = 10**-9)
print('v8155')
v3Results = fn.epCalc(v3,'8155',whiteNoiseRemove = 10**-9)

#combine instruments into one dataset
print('combining')
resultsWN2 = v1Results.merge(v2Results).merge(v3Results)

#save results
resultsWN2.to_netcdf(dataPath+'/dissipationsNoiseRemovedTest_10-9.nc')

#endregion

"""
resultsWN2 = xr.open_dataset(dataPath+'/dissipationsNoiseRemovedTest_10-9.nc')

#various ways of evaluating and plotting change that I am commenting out for now

er12412 = np.nanmax(np.vstack((results.statEr12412.values,results.sysEr12412.values)),0)
er12414 = np.nanmax(np.vstack((results.statEr12414.values,results.sysEr12414.values)),0)
er8155 = np.nanmax(np.vstack((results.statEr8155.values,results.sysEr8155.values)),0)

(np.nanmean(er12412/results.ep12412)+np.nanmean(er12414/results.ep12414)+np.nanmean(er8155/results.ep8155))/3

(np.nanmean(results.maxEr12412/results.ep12412)+np.nanmean(results.maxEr12414/results.ep12414)+np.nanmean(results.maxEr8155/results.ep8155))/3

np.nanmax((np.nanmax(results.maxEr12412/results.ep12412),np.nanmax(results.maxEr12414/results.ep12414),np.nanmax(results.maxEr8155/results.ep8155)))

np.sum(results.maxEr12412/results.ep12412>.5)+np.sum(results.maxEr12414/results.ep12414>.5)+np.sum(results.maxEr8155/results.ep8155>.5)

i12412 = results.maxEr12412/results.ep12412<.5
i12414 = results.maxEr12414/results.ep12414<.5
i8155 = results.maxEr8155/results.ep8155<.5


(np.nanmean(results.maxEr12412[i12412]/results.ep12412[i12412])+\
    np.nanmean(results.maxEr12414[i12414]/results.ep12414[i12414])+\
        np.nanmean(results.maxEr8155[i8155]/results.ep8155[i8155]))/3

(np.nanmean(er12412[i12412]/results.ep12412[i12412])+\
    np.nanmean(er12414[i12414]/results.ep12414[i12414])+\
        np.nanmean(er8155[i8155]/results.ep8155[i8155]))/3

np.nanmax([np.nanmax(results.maxEr12412[i12412]/results.ep12412[i12412]),\
    np.nanmax(results.maxEr12414[i12414]/results.ep12414[i12414]),\
        np.nanmax(results.maxEr8155[i8155]/results.ep8155[i8155])])

(np.nanmean(results.statEr12412[i12412]/results.ep12412[i12412])+\
    np.nanmean(results.statEr12414[i12414]/results.ep12414[i12414])+\
        np.nanmean(results.statEr8155[i8155]/results.ep8155[i8155]))/3


np.sum(np.abs(resultsWN2.ep12412-results.ep12412)>er12412)+\
np.sum(np.abs(resultsWN2.ep12414-results.ep12414)>er12414)+\
np.sum(np.abs(resultsWN2.ep8155-results.ep8155)>er8155)

np.sum(np.abs(resultsWN2.ep12412-results.ep12412)>results.maxEr12412)+\
np.sum(np.abs(resultsWN2.ep12414-results.ep12414)>results.maxEr12414)+\
np.sum(np.abs(resultsWN2.ep8155-results.ep8155)>results.maxEr8155)

#region evaluate and plot

np.sum(np.abs(resultsWN2.ep12412-results.ep12412)>results.er12412)+\
np.sum(np.abs(resultsWN2.ep12414-results.ep12414)>results.er12414)+\
np.sum(np.abs(resultsWN2.ep8155-results.ep8155)>results.er8155)

np.sum(np.abs(resultsWN.ep12412-results.ep12412)>results.er12412)+\
np.sum(np.abs(resultsWN.ep12414-results.ep12414)>results.er12414)+\
np.sum(np.abs(resultsWN.ep8155-results.ep8155)>results.er8155)

np.sum(~np.isnan(results.ep12412))+np.sum(~np.isnan(results.ep12414))+np.sum(~np.isnan(results.ep8155))

fig = plt.figure()
plt.loglog(results.ep12412,results.ep12412,'.',color='black')
plt.loglog(results.ep12414,results.ep12414,'.',color='black')
plt.loglog(results.ep8155,results.ep8155,'.',color='black')

plt.loglog(results.ep12412,results.ep12412+results.maxEr12412,'.',color='blue')
plt.loglog(results.ep12412,results.ep12412-results.maxEr12412,'.',color='blue')
plt.loglog(results.ep12414,results.ep12414+results.maxEr12414,'.',color='blue')
plt.loglog(results.ep12414,results.ep12414-results.maxEr12414,'.',color='blue')
plt.loglog(results.ep8155,results.ep8155+results.maxEr8155,'.',color='blue')
plt.loglog(results.ep8155,results.ep8155-results.maxEr8155,'.',color='blue')

plt.loglog(results.ep12412,resultsWN2.ep12412,'.',color='red')
plt.loglog(results.ep12414,resultsWN2.ep12414,'.',color='red')
plt.loglog(results.ep8155,resultsWN2.ep8155,'.',color='red')

fig = plt.figure()
plt.loglog(results.ep12412,results.ep12412,'.',color='black')
plt.loglog(results.ep12414,results.ep12414,'.',color='black')
plt.loglog(results.ep8155,results.ep8155,'.',color='black')

plt.loglog(results.ep12412,results.ep12412+results.er12412,'.',color='blue')
plt.loglog(results.ep12412,results.ep12412-results.er12412,'.',color='blue')
plt.loglog(results.ep12414,results.ep12414+results.er12414,'.',color='blue')
plt.loglog(results.ep12414,results.ep12414-results.er12414,'.',color='blue')
plt.loglog(results.ep8155,results.ep8155+results.er8155,'.',color='blue')
plt.loglog(results.ep8155,results.ep8155-results.er8155,'.',color='blue')

plt.loglog(results.ep12412,resultsWN.ep12412,'.',color='red')
plt.loglog(results.ep12414,resultsWN.ep12414,'.',color='red')
plt.loglog(results.ep8155,resultsWN.ep8155,'.',color='red')
#endregion
"""