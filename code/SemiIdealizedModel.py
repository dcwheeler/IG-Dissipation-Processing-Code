"""
Code for testing spectra calculation with semi-idealized model
Saves SemiIdealizeResults
Save file used in MakeIdealizedErrorPlot.py
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
from scipy import signal
from scipy.interpolate import interp1d

#endregion

#region load data
print('loading data')

v1 = xr.open_dataset(dataPath + '/vec12412despiked.nc')

v2 = xr.open_dataset(dataPath + '/vec12414despiked.nc')

v3 = xr.open_dataset(dataPath + '/vec8155despiked.nc')

#endregion

#region define advective cases

def defineAdvective(vec,filtFreq = 1/5):
    #takes vector vec, and uses velocities from each burst
    #to define advection velocity cases for semi-idealized model

    #pull burst numbers
    bnum = vec.burst30.values

    #initialize arrays
    uReal = [np.empty(1)]*bnum.size
    uvar = np.empty(bnum.size)
    umean = np.empty(bnum.size)

    #pull sampling frequency
    hz = vec.attrs['Sampling Rate (Hz)']

    #define low pass filter
    sos = signal.butter(4,filtFreq,'lp',fs=hz,output='sos')

    #loop through bursts
    for i in np.arange(bnum.size):

        #pull velocities from corresponding burst and interpolate out nans
        utemp = fn.nan_interp(vec.Primary.values[vec.bNum30==bnum[i]])

        #store low pass filtered velocities as advection velocities
        uReal[i] = np.nanmean(utemp)+signal.sosfiltfilt(sos,utemp-np.nanmean(utemp))

        #calculate and store variance and means
        uvar[i] = np.var(uReal[i])
        umean[i] = np.mean(uReal[i])
    
    return uReal,uvar,umean

print('pulling velocities')

def pullVel(filtFreq):
    #helper function to get advection cases from each vector and combine into one large array

    #get advection cases for each vector
    uReal1,uvar1,umean1 = defineAdvective(v1,filtFreq = filtFreq)
    uReal2,uvar2,umean2 = defineAdvective(v2,filtFreq = filtFreq)
    uReal3,uvar3,umean3 = defineAdvective(v3,filtFreq = filtFreq)

    #combine results into large arrays
    uReal = uReal1+uReal2+uReal3
    uvar = np.concatenate((uvar1,uvar2,uvar3))
    umean = np.concatenate((umean1,umean2,umean3))

    return (uReal,uvar,umean)

#run above functions to define all advection cases
(uReal,uVar,uMean) = pullVel(1/5)

#endregion

#region define dissipations, constants, and set up placeholders
print('defining constants')

#define ideal dissipations
epsilons = np.array([10**-3,10**-4,10**-5,10**-6,10**-7,10**-8])

#set constants for ideal inertial subrange

L = 2                   #generation length scale for test case
sec = 10                #seconds for length of windows used to calculate corrected spectrum
dx = .0001              #desired spatial resoluition
T = 30*60               #total time of our test
Umax = .5
D = 4*T*Umax            #total distance of our test
hz=16

#define array to store burst numbers
bursts = np.arange(uVar.size)

#define array to store what instrument each burst comes from
instruments = np.concatenate((np.ones(np.size(v1.burst30))*int(v1.attrs['Hardware Serial Number'][4:]),\
    np.ones(np.size(v2.burst30))*int(v2.attrs['Hardware Serial Number'][4:]),\
        np.ones(np.size(v3.burst30))*int(v3.attrs['Hardware Serial Number'][4:]))).astype('int')

#define array to store the burst numbers as defined by each individual instrument
burstNums = np.concatenate((v1.burst30.values,v2.burst30.values,v3.burst30.values))

#initialize arrays
ep = np.empty((epsilons.size,bursts.size))*np.nan
er = np.empty(ep.shape)*np.nan
pv = np.empty(ep.shape)*np.nan
kept = np.empty(ep.shape)*np.nan
fitType = np.empty(ep.shape)*np.nan

erIdeal = np.empty(epsilons.shape)*np.nan
pvIdeal = np.empty(epsilons.shape)*np.nan

specIdeal = np.empty((epsilons.size,int(D/(2*dx))))*np.nan

epIdeal = np.empty(epsilons.shape)*np.nan

k = [ [np.empty(0)]*bursts.size for _ in range(epsilons.size) ]
spec = [ [np.empty(0)]*bursts.size for _ in range(epsilons.size) ]
dof = [ [np.empty(0)]*bursts.size for _ in range(epsilons.size) ]
epDist = [ [np.empty(0)]*bursts.size for _ in range(epsilons.size) ]
xfit = [ [np.empty(0)]*bursts.size for _ in range(epsilons.size) ]
yfit = [ [np.empty(0)]*bursts.size for _ in range(epsilons.size) ]

idealDist = [np.empty(1)]*epsilons.size
xIdeal = [np.empty(1)]*epsilons.size
yIdeal = [np.empty(1)]*epsilons.size

#endregion

#region loop through epsilons
for i in np.arange(epsilons.size):

    #region ideal spectrum and dataset

    print('defining ideal spectrum '+str(i+1)+' of ' + str(epsilons.size))

    #calculate ideal spectrum
    (kIdeal,specIdeal[i,:],kTemp,specTemp) = fn.IdealSpectrum(epsilons[i],L=L,dx=dx,T=T,Umax=Umax,fullOut=True)

    #fit a dissipation to the ideal spectrum
    (epIdeal[i],xIdeal[i],yIdeal[i],erIdeal[i],pvIdeal[i],idealDist[i],_,_,_) = fn.dissipationFit(kTemp[::10],\
        specTemp[::10], np.ones(kTemp[::10].shape)*100,nu=10**-6,L=1,eptest=epsilons[i]*10,outFull=True,\
            generationScaling = 1/2, sizeCutoff=2.5, peakLimit=.8, debug=False)

    print('generating ideal dataset')
    
    #use ideal spectrum with random phases to create ideal spatial dataset
    (x,wx) = fn.IdealData(kIdeal,specIdeal[i,:],dx=dx,T=T,Umax=Umax)
    
    #Pick a spot to make temporal observations
    loc = int(x.size/2)

    #set up interpolation
    winterp = interp1d(x,wx)

    #endregion

    #region loop through bursts
    for j in bursts:

        print(str(j+1)+' of '+str(bursts.size))

        #pull how many advection velocity data points are original (not replaced spikes)
        orig = np.ones(uReal[j].size,dtype='bool')

        print('Creating Temporal Datasets')

        #use advection velocities to determine observation locations
        xObs = fn.uToX(uReal[j],np.arange(uReal[j].size)/hz,x[loc])

        #sample from observation locations to generate temporal semi-ideal dataset
        wt = winterp(xObs)

        print('Calculating Spectrum')
        #calculate corrected spectrum
        #(k[i][j],spec[i][j],dof[i][j],_,_,keptTemp,_,_) = fn.waveCorrectedSpectrum(wt,uReal[j],np.zeros(uReal[j].size),\
        #    orig,seconds=sec,hz=hz,fullOut=True,useW=False)
        
        (k[i][j],spec[i][j],dof[i][j],_,_,keptTemp,_,_) = fn.waveCorrectedSpectrum(wt,uReal[j],np.zeros(uReal[j].size),\
            orig,seconds=sec,hz=hz,fullOut=True,useW=False,highSpectrumRemoval=10)

        #store how many of the individual 10 second spectra were kept before averaging
        kept[i,j] = np.sum(keptTemp)

        print('Calculating Dissipation')

        if np.logical_or(np.isnan(k[i][j][0]),np.size(k[i][j])<10):
            #if insufficient spectrum is calculated, store failure values
            ep[i,j] = np.nan
            er[i,j] = np.nan
            pv[i,j] = np.nan
            epDist[i][j] = np.empty(0)
            fitType[i,j] = 0
        else:
            #otherwise fit dissipation to calculated spectrum
            (ep[i,j],xfit[i][j],yfit[i][j],er[i,j],pv[i,j],epDist[i][j],_,fitType[i,j],_) = fn.dissipationFit(k[i][j],spec[i][j],dof[i][j],\
                L=L,eptest=10**-3,outFull=True, generationScaling = 1/4, sizeCutoff=2.5, peakLimit=.8, debug=False)

    #endregion

#endregion

#region save results

#helper functions to get correctly shaped arrays for putting into xarray dataset
def fillArray(array,size):
    return np.array([np.append(temp,np.empty(size-temp.size)*np.nan) for temp in array])
    
def fillArray2(array,size):
    return np.array([[np.append(temp2,np.empty(size-temp2.size)*np.nan) for temp2 in temp1] for temp1 in array])

def size(list):
    return np.max([np.size(temp) for temp in list])

def size2(list):
    return np.max([[np.size(temp2) for temp2 in temp1] for temp1 in list])

#get correct array sizes and create correctly sized arrays
kSize = size2(k)

knew = fillArray2(k,kSize)
dofnew = fillArray2(dof,kSize)
specnew = fillArray2(spec,kSize)

uSize = size(uReal)

unew = fillArray(uReal,uSize)

epSize = np.max([size2(epDist),size(idealDist)])

epDistNew = fillArray2(epDist,epSize)
idealDistNew = fillArray(idealDist,epSize)

fitSize = np.max([size2(xfit),size(xIdeal)])

xfitNew = fillArray2(xfit,fitSize)
xIdealNew = fillArray(xIdeal,fitSize)

yfitNew = fillArray2(yfit,fitSize)
yIdealNew = fillArray(yIdeal,fitSize)

#store results in xarray dataset
results = xr.Dataset()
results.coords['burst'] = bursts
results.coords['time'] = np.arange(uSize)*(1/hz)
results['uVar'] = ('burst',uVar)
results['uMean'] = ('burst',uMean)
results['uReal'] = (('burst','time'),unew)
results['instruments'] = ('burst',instruments)
results['burstNums'] = ('burst',burstNums)
results.coords['kIdeal'] = kIdeal
results.coords['epsilon'] = epsilons
results.coords['kNum'] = np.arange(kSize)
results.coords['epNum'] = np.arange(epSize)
results.coords['fitNum'] = np.arange(fitSize)
results['specIdeal'] = (('epsilon','kIdeal'),specIdeal)
results['epIdeal'] = ('epsilon',epIdeal)
results['k'] = (('epsilon','burst','knum'),knew)
results['spec'] = (('epsilon','burst','knum'),specnew)
results['dof'] = (('epsilon','burst','knum'),dofnew)
results['ep'] = (('epsilon','burst'),ep)
results['er'] = (('epsilon','burst'),er)
results['pv'] = (('epsilon','burst'),pv)
results['kept'] = (('epsilon','burst'),kept)
results['epDist'] = (('epsilon','burst','epNum'),epDistNew)
results['idealDist'] = (('epsilon','epNum'),idealDistNew)
results['erIdeal'] = ('epsilon',erIdeal)
results['pvIdeal'] = ('epsilon',pvIdeal)
results['fitType'] = (('epsilon','burst'),fitType)
results.fitType.attrs['notes'] = '0 = not a long enough spectrum to even try,1 = no good fits,\
                2 = no fits that satisfy their own disspation derived cutoffs, 3 = good fits'
results['xfit'] = (('epsilon','burst','fitNum'),xfitNew)
results['xIdeal'] = (('epsilon','fitNum'),xIdealNew)
results['yfit'] = (('epsilon','burst','fitNum'),yfitNew)
results['yIdeal'] = (('epsilon','fitNum'),yIdealNew)

print('saving results')
#save
results.to_netcdf(dataPath+'/SemiIdealizedResultsHighSpectrumRemoval10.nc') 

#endregion
