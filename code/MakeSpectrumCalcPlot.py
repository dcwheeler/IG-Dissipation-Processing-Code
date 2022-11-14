"""
Code plotting the results of the wave corrected spectrum results on the semi-idealized model.
@author: Duncan Wheeler
"""

#before running, make sure that functionPath points to where Functions.py is located, dataPath points to where the data files are located,
#plotPath points to where the plots are located, and tolerancePath points to where the tolerance test data is located.

functionPath = "/data/tidalpower/LosPen/IGexperiment2020/MethodsPaper/code"
dataPath = '/data/tidalpower/LosPen/IGexperiment2020/MethodsPaper/data'
plotPath = '/data/tidalpower/LosPen/IGexperiment2020/MethodsPaper/plots'
tolerancePath = '/data/tidalpower/LosPen/IGexperiment2020/MethodsPaper/toleranceTests'
submissionPath = '/data/tidalpower/LosPen/IGexperiment2020/MethodsPaper/submission'

#region setup
print('importing libraries')

# always import python debugger in case an error occurs!!!
import pdb

# Add my code base to be able to import InstrumentLoader library
import sys
sys.path.append(functionPath)
import Functions as fn

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

from scipy import signal
from scipy.interpolate import interp1d

import palettable
colors = palettable.cmocean.sequential.Ice_20.mpl_colors
color_cycle = [colors[0],colors[8],colors[16]]

#endregion

#region load data

v1 = xr.open_dataset(dataPath+'/vec12412despiked.nc')

#endregion

#region setup

#select burst number
bnum = 54

#pull sampling frequency
hz = v1.attrs['Sampling Rate (Hz)']

#create low pass filter to isolate high and low frequencies
sos = signal.butter(4,1/5,'lp',fs=hz,output='sos')

#interpolate nans on primary velocity of selected burst
utemp = fn.nan_interp(v1.Primary.values[v1.bNum30==bnum])

#obtain advection velocities using the low pass filter
uReal = np.nanmean(utemp)+signal.sosfiltfilt(sos,utemp-np.nanmean(utemp))

#pul which data points are original data (as opposed to replaced spikes)
orig = v1.UOrig[v1.bNum30==bnum]

#define an initial ideal dissipation
epsilon = 10**-5

#set constants for ideal inertial subrange

L = 2                   #generation length scale for test case
sec = 10                #seconds for length of windows used to calculate corrected spectrum
dx = .0001              #desired spatial resoluition
T = 30*60               #total time of our test
Umax = .5
D = 4*T*Umax            #total distance of our test

#endregion

#region create data
print('defining ideal spectrum')

#create ideal inertial spectrum
(kIdeal,specIdeal,kTemp,specTemp) = fn.IdealSpectrum(epsilon,L=L,dx=dx,T=T,Umax=Umax,fullOut=True)

print('Creating Temporal Datasets')

#use random phases to create ideal spatial turbulent dataset
(x,wx) = fn.IdealData(kIdeal,specIdeal,dx=dx,T=T,Umax=Umax)

#create interpolation function to use for measurements
winterp = interp1d(x,wx)

#set sampling location
loc = int(x.size/2)  

#use propagation velocities to determine sampling points from ideal data
xObs = fn.uToX(uReal,np.arange(uReal.size)/hz,x[loc])

#sample at corresponding points to generate temporal observations
wt = winterp(xObs)

#endregion

#region calculate spectra
print('Calculating Spectra')

#all generated observations are original data points
orig = np.ones(uReal.size,dtype='bool')

#calculate wave corrected spectrum
(k,spec,dof,kall,specall,keptTemp,_,_) = fn.waveCorrectedSpectrum(wt,uReal,np.zeros(uReal.size),\
    orig,seconds=sec,hz=hz,fullOut=True,useW=False)

#set segment number for calculating spectrum normally
segnum = 30

#calculate normal spectrum
(freqNorm,specNorm) = fn.specSegnum(wt,segnum,hz,plot=False,errorReturn=False,filt=False,wind=True)

#use frozen turbulence hypothesis to convert to spatial spectrum
specNormK = np.abs(specNorm*np.nanmean(uReal)/(2*np.pi))
kNorm = np.abs(2*np.pi*freqNorm/np.nanmean(uReal))

#calculate wave correction constant
specCorrectionConstant = fn.specCorrectionValue(np.nanstd(uReal),np.nanmean(uReal))

#endregion

plt.ioff()

#region plot

#create figure
fig,ax = plt.subplots(1,1,figsize=(10,8))

#plot individual 10 second wavenumber spectra that are averaged to produce wave corrected spectrum
for i in np.arange(np.size(kall[:-1])):
    plt.loglog(kall[i]/(2*np.pi),specall[i]*2*np.pi,color=color_cycle[1],alpha=.2)

#plot last individual 10 second spectrum so we have a label for the legend
plt.loglog(kall[-1]/(2*np.pi),specall[-1]*2*np.pi,color=color_cycle[1],alpha=.2,label='10 Second Spectra (sec. 4d)')

#plot initial ideal spectrum
plt.loglog(kTemp/(2*np.pi),specTemp*2*np.pi,color=color_cycle[0],lw=3,label='Ideal Spectrum (sec. 4b)')

plt.loglog(kTemp/(2*np.pi),(2*np.pi)*0.65*epsilon**(2/3)*(kTemp)**(-5/3),ls='--',c='k',label='-5/3 slope')

#plot normal wave correction method spectrum
plt.loglog(kNorm/(2*np.pi),specNormK*2*np.pi/specCorrectionConstant,color=color_cycle[2],lw=3,label='Wave Correction Constant (sec. 4c)')

#plot average wave corrected spectrum
plt.loglog(k/(2*np.pi),spec*2*np.pi,color=color_cycle[1],lw=3,label='Average Corrected Spectrum (sec. 4d)')

#plot generation scale
plt.axvline(2/L,ls=':',c='k')
plt.text(.9*2/L,10**-7,r'$2/L$ = {:.2f}'.format(2/L),fontsize=16,horizontalalignment='right')

#plot komogorov scale
eta = ((10**-6)**(3)/epsilon)**(1/4)*60
plt.axvline(1/(eta),ls=':',c='k')
plt.text(1.1*1/eta,10**-5,r'$1/(60*\eta) = {:.2f}$'.format(1/(eta)),fontsize=16)

#labels
plt.title('Wave Corrected Spectra Tests',fontsize=24)
plt.ylabel(r'Vertical Velocity Power Spectrum ($m^3s^{-2}$)',fontsize=16)
plt.xlabel(r'1/Wavelength ($m^{-1}$)',fontsize=16)

#scale tick labels
for tick in ax.get_xticklabels():
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    tick.set_fontsize(16)

#limits
plt.xlim(.2,2*10**2)
plt.ylim(.5*10**-8,2*10**-3)

#legend
plt.legend(fontsize=16,loc='upper right')

#endregion

plt.savefig(submissionPath+'/spectrumCalc.png',dpi=300)
#plt.savefig(plotPath+'/spectrumCalc.png')
plt.close()