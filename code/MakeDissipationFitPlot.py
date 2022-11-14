"""
Code for creating a plot of the inertial subrange fitting process
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
import matplotlib.transforms as transforms
import seaborn as sns

from scipy import stats

import palettable
color_cycle = palettable.cmocean.sequential.Gray_3.mpl_colors
colorFace = palettable.cmocean.sequential.Gray_4.mpl_colors[1]
color_cycle = [color_cycle[0],color_cycle[1],color_cycle[2]]
colors2 = palettable.cmocean.sequential.Ice_20.mpl_colors
color_cycle2 = [colors2[0],colors2[8],colors2[16]]

#endregion

#region load data
print('loading data')

dissipations = xr.open_dataset(dataPath+'/dissipations.nc')

#endregion

#region search for fit examples
fitType = dissipations.fitType12412.values
er = dissipations.squareError12412.values
ep = dissipations.epDist12412.values
length = dissipations.fitLength12412.values

er[np.isnan(er)] = 10000000
er[fitType!=3] = 10000000

differences = (np.abs((ep[np.arange(ep.shape[0]),np.argsort(er)[:,1]] - ep[np.arange(ep.shape[0]),np.argsort(er)[:,0]])/ep[np.arange(ep.shape[0]),np.argsort(er)[:,1]]))
differences[np.isnan(differences)] = 0

np.argsort(differences)[-5:]

differences = (np.abs((ep[np.arange(ep.shape[0]),np.argsort(er/length)[:,1]] - ep[np.arange(ep.shape[0]),np.argsort(er/length)[:,0]])/ep[np.arange(ep.shape[0]),np.argsort(er/length)[:,1]]))
differences[np.isnan(differences)] = 0

np.argsort(differences)[-5:]

#endregion

#region define constants
C = (18/55)*1.5*(4/3) #inertial subrange scale
m = -5/3 #inertial subrange slope
kolScale = 60 #scale factor for kolmogorov scale based cutoff
genScale = 1/2 #scale factor for generation length scale based cutoff
L = 1 #estimated length scale (meter) - set to 1 for roughly 1 meter deep water
#endregion

#region select burst and prep plotting

print('select wanted burst')
#[282, 107, 381,   2, 148, 555, 285,   9, 159,   8]
#burst number and instrument to use

bnum1 = 555
bnum2 = 57
bnum3 = 54
instrument = '12412'

def plotBurst(bnum, instrument, lengthNorm, ax1, ax2, xlabel,ylabel,legend,disx,label1,label2,title,first):
    #grab spectrum values for selected burst and instrument
    k = dissipations['k'+instrument][bnum].values
    s = dissipations['s'+instrument][bnum].values

    #grab the fit results to that spectrum
    xfit = dissipations['xfit'+instrument][bnum].values
    yfit = dissipations['yfit'+instrument][bnum].values

    #grab the dissipation histogram information for that spectrum
    epDist = dissipations['epDist'+instrument][bnum].values

    if lengthNorm:
        squareError = dissipations['squareError'+instrument][bnum].values/dissipations['fitLength'+instrument][bnum].values
    else:
        squareError = dissipations['squareError'+instrument][bnum].values

    squareError[np.isnan(squareError)] = 1000

    ep1 = epDist[np.argsort(squareError)[0]]
    ep2 = epDist[np.argsort(squareError)[1]]

    yfit1 = (-5/3)*xfit + np.log10(np.exp(np.log((ep1**(2/3))*(18/55)*1.5*(4/3))))
    yfit2 = (-5/3)*xfit + np.log10(np.exp(np.log((ep2**(2/3))*(18/55)*1.5*(4/3))))


    #grab the dissipation and viscosity values for that burst
    ep = dissipations['ep'+instrument][bnum].values
    nu = dissipations['nu'+instrument][bnum].values

    #remove nans from dissipation histogram
    epClean = epDist[~np.isnan(epDist)]

    #estimate kolmogorov length scale
    eta = (nu**3/(ep))**(1/4)

    #store number of data points used to create histogram
    size = epClean.size

    #calculate kernel density estimate
    kernel = stats.gaussian_kde(epClean)

    #calculate minimum and max of the dissipation distribution
    epmin = np.min(epClean)
    epmax = np.max(epClean)

    #calculate various ranges of the dissipation distribution
    Q1 = np.quantile(epClean, 0.25)
    Q3 = np.quantile(epClean, 0.75)
    IQR = Q3 - Q1
    cube = np.cbrt(size)

    #use the above ranges to calculate the bandwith used to create the kernel density estimate
    bwidth = epClean.ptp()/np.ceil(epClean.ptp()/(2*IQR/cube))

    #use the bandwidth to calculate a scaling factor for the kde to match the histogram
    A = bwidth*size

    #create an x vector for later plotting of the kde
    epPlot = np.linspace(epmin,epmax,1000)

    print('plotting')   

    #plot histogram
    sns.distplot(epClean,ax=ax1,kde=False,norm_hist=False,color=colorFace)

    #plot kde
    ax1.plot(epPlot,kernel(epPlot)*A,color=color_cycle[1])

    #plot vertical line marking final dissipation value
    ax1.axvline(ep,color=color_cycle2[0],linewidth=3)
    ax1.axvline(ep1,color=color_cycle2[1],linewidth=2,ls=':')
    ax1.axvline(ep2,color=color_cycle2[2],linewidth=2,ls='--')
    #ax1.axvline(ep3,color='r')

    ax1.set_title(title,fontsize=16)

    #labels
    if xlabel:
        ax1.set_xlabel(r'Dissipation ($m^2s^{-3}$)',fontsize=16)
    if ylabel:
        ax1.set_ylabel('Counts',fontsize=16)
    ax1.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    #plot spectrum
    l1, = ax2.loglog(k/(2*np.pi),s*2*np.pi,color=color_cycle[1],label='Spectrum')

    #plot fit to spectrum
    l2, = ax2.loglog((10**xfit)/(2*np.pi),(10**yfit)*2*np.pi,color=color_cycle2[0],linewidth=3,label='KDE fit')
    l3, = ax2.loglog((10**xfit)/(2*np.pi),(10**yfit1)*2*np.pi,color=color_cycle2[1],linewidth=2,ls=':',label = 'Best Fit')
    l4, = ax2.loglog((10**xfit)/(2*np.pi),(10**yfit2)*2*np.pi,color=color_cycle2[2],linewidth=2,ls='--',label = 'Second Best Fit')

    #plot upper and lower limits for location of inertial subrange
    ax2.axvline(1/(kolScale*eta),color='black')
    if first:
        ax2.text(.9/(kolScale*eta)*1.1,-.135,r'$1/(60*\eta) = {:.0f}$'.format(1/(kolScale*eta)),fontsize=16,horizontalalignment='center',transform=\
            transforms.blended_transform_factory(ax2.transData, ax2.transAxes))
    else:
        ax2.text(.9/(kolScale*eta)*1.1,-.135,r'{:.0f}'.format(1/(kolScale*eta)),fontsize=16,horizontalalignment='center',transform=\
            transforms.blended_transform_factory(ax2.transData, ax2.transAxes))
    ax2.axvline(1/(genScale*L),color='black')
    if first:
        ax2.text(.9/(genScale*L)*1.1,-.135,r'$2/L$ = {:.0f}'.format(2/L),fontsize=16,horizontalalignment='center',transform=\
            transforms.blended_transform_factory(ax2.transData, ax2.transAxes))        
    else:
        ax2.text(.9/(genScale*L)*1.1,-.135,r'{:.0f}'.format(2/L),fontsize=16,horizontalalignment='center',transform=\
            transforms.blended_transform_factory(ax2.transData, ax2.transAxes)) 
    if legend:
        leg1 = ax2.legend([l1,l2], ['Spectrum','KDE fit'], fontsize=16, loc='lower left', framealpha=1,borderpad=0,borderaxespad=.3,facecolor='white',edgecolor='white')
        ax2.legend([l3,l4],['Best Fit','Second Best Fit'], fontsize=16,loc=(.4,0), framealpha=0,borderpad=0,borderaxespad=.3)
        ax2.add_artist(leg1)
        #ax2.legend(fontsize=16,loc = 'lower left',ncol=2,framealpha=1,borderpad=0,labelspacing=0.1,borderaxespad=.3,facecolor='white',edgecolor='white',columnspacing=1)

    #labels
    if xlabel:
        ax2.set_xlabel(r'1/wavelength ($m^{-1}$)',fontsize=16)
    if ylabel:
        ax2.set_ylabel(r'Sww ($m^3s^{-2}$)',fontsize=16)

    #more labels
    ax1.text(.02, 1.1,label1, ha='center', va='center', transform=ax1.transAxes,fontsize=16,horizontalalignment='left')
    #ax1.text(disx,0.9,r'$\epsilon$ = {0:.3e}'.format(ep), ha='center',va='center',transform=ax1.transAxes,fontsize=16)
    ax2.text(.02, 1.1,label2, ha='center', va='center', transform=ax2.transAxes,fontsize=16,horizontalalignment='left')

    #scale axis ticklabels
    for tick in ax1.get_xticklabels():
        tick.set_fontsize(16)
    for tick in ax1.get_yticklabels():
        tick.set_fontsize(16)

    for tick in ax2.get_xticklabels():
        tick.set_fontsize(16)
    for tick in ax2.get_yticklabels():
        tick.set_fontsize(16)

#endregion

plt.ioff()

#region plot

fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize=(13,10))

plotBurst(bnum1,instrument,False,ax1,ax2,False,True,False,0.3,'(a)','(b)','Burst 555: Minimum Error',True)
plotBurst(bnum2,instrument,True,ax3,ax4,False,False,False,0.6,'(c)','(d)','Burst 57: Minimum Average Error',False)
plotBurst(bnum3, instrument,True,ax5,ax6,True,False,True,0.6,'(e)','(f)','Burst 54: Minimum Average Error',False)

#title
#fig.suptitle('Inertial Subrange Fit',fontsize=24)
plt.subplots_adjust(left = .08, right = .97, top = .9, wspace=.25,hspace=.4)

#endregion

plt.savefig(submissionPath+'/dissipationFit.png',dpi=300)
#plt.savefig(plotPath+'/dissipationFit.png')
plt.close()
