"""
Code for plotting the results of the semi-idealized model based error analysis
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

import numpy as np
import xarray as xr
import pandas as pd

import glob

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from matplotlib import cm
from matplotlib import colors
import palettable
color = palettable.cmocean.sequential.Gray_20.mpl_colors
color_cycle = [color[0],color[8],color[16]]
cmap = colors.ListedColormap(palettable.cmocean.sequential.Gray_7_r.mpl_colors[1:])

#endregion

#region load data
print('loading data')

results = xr.open_dataset(dataPath+'/SemiIdealizedResults.nc')
#results2 = xr.open_dataset(dataPath+'/SemiIdealizedResultsTry2.nc')
#results = xr.open_dataset(dataPath+'/SemiIdealizedResultsHighSpectrumRemoval10.nc')
#results = xr.open_dataset(dataPath+'/SemiIdealizedResultsIdealOscillationAdvectionWaveCorrectedSpectrum2Min.nc')


#endregion

#region prep distributions

#pull indices of good fits
ind = results.fitType.values==3
#ind2 = results2.fitType.values==3

#calculate error of each dissipation fit
epErAll = ((results.ep.values.T-results.epsilon.values).T/results.ep.values)
#epErAll2 = ((results2.ep.values.T-results2.epsilon.values).T/results2.ep.values)

#pull only errors from the good fits
epEr = epErAll[ind]
#epEr = np.concatenate((epErAll[ind],epErAll2[ind2]))

#pull mean advection velocity
uMean = results.uMean.values[np.where(ind)[1]]
#uMean = np.concatenate((results.uMean.values[np.where(ind)[1]],results2.uMean.values[np.where(ind2)[1]]))

#pull initial ideal dissipation value
epC = results.epsilon.values[np.where(ind)[0]]
#epC = np.concatenate((results.epsilon.values[np.where(ind)[0]],results2.epsilon.values[np.where(ind2)[0]]))

#pull calculated dissipations
ep = results.ep.values[ind]
#ep = np.concatenate((results.ep.values[ind],results2.ep.values[ind2]))

#calculate average error for each ideal dissipation value
means = np.zeros(ind.shape[0])
for i in np.arange(means.size):
    means[i] = np.mean(epErAll[i,:][ind[i,:]])

#means2 = np.zeros(ind2.shape[0])
#for i in np.arange(means2.size):
#    means2[i] = np.mean(epErAll2[i,:][ind2[i,:]])

#means = (means+means2)/2

#calculate gaussian histogram of all dissipation errors
gau = stats.norm.rvs(np.mean(epEr),np.std(epEr),epEr.size)

#calculate x vector for gaussian kde
gaux = np.linspace(np.nanmin(epEr),np.nanmax(epEr),1000)

#calculate scaling factor for kde
Q1 = np.quantile(epEr, 0.25)
Q3 = np.quantile(epEr, 0.75)
IQR = Q3 - Q1
cube = np.cbrt(np.size(epEr))

methodEr = (np.quantile(epEr,0.975)-np.quantile(epEr,0.025))/2

bwidth = epEr.ptp()/np.ceil(epEr.ptp()/(2*IQR/cube))

A = bwidth*np.size(epEr)

#calculate scaled kde
gauy = A*(1/(np.nanstd(epEr)*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((gaux-np.nanmean(epEr))/np.nanstd(epEr))**2)

#set up logarithmic colorscale
norm= colors.LogNorm(vmin=results.epsilon[-1], vmax=results.epsilon[0])

#endregion

plt.ioff()

#region plot

#create figure
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,8))

#plot histogram
sns.histplot(epEr,ax=ax1,color=color_cycle[1],label='All Data',binwidth=bwidth,edgecolor=None,alpha=0.5)

#plot kde
ax1.plot(gaux,gauy,'--',color=color_cycle[0],label='Gaussian')

#plot vertical mean lines for each ideal dissipation
for i in np.arange(means.size):
    if i == int(means.size/2):
        ax1.axvline(means[i],label='Mean Error',color=cmap(norm(results.epsilon.values[i])))
    else:
        ax1.axvline(means[i],color=cmap(norm(results.epsilon.values[i])))

#plot vertical error lines for 16.3% error
ax1.axvline(methodEr,ls=':',color='k',label='{0:.1f}% Error'.format(methodEr*100))
ax1.axvline(-methodEr,ls=':',color='k')

#labels
ax1.legend(loc='upper left', fontsize=16)
ax1.set_ylabel('Counts',fontsize=16)
ax1.set_xlabel('Fractional Error', fontsize=16)

#make DataFrame for effective boxplot plotting
data = pd.DataFrame(data={'er':epEr,'speed':np.round(np.abs(uMean),decimals=1),'ep':epC})
#data = pd.DataFrame(data={'er':epEr,'speed':np.round(uMean,decimals=1),'ep':epC})

#create dictionary so that boxplot has correct colors
colorDic = dict(zip(np.unique(epC),cmap(norm(np.unique(epC)))))

#plot boxplot
sns.boxplot(x='speed',y='er',hue='ep',data=data,palette=colorDic,saturation=1,ax=ax2,fliersize=1)

#correct boxplot color defaults
for i,artist in enumerate(ax2.artists):
    # get the facecolor of the artist and get rid of black edges
    col = artist.get_facecolor()
    artist.set_edgecolor(col)

    # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
    # Loop over them here, and use the color from above
    for j in range(i*6,i*6+6):
        line = ax2.lines[j]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)
    #make median lines transparent
    ax2.lines[i*6+4].set_alpha(0)

#add in 0 line and error lines
ax2.axhline(0,color='k')
ax2.axhline(methodEr,color='k',ls=':')
ax2.axhline(-methodEr,color='k',ls=':')

#remove legend and set ylimits
ax2.get_legend().remove()
ax2.set_ylim(-0.4,0.3)

#labels
ax2.set_ylabel('Fractional Error', fontsize=16)
ax2.set_xlabel('Average Speed (m/s)',fontsize=16)

#scale axis tick labels
for tick in ax1.get_xticklabels():
    tick.set_fontsize(16)
for tick in ax1.get_yticklabels():
    tick.set_fontsize(16)

for tick in ax2.get_xticklabels():
    tick.set_fontsize(16)
for tick in ax2.get_yticklabels():
    tick.set_fontsize(16)

#title
ax1.set_title('Semi-Idealized Model Error',fontsize=24)

#more labels
ax1.text(0.965, 0.92,'(a)', ha='center', va='center', transform=ax1.transAxes,fontsize=16)
ax2.text(0.965, 0.92,'(b)', ha='center', va='center', transform=ax2.transAxes,fontsize=16)

#adjust spacing
plt.subplots_adjust(left = .12,right = .97, hspace=.3)

#make colorbar
cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ax=(ax1,ax2))
cb.set_label(r'Dissipation ($m^2s^{-3}$)',fontsize=16)

#scale color bar tick labels
for tick in cb.ax.get_yticklabels():
    tick.set_fontsize(16)

#endregion

plt.savefig(submissionPath+'/idealizeError.png',dpi=300)
#plt.savefig(plotPath+'/idealizeError.png')
plt.close()

