"""
Code for plotting the time series of final dissipation values
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

import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import palettable
colors = palettable.cmocean.sequential.Ice_20.mpl_colors
color_cycle = [colors[0],colors[8],colors[16]]

#endregion

#region load data
print('loading data')

results = xr.open_dataset(dataPath+'/dissipations.nc')

v1 = xr.open_dataset(dataPath+'/vec12412despiked.nc')

#endregion

plt.ioff()

#region plot1

#make figure
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(15,7),sharex=True)

#plot dissipation results
ax1.semilogy(results.time,results.ep12412,'-o',color=color_cycle[0],fillstyle='none',label='Lower')
ax1.semilogy(results.time,results.ep8155,'-o',color=color_cycle[2],fillstyle='none',label='Upper')
#ax1.semilogy(results.time,results.ep12414,'-o',color=color_cycle[1],fillstyle='none',label='Upstream')

#ax1.errorbar(results.time.values,results.ep12412.values,results.maxEr12412.values,fmt='-o',fillstyle='none',color=color_cycle[1],label='Lower')
#ax1.errorbar(results.time.values,results.ep8155.values,results.maxEr8155.values,fmt='-o',fillstyle='none',color=color_cycle[2],label='Upper')
#ax1.set_yscale('log')

#labels
ax1.legend(loc='upper left',fontsize=16,ncol=2,borderpad=.2,columnspacing=.6,labelspacing=.3,borderaxespad=.2)
ax1.set_ylabel(r'Dissipation ($m^2s^{-3}$)',fontsize=16)
ax1.set_xticklabels('')

#plot velocity
ax2.plot(v1.time,v1.Primary,color=color_cycle[1])

#labels
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax2.set_ylabel('Along Channel Velocity (m/s)',fontsize=16)
ax2.set_xlabel('Time (month-day)',fontsize=16)

#scale tick labels
for tick in ax1.get_xticklabels():
    tick.set_fontsize(16)
for tick in ax1.get_yticklabels():
    tick.set_fontsize(16)

for tick in ax2.get_xticklabels():
    tick.set_fontsize(16)
for tick in ax2.get_yticklabels():
    tick.set_fontsize(16)

#title
ax1.set_title('Final Results',fontsize=24)

#more labels
ax1.text(0.98, 0.9,'(a)', ha='center', va='center', transform=ax1.transAxes,fontsize=16)
ax2.text(0.98, 0.9,'(b)', ha='center', va='center', transform=ax2.transAxes,fontsize=16)

#spacing adjustment
plt.subplots_adjust(left = .1,right = .95, top=.9, hspace=.1)

#endregion

plt.savefig(submissionPath+'/results.png',dpi=300)
#plt.savefig(plotPath+'/results.png')
plt.close()