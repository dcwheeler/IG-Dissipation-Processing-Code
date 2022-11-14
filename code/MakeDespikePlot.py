"""
Code for making plot of despiking algorithm results
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

# Add code base to be able to import Functions library
import sys
sys.path.append(functionPath)
import Functions as fn

#import numpy and xarray
import numpy as np
import xarray as xr

#import plotting libraries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.lines as mlines
from matplotlib.patches import ConnectionPatch

#import scipy libraries
from scipy import signal
from scipy import stats

#set colorbar
import palettable

colors = palettable.cmocean.sequential.Ice_20.mpl_colors
color_cycle = [colors[0],colors[8],colors[16]]
color_cycle2 = palettable.cmocean.sequential.Ice_12.mpl_colors[2:]

#endregion


#region load data
print('loading data')

data = xr.open_dataset(dataPath+'/vec12412raw.nc')
BPressure = xr.open_dataset(dataPath+'/met.nc').BarometricPressure

#endregion


#region time stamp management

print('keep only data before timestamp goes crazy')

#pull time values
t = data.time.values

#grab all values before first time time stamp skips more than 6 seconds
tnew = t[:np.where(np.diff(t)>np.timedelta64(6,'s'))[0][0]]

#grab all data new time stamp as a new dataset
data_temp = data.sel(time=tnew,time_start=slice(None,tnew[-1]),time_sen=slice(None,tnew[-1]))

print('assign burst numbers')
#creat a burstnumber vector corresponding to hour long bursts on the full time stamp
BurstNum = np.hstack([np.ones(data_temp.time.sel(time = slice(data_temp.time_start[i-1],\
    data_temp.time_start[i])).size-1)*i for i in data_temp.BurstCounter[:-1].values.astype('int')]+\
        [np.ones(data_temp.time.sel(time=slice(data_temp.time_start[-1],None)).size)*data_temp.BurstCounter[-1].values.astype('int')])

#save burst numbers in dataset
data_temp['BurstNum'] = ('time',BurstNum)

#calculate 30 minute burst values
data_temp = fn.avTime30(data_temp)

#endregion

#region select burst
print('select wanted burst')

#set burst number to plot
bnum = 54 #high IG wave energy burst for actual paper plot
#bnum = 457 #low IG wave energy burst for giving to reviewer

#pull data for corresponding burst in a new dataset
dtemp = data_temp.isel(time=np.where(data_temp.bNum30 == bnum)[0])

#endregion

#region process burst
print('processing burst')

print('cleaning vector 12412 based on correlation and snr cutoffs')

#initial cleaning
dtemp = fn.cleanVec(dtemp,corrCutoff=70,snrCutoff=10)

#reverse U velocity for vector 12412 so that positive is in the flooding direction
velocities = -dtemp.U.values

#pull sampling frequency
hz = dtemp.attrs['Sampling Rate (Hz)']

#set lowpass filter frequency
lp = 1/20

#endregion

#region despiking
print('despiking')

#perform standard expanded cutoff despiking algorithm
(uExp,detectedExp, limExp) = fn.despike_iterate(velocities,hz,lp,expand = True,plot=False,verbose=True)

#perform standard universal threshold based despiking algorithm
(uGau,detectedGau, limGau) = fn.despike_iterate(velocities,hz,lp,expand = False,plot=False,verbose=True)

#endregion

#region derivatives and limits
print('calculating distributions and limits')

#get rid of nans for now but save locations of nans
nanloc = np.where(np.isnan(velocities))
vel = fn.nan_interp(velocities.copy())

#find, save, and remove low pass signal
sos = signal.butter(4,lp,'lp',fs=hz,output='sos') 
low = signal.sosfiltfilt(sos,vel)
utest = uExp-low

#save size of burst
n = utest.size

#calculate derivative
du = np.empty(n)*np.nan
du[1:-1] = (utest[2:]-utest[0:-2])/2
du[0] = utest[1]-utest[0]
du[-1] = utest[-1]-utest[-2]

#calculate universal threshold
lamda_u = (np.sqrt(2*np.log(n)))*np.nanstd(utest)
lamda_du = (np.sqrt(2*np.log(n)))*np.nanstd(du)

#start cutoffs at universal threshold
a = lamda_u
b = lamda_du

#helper function for calculating density
def numIn(a,b):
    return np.sum((np.square(utest)/np.square(a)+np.square(du)/np.square(b))<=1)

#define step size for cutoff expansion
diffA = a*.01
diffB = b*.01

#calculate elliptical density betwen cutoffs and previous cutoffs
def localDensity(a,b):
    return (numIn(a,b)-numIn(a-diffA,b-diffB))/(np.pi*a*b-np.pi*(a-diffA)*(b-diffB))

#loop until elliptical density decreases by more than 95% of previous elliptical density
while (localDensity(a,b) - localDensity(a+diffA,b+diffB))/localDensity(a,b) < .95:
    a += diffA
    b += diffB

#endregion

#region distribution calc

#define x variable for gaussian pdf
gaux = np.linspace(np.nanmin(utest),np.nanmax(utest),1000)

#calculate gaussian pdf along gaux. Use mean and standard deviation of high frequency velocity (utest)
gauy = (1/(np.nanstd(utest)*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((gaux-np.nanmean(utest))/np.nanstd(utest))**2)

#endregion


plt.ioff()

#region plot
print('plotting')   

#save high frequency signal and size of velocity vector
u = vel-low
n = u.size

#return original nan's back to nan
u[nanloc] = np.nan

uExp[detectedExp] = np.nan
uGau[detectedGau] = np.nan
uExp[nanloc] = np.nan
uGau[nanloc] = np.nan

#create figure
fig = plt.figure(figsize=(13,8))

#define upper and lower axes for velocity histogram
ax_high = plt.subplot2grid((26,1),(0,0),rowspan=3,fig=fig)
ax_low = plt.subplot2grid((26,1),(3,0),rowspan =2, fig=fig)

#define final axes for velocity time series and zoom in
ax2 = plt.subplot2grid((18,40),(10,0),colspan=37,rowspan=4,fig=fig)
ax6 = plt.subplot2grid((18,40),(15,0),colspan=37,rowspan=4,fig=fig)

#define middle axes for phase space scatter plots
ax3 = plt.subplot(4,3,4)
ax4 = plt.subplot(4,3,5)
ax5 = plt.subplot(4,3,6)

#define histogram plotting function
def histplot(ax1):

    #plot histogram
    #sns.distplot(utest,label='Original Data',ax=ax1,color=color_cycle[0])
    sns.distplot(vel-low,label='Original Data',ax=ax1,color=color_cycle[0])

    #plot gaussian pdf
    ax1.plot(gaux,gauy,label='Gaussian',color=color_cycle[1])

    #plot vertical lines for each cutoff
    ax1.axvline(lamda_u,ls='--',color=color_cycle[1],label='Gaussian Cutoff',lw=3)
    ax1.axvline(a,ls='-',color=color_cycle[2],label='Expanded Cutoff',lw=3)
    ax1.axvline(-lamda_u,ls='--',color=color_cycle[1],lw=3)
    ax1.axvline(-a,ls='-',color=color_cycle[2],lw=3)

#plot histogram on upper and lower histogram axes
histplot(ax_low)
histplot(ax_high)

histLim = np.max(np.abs(vel-low))
ax_high.set_xlim(-histLim,histLim)
ax_low.set_xlim(-histLim,histLim)

#define at what y value the upper and lower axes connect
ysep = .2

#set corresponding axes limits
ax_low.set_ylim(0,ysep)
ax_high.set_ylim(ysep,15)
#ax_high.set_ylim(ysep,ax_high.get_ylim()[-1])

#set yticks for histogram
ax_high.set_yticks([5,10,15])
ax_low.set_yticks([0, .1, .2])

#hide xlabels of upper axis
ax_high.get_xaxis().set_visible(False)

#creat labels
ax_high.legend(fontsize=12,bbox_to_anchor=(1.014, 1), loc='upper left',borderaxespad=0.)
ax_low.set_xlabel('Velocity (m/s)',fontsize=16,labelpad=0)
ax_high.set_ylabel('Data Density',fontsize=16)
ax_low.set_ylabel('')
ax_high.yaxis.set_label_coords(-.075, 0.35)
ax_high.text(0.975, 0.8,'(a)', ha='center', va='center', transform=ax_high.transAxes,fontsize=16)

#create double axis for bottom plot
ax2_twin = ax2.twinx()

#plot velocity data
ln1 = ax2_twin.plot(dtemp.time,low,label='Low-pass Signal',lw=3,ls='--',color=color_cycle[0])
ln2 = ax2.plot(dtemp.time,u,label='High-pass Signal',lw=1,color=color_cycle[0])
ln3 = ax2.plot(dtemp.time,uExp-low,label='Expanded Limits',lw=1,color=color_cycle[2])
ln4 = ax2.plot(dtemp.time,uGau-low,label='Gaussian Limits',lw=1,color=color_cycle[1])

#create a list of lines for legend
leg = ln1+ln2+ln3+ln4
labels = [l.get_label() for l in leg]

#create legend
ax2_twin.legend(leg,labels,fontsize=12,bbox_to_anchor=(1.1, 0.25), loc='upper left',borderaxespad=0.)

#create labels and set axis limits
ax2.set_ylim(-.4,.4)
ax2_twin.set_ylim(-np.nanmax(np.abs(low))*1.01,np.nanmax(np.abs(low))*1.01)

ax2.set_xlim(np.min(dtemp.time.values),np.max(dtemp.time.values))
ax2_twin.set_xlim(np.min(dtemp.time.values),np.max(dtemp.time.values))

#draw horizontal line to mark 0 axis
ax2.axhline(0,lw=1,c='k')

ax2.set_zorder(ax2_twin.get_zorder()+1) # put ax in front of ax2
ax2.patch.set_visible(False) # hide the 'canvas'

#more labels
ax_high.set_title('Relative Performance of Different Cutoffs in Burst 54',fontsize=24)
ax2.text(0.975, 0.8,'(e)', ha='center', va='center', transform=ax2.transAxes,fontsize=16)

#change time format on bottom x axis
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

#set x tick locations
#ax2.set_xticks([np.datetime64('2020-02-03T11:05:00'),\
#                np.datetime64('2020-02-03T11:15:00'),np.datetime64('2020-02-03T11:25:00')])

ax2.set_xticks(dtemp.time.values[0]+[np.timedelta64(5,'m'),\
    np.timedelta64(15,'m'),np.timedelta64(25,'m')])

#define times for zoom in
#t1 = np.datetime64('2020-02-03T11:21:30')
#t2 = np.datetime64('2020-02-03T11:23:00')

#t1 = np.datetime64('2020-02-03T11:02:15')
#t2 = np.datetime64('2020-02-03T11:03:45')

t1 = dtemp.time.values[0]+np.timedelta64(135,'s')
t2 = t1 + np.timedelta64(90,'s')

#color background of zoom in section
ax2.axvspan(t1,t2,color='red',alpha=.3)

#pull data from zoom in section
zoomInd = np.where(np.logical_and(dtemp.time<=t2,dtemp.time>=t1))[0]
dtemp2 = dtemp.sel(time=slice(t1,t2))

#create twin axis for zoom in
ax6_twin = ax6.twinx()

#plot zoom in
ax6_twin.plot(dtemp2.time,low[zoomInd],label='Low-pass Signal',lw=3,ls='--',color=color_cycle[0])
ax6.plot(dtemp2.time,u[zoomInd],label='High-pass Signal',lw=1,color=color_cycle[0])
ax6.plot(dtemp2.time,uExp[zoomInd]-low[zoomInd],label='Expanded Limits',lw=1,color=color_cycle[2])
ax6.plot(dtemp2.time,uGau[zoomInd]-low[zoomInd],label='Gaussian Limits',lw=1,color=color_cycle[1])

#labels and limits
ax6.text(0.98, 0.8,'(f)', ha='center', va='center', transform=ax6.transAxes,fontsize=16)
ax6.set_ylabel('Velocity (m/s)',fontsize=16)
ax6.set_xlabel('Time (HH:MM:SS)',fontsize=16)
ax6.set_ylim(-.4,.4)
ax6_twin.set_ylabel('Low-Pass Velocity',fontsize=16,)
ax6_twin.set_ylim(-np.nanmax(np.abs(low))*1.01,np.nanmax(np.abs(low))*1.01)
ax6.axhline(0,lw=1,c='k')
ax6.yaxis.set_label_coords(-.0803, 1.4)
ax6_twin.yaxis.set_label_coords(1.06, 1.4)
ax6.set_xticks([np.datetime64('2020-02-03T11:02:20'),np.datetime64('2020-02-03T11:02:40'),\
    np.datetime64('2020-02-03T11:03:00'),np.datetime64('2020-02-03T11:03:40'),np.datetime64('2020-02-03T11:03:20')])

#hide back axis
ax6.set_zorder(ax6_twin.get_zorder()+1) # put ax in front of ax2
ax6.patch.set_visible(False) # hide the 'canvas'

#more limits and labels
ax6.set_xlim(np.min(dtemp2.time.values),np.max(dtemp2.time.values))
ax6_twin.set_xlim(np.min(dtemp2.time.values),np.max(dtemp2.time.values))
ax6.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

#draw connecting lines for zoom in
con1 = ConnectionPatch(xyA=(mdates.date2num(t1),0.2),coordsA=ax2.get_xaxis_transform(),xyB=(0,1),coordsB=ax6.transAxes,arrowstyle='-',color='red',alpha=.4)
fig.add_artist(con1)
con2 = ConnectionPatch(xyA=(mdates.date2num(t2),0.2),coordsA=ax2.get_xaxis_transform(),xyB=(1.0,1.0),coordsB=ax6.transAxes,arrowstyle='-',color='red',alpha=.4)
fig.add_artist(con2)


#calculate derivatives for phase space scatter plots
du = np.empty(n)*np.nan
du2 = np.empty(n)*np.nan
du[1:-1] = (u[2:]-u[0:-2])/2
du[0] = u[1]-u[0]
du[-1] = u[-1]-u[-2]
du2[1:-1] = (du[2:]-du[0:-2])/2
du2[0] = du[1]-du[0]
du2[-1] = du[-1]-du[-2]

#plot phase space scatter plots
fn.plotDespikePlane(u,du,limExp[0],limExp[1],0,r'u',r'u$^{(1)}$',ax3,color_cycle[0],color_cycle[2],histogram=False,size=1)
fn.plotDespikePlane(du,du2,limExp[2],limExp[3],0,r'u$^{(1)}$',r'u$^{(2)}$',ax4,color_cycle[0],color_cycle[2],histogram=False,size=1)
fn.plotDespikePlane(u,du2,limExp[4],limExp[5],limExp[6],r'u',r'u$^{(2)}$',ax5,color_cycle[0],color_cycle[2],histogram=False,size=1)

#plot extra limits on phase space scatter plots
fn.plotEllipse(limGau[0],limGau[1],0,ax3,color=color_cycle[1],ls='--')
fn.plotEllipse(limGau[2],limGau[3],0,ax4,color=color_cycle[1],ls='--')
fn.plotEllipse(limExp[4],limExp[5],limExp[6],ax5,color=color_cycle[2],label='Expanded Cutoff',ls='-')
fn.plotEllipse(limGau[4],limGau[5],limGau[6],ax5,color=color_cycle[1],label='Gaussian Cutoff',ls='--')

#labels
ax5.legend(fontsize=12,bbox_to_anchor=(1.05, 1), loc='upper left',borderaxespad=0.)
ax3.yaxis.set_label_coords(-.3, 0.5)
ax3.text(0.9, 0.8,'(b)', ha='center', va='center', transform=ax3.transAxes,fontsize=16)
ax4.text(0.9, 0.8,'(c)', ha='center', va='center', transform=ax4.transAxes,fontsize=16)
ax5.text(0.9, 0.8,'(d)', ha='center', va='center', transform=ax5.transAxes,fontsize=16)

#hide ticks on upper plot
ax_high.get_xaxis().set_ticks([])

#scale tick labels
for tick in ax_low.get_yticklabels():
    tick.set_fontsize(16)

for tick in ax_high.get_yticklabels():
    tick.set_fontsize(16)

for tick in ax_low.get_xticklabels():
    tick.set_fontsize(16)

for tick in ax2.get_xticklabels():
    tick.set_fontsize(16)
for tick in ax2.get_yticklabels():
    tick.set_fontsize(16)

for tick in ax2_twin.get_yticklabels():
    tick.set_fontsize(16)

for tick in ax3.get_xticklabels():
    tick.set_fontsize(16)
for tick in ax3.get_yticklabels():
    tick.set_fontsize(16)

for tick in ax4.get_xticklabels():
    tick.set_fontsize(16)
for tick in ax4.get_yticklabels():
    tick.set_fontsize(16)

for tick in ax5.get_xticklabels():
    tick.set_fontsize(16)
for tick in ax5.get_yticklabels():
    tick.set_fontsize(16)

for tick in ax6.get_xticklabels():
    tick.set_fontsize(16)
for tick in ax6.get_yticklabels():
    tick.set_fontsize(16)

for tick in ax6_twin.get_xticklabels():
    tick.set_fontsize(16)
for tick in ax6_twin.get_yticklabels():
    tick.set_fontsize(16)

#adjust plot spacing

plt.subplots_adjust(left = .08,right = .83, hspace=.7, wspace=.5)

#endregion

plt.savefig(submissionPath+'/despikeComparison.png',dpi=300)
#plt.savefig(plotPath+'/despikeComparison.png')
#plt.savefig(plotPath+'/despikeComparisonNoWaves.png')
plt.close()
