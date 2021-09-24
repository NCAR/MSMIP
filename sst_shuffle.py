#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
import random
# from dask.distributed import Client, LocalCluster
# client = Client(n_workers=1,threads_per_worker=100,processes=False)

import warnings
warnings.simplefilter("ignore")

# # User input

# directory with input data
diri = "/maloney-scratch/joedhsu/proj1/data/sst_locking/"

# input file name.  code assumes full time seires in single file
fili = "SPCCSM3.TS.daily.0004-0023.nc"

# directory for output data
diro = "/maloney-scratch/joedhsu/proj1/data/sst_locking/"

#  output filename prefix (usually includes model and/or simulation details)
Center = "NCAR"

#  output filename prefix (usually includes model and/or simulation details)
Model = "SPCCSM"

# time coordinate variable name
timeName = "time" # units should be similar to "days since YYYY-MM-DD" or "hours since YYYY-MM-DD-HH:MM"

# longitude coordinate variable name
lonName = "lon"

# latitude coordinate variable name
latName = "lat"

#  surface temperature or equivalent variable name
varName = "TS"

#  land surface included?  Enter True or False (no quotation marks)
landData = True

#  skin temperature or foundation temperature?  enter "skin" or "foundation" (inlcude quotes)
sstType = "skin"

#  number of days per year (typically 365, sometimes 360)
DaysPerYear = 365 # some models have 360-day years.  Use 365 if Leap Days included


# # Data IO
ds = xr.open_dataset(diri+fili,use_cftime=True)


# ----- skin or foundation SST?
if sstType != "skin" and sstType != "foundation" :
    print("********* ABORT.  sstType is case-sensitive")
    print("          Must be 'skin' or 'foundation'")

if sstType == "foundation" :
    ds[varName] = ds[varName] - 0.2 # estimate skin SST from foundation SST (C. A Clayson, personal communication)


# # Land masking
from ocean_mask import ocn_mask

da_omask_regrid = ocn_mask(ds[varName])
da_lmask_regrid = da_omask_regrid.where(da_omask_regrid.notnull(),other=2)-1
da_lmask_regrid = da_lmask_regrid.where(da_lmask_regrid==1, other=0)
ds['%s_ocn'%varName] = ds[varName]*da_omask_regrid


# # Calculate background signal
print("============")
print("calculating background signal")
import lanczos_filter as lf

ds['%s_ocn_mean'%varName] = ds['%s_ocn'%varName].mean(dim=timeName)
ds['%s_ocn_nomean'%varName] = ds['%s_ocn'%varName]-ds['%s_ocn_mean'%varName]
ds['%s_ocn_bg'%varName] = lf.lanczos_low_pass(ds['%s_ocn_nomean'%varName],
                                              201,
                                              1/100.,
                                              dim=timeName,
                                              opt='symm')+ds['%s_ocn_mean'%varName]
ds['%s_ocn_anom'%varName] = ds['%s_ocn'%varName]-ds['%s_ocn_bg'%varName]


# make sure dimension order
ds = ds.transpose(latName,lonName,timeName)
ds['dayofyear']=ds['%s.dayofyear'%timeName]


newindex_list = []
for i in range(1,DaysPerYear+1):
    newindex_list.append(np.where(ds['dayofyear']==i)[0])

# # Random Pattern
print("============")
print("generating random pattern 1d")
da_randpatt = ds['%s_ocn_anom'%varName].copy()*np.nan
for i in range(DaysPerYear):
    dayindex = np.copy(newindex_list[i])
    random.shuffle(dayindex)
    da_randpatt[:,:,newindex_list[i]] = ds['%s_ocn_anom'%varName][:,:,dayindex].values


# # Random single point
print("============")
print("generating random single point 1d")
da_randpt = ds['%s_ocn_anom'%varName].copy()*np.nan
for i in range(len(ds['%s'%lonName])):
#     print("swapping pointwise on lon index %i"%i)
    for j in range(len(ds['%s'%latName])):
        if da_omask_regrid[j,i].notnull():
            for ii in range(DaysPerYear):
                dayindex = np.copy(newindex_list[ii])
                random.shuffle(dayindex)
                da_randpt[j,i,newindex_list[ii]] = ds['%s_ocn_anom'%varName][j,i,dayindex].values

# # Random 5days
print("============")
print("generating random pattern 5d")
da_randpatt5days = ds['%s_ocn_anom'%varName].copy()*np.nan
for i in range(0,DaysPerYear,5):
    dayindex = np.copy(newindex_list[i])
    random.shuffle(dayindex)
    for newind,oldind in enumerate(dayindex):
        da_randpatt5days[:,:,newindex_list[i][newind]:newindex_list[i][newind]+5] \
        = ds['%s_ocn_anom'%varName][:,:,oldind:oldind+5].values


# total signal
ds['RandPatt1d'] = (da_randpatt+ds['%s_ocn_bg'%varName])
ds['RandPt1d'] = (da_randpt+ds['%s_ocn_bg'%varName])
ds['RandPatt5d'] = (da_randpatt5days+ds['%s_ocn_bg'%varName])

# # Putting Land points back
if landData :
    ds['RandPatt1d'] = ds['RandPatt1d'].where(ds['RandPatt1d'].notnull(),other=0.)
    ds['RandPt1d'] = ds['RandPt1d'].where(ds['RandPt1d'].notnull(),other=0.)
    ds['RandPatt5d'] = ds['RandPatt5d'].where(ds['RandPatt5d'].notnull(),other=0.)

    ds['RandPatt1d'] = da_lmask_regrid*ds['%s'%varName]+ds['RandPatt1d']
    ds['RandPt1d'] = da_lmask_regrid*ds['%s'%varName]+ds['RandPt1d']
    ds['RandPatt5d'] = da_lmask_regrid*ds['%s'%varName]+ds['RandPatt5d']


# # Output file
print("============")
print("output file....")

# CGCM output
ds_output = xr.Dataset()
ds_output['TS_CGCM_bg'] = ds['%s_ocn_bg'%varName]
ds_output['TS_CGCM_anom'] = ds['%s_ocn_anom'%varName]
ds_output['TS_CGCM'] = ds['%s'%varName]
ds_output.to_netcdf('%s%s.%s.TS.CGCM.0004-0023.nc'%(diro,Center,Model))
print("file at %s%s.%s.TS.CGCM.0004-0023.nc"%(diro,Center,Model))

# AGCM_1dRandPatt
ds_output = xr.Dataset()
ds_output['TS_AGCM_1dRandPatt_bg'] = ds['%s_ocn_bg'%varName]
ds_output['TS_AGCM_1dRandPatt_anom'] = da_randpatt
ds_output['TS_AGCM_1dRandPatt'] = ds['RandPatt1d']
ds_output.to_netcdf('%s%s.%s.TS.AGCM_1dRandPatt.0004-0023.nc'%(diro,Center,Model))
print("file at %s%s.%s.TS.AGCM_1dRandPatt.0004-0023.nc"%(diro,Center,Model))

# AGCM_5dRandPatt
ds_output = xr.Dataset()
ds_output['TS_AGCM_5dRandPatt_bg'] = ds['%s_ocn_bg'%varName]
ds_output['TS_AGCM_5dRandPatt_anom'] = da_randpatt5days
ds_output['TS_AGCM_5dRandPatt'] = ds['RandPatt5d']
ds_output.to_netcdf('%s%s.%s.TS.AGCM_5dRandPatt.0004-0023.nc'%(diro,Center,Model))
print("file at %s%s.%s.TS.AGCM_5dRandPatt.0004-0023.nc"%(diro,Center,Model))

# AGCM_1dRandPt
ds_output = xr.Dataset()
ds_output['TS_AGCM_1dRandpt_bg'] = ds['%s_ocn_bg'%varName]
ds_output['TS_AGCM_1dRandpt_anom'] = da_randpt
ds_output['TS_AGCM_1dRandPt'] = ds['RandPt1d']
ds_output.to_netcdf('%s%s.%s.TS.AGCM_1dRandPt.0004-0023.nc'%(diro,Center,Model))
print("file at %s%s.%s.TS.AGCM_1dRandPt.0004-0023.nc"%(diro,Center,Model))



# # Demo plots
print("============")
print("generating demo plot ....")

import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

fig = plt.figure(2,figsize=(20,10))
devy = 0.5
dlevel = np.arange(-1, 1+0.01, 0.1)
timeindex = np.arange(0,10)


######################################## plotting ############################################
for nindex,timeindex in enumerate(timeindex):

    #### original ####
    ax2 = fig.add_axes([0,0-nindex*devy-0.4,1,0.4],projection=ccrs.PlateCarree(central_longitude=180))
    im = ds['%s_ocn_anom'%varName].isel(time=timeindex)\
           .plot.pcolormesh(x='lon',
                            y='lat',
                            ax=ax2,
                            levels=dlevel,
                            extend='both',
                            cmap='RdBu_r',
                            transform=ccrs.PlateCarree(central_longitude=0))

    cb=im.colorbar
    cb.remove()
    ax2.coastlines(resolution='110m',linewidths=0.8)
    ax2.add_feature(cfeature.LAND,color='lightgrey')

    ax2.set_xticks([60,120,180,240,300], crs=ccrs.PlateCarree())
    ax2.set_xticklabels([60,120,180,-120,-60], color='black', weight='bold',size=22)
    ax2.set_yticks([-80,-60,-40,-20,0,20,40,60,80], crs=ccrs.PlateCarree())
    ax2.set_yticklabels([-80,-60,-40,-20,0,20,40,60,80], color='black', weight='bold',size=22)
    ax2.yaxis.tick_left()

    ax2.text(0.02,0.05, 'Time = %s'%ds['%s_ocn_anom'%varName].time.isel(time=timeindex).values,
             fontsize=24, color='k', weight='bold', transform=ax2.transAxes)


    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
#     ax2.grid(linewidth=2, color='black', alpha=0.3, linestyle='--')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    if nindex == 0:
        ax2.set_title('Original anomaly', color='black', weight='bold',size=35,pad=20)
    else:
        ax2.set_title('', color='black', weight='bold',size=35,pad=20)
#     ax2.set_aspect('auto')


    #### random pattern ####
    ax2 = fig.add_axes([0+0.55,0-nindex*devy-0.4,1,0.4],projection=ccrs.PlateCarree(central_longitude=180))
    im = da_randpatt.isel(time=timeindex)\
           .plot.pcolormesh(x='lon',
                            y='lat',
                            ax=ax2,
                            levels=dlevel,
                            extend='both',
                            cmap='RdBu_r',
                            transform=ccrs.PlateCarree(central_longitude=0))

    cb=im.colorbar
    cb.remove()
    ax2.coastlines(resolution='110m',linewidths=0.8)
    ax2.add_feature(cfeature.LAND,color='lightgrey')

    ax2.set_xticks([60,120,180,240,300], crs=ccrs.PlateCarree())
    ax2.set_xticklabels([60,120,180,-120,-60], color='black', weight='bold',size=22)
    ax2.set_yticks([-80,-60,-40,-20,0,20,40,60,80], crs=ccrs.PlateCarree())
    ax2.set_yticklabels([-80,-60,-40,-20,0,20,40,60,80], color='black', weight='bold',size=22)
    ax2.yaxis.tick_left()

    ax2.text(0.02,0.05, 'Time = %s'%ds['%s_ocn_anom'%varName].time.isel(time=timeindex).values,
             fontsize=24, color='k', weight='bold', transform=ax2.transAxes)


    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
#     ax2.grid(linewidth=2, color='black', alpha=0.3, linestyle='--')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    if nindex == 0:
        ax2.set_title('Random pattern', color='black', weight='bold',size=35,pad=20)
    else:
        ax2.set_title('', color='black', weight='bold',size=35,pad=20)
#     ax2.set_aspect('auto')

    #### random pattern 5days ####
    ax2 = fig.add_axes([0+0.55*2,0-nindex*devy-0.4,1,0.4],projection=ccrs.PlateCarree(central_longitude=180))
    im = da_randpatt5days.isel(time=timeindex)\
           .plot.pcolormesh(x='lon',
                            y='lat',
                            ax=ax2,
                            levels=dlevel,
                            extend='both',
                            cmap='RdBu_r',
                            transform=ccrs.PlateCarree(central_longitude=0))

    cb=im.colorbar
    cb.remove()
    ax2.coastlines(resolution='110m',linewidths=0.8)
    ax2.add_feature(cfeature.LAND,color='lightgrey')

    ax2.set_xticks([60,120,180,240,300], crs=ccrs.PlateCarree())
    ax2.set_xticklabels([60,120,180,-120,-60], color='black', weight='bold',size=22)
    ax2.set_yticks([-80,-60,-40,-20,0,20,40,60,80], crs=ccrs.PlateCarree())
    ax2.set_yticklabels([-80,-60,-40,-20,0,20,40,60,80], color='black', weight='bold',size=22)
    ax2.yaxis.tick_left()

    ax2.text(0.02,0.05, 'Time = %s'%ds['%s_ocn_anom'%varName].time.isel(time=timeindex).values,
             fontsize=24, color='k', weight='bold', transform=ax2.transAxes)


    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
#     ax2.grid(linewidth=2, color='black', alpha=0.3, linestyle='--')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    if nindex == 0:
        ax2.set_title('Random pattern 5days', color='black', weight='bold',size=35,pad=20)
    else:
        ax2.set_title('', color='black', weight='bold',size=35,pad=20)
#     ax2.set_aspect('auto')

    #### random pointwise ####
    ax2 = fig.add_axes([0+0.55*3,0-nindex*devy-0.4,1,0.4],projection=ccrs.PlateCarree(central_longitude=180))
    im = da_randpt.isel(time=timeindex)\
           .plot.pcolormesh(x='lon',
                            y='lat',
                            ax=ax2,
                            levels=dlevel,
                            extend='both',
                            cmap='RdBu_r',
                            transform=ccrs.PlateCarree(central_longitude=0))

    cb=im.colorbar
    cb.remove()
    ax2.coastlines(resolution='110m',linewidths=0.8)
    ax2.add_feature(cfeature.LAND,color='lightgrey')

    ax2.set_xticks([60,120,180,240,300], crs=ccrs.PlateCarree())
    ax2.set_xticklabels([60,120,180,-120,-60], color='black', weight='bold',size=22)
    ax2.set_yticks([-80,-60,-40,-20,0,20,40,60,80], crs=ccrs.PlateCarree())
    ax2.set_yticklabels([-80,-60,-40,-20,0,20,40,60,80], color='black', weight='bold',size=22)
    ax2.yaxis.tick_left()

    ax2.text(0.02,0.05, 'Time = %s'%ds['%s_ocn_anom'%varName].time.isel(time=timeindex).values,
             fontsize=24, color='k', weight='bold', transform=ax2.transAxes)


    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
#     ax2.grid(linewidth=2, color='black', alpha=0.3, linestyle='--')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    if nindex == 0:
        ax2.set_title('Random pointwise', color='black', weight='bold',size=35,pad=20)
    else:
        ax2.set_title('', color='black', weight='bold',size=35,pad=20)
#     ax2.set_aspect('auto')



cbaxes=fig.add_axes([0+0.4,0-nindex*devy-0.6,2,0.03])
cbar=fig.colorbar(im,cax=cbaxes,orientation='horizontal')
cbar.set_ticks(dlevel)
cbar.set_ticklabels(["%0.2f"%(n) for n in dlevel])
cbar.ax.tick_params(labelsize=40,rotation=45)
cbar.set_label(label='Surface temperature anomaly',size=45, labelpad=15)



fig.savefig('SST_locking_anom_global.pdf', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format=None,
                transparent=False, bbox_inches="tight", pad_inches=None)




# fig1 = plt.figure(3,figsize=(20,10))
# devy = 0.5
# dlevel = np.arange(230, 320+0.01, 5)
# timeindex = np.arange(0,10)


# ######################################## plotting ############################################
# ax2 = None
# for nindex,timeindex in enumerate(timeindex):

#     #### original ####
#     ax2 = fig1.add_axes([0,0-nindex*devy-0.4,1,0.4],projection=ccrs.PlateCarree(central_longitude=180))
#     im = ds['%s'%varName].isel(time=timeindex)\
#            .plot.pcolormesh(x='lon',
#                             y='lat',
#                             ax=ax2,
#                             levels=dlevel,
#                             extend='both',
#                             cmap='hot_r',
#                             transform=ccrs.PlateCarree(central_longitude=0))

#     cb=im.colorbar
#     cb.remove()
#     ax2.coastlines(resolution='110m',linewidths=0.8)
# #     ax2.add_feature(cfeature.LAND,color='lightgrey')

#     ax2.set_xticks([60,120,180,240,300], crs=ccrs.PlateCarree())
#     ax2.set_xticklabels([60,120,180,-120,-60], color='black', weight='bold',size=22)
#     ax2.set_yticks([-80,-60,-40,-20,0,20,40,60,80], crs=ccrs.PlateCarree())
#     ax2.set_yticklabels([-80,-60,-40,-20,0,20,40,60,80], color='black', weight='bold',size=22)
#     ax2.yaxis.tick_left()

#     ax2.text(0.02,0.05, 'Time = %s'%ds['%s_ocn_anom'%varName].time.isel(time=timeindex).values,
#              fontsize=24, color='k', weight='bold', transform=ax2.transAxes)


#     lon_formatter = cticker.LongitudeFormatter()
#     lat_formatter = cticker.LatitudeFormatter()
#     ax2.xaxis.set_major_formatter(lon_formatter)
#     ax2.yaxis.set_major_formatter(lat_formatter)
# #     ax2.grid(linewidth=2, color='black', alpha=0.3, linestyle='--')
#     ax2.set_xlabel('')
#     ax2.set_ylabel('')
#     if nindex == 0:
#         ax2.set_title('Original anomaly', color='black', weight='bold',size=35,pad=20)
#     else:
#         ax2.set_title('', color='black', weight='bold',size=35,pad=20)
# #     ax2.set_aspect('auto')


#     #### random pattern ####
#     ax2 = fig1.add_axes([0+0.55,0-nindex*devy-0.4,1,0.4],projection=ccrs.PlateCarree(central_longitude=180))
#     im = ds['RandPatt1d'].isel(time=timeindex)\
#            .plot.pcolormesh(x='lon',
#                             y='lat',
#                             ax=ax2,
#                             levels=dlevel,
#                             extend='both',
#                             cmap='hot_r',
#                             transform=ccrs.PlateCarree(central_longitude=0))

#     cb=im.colorbar
#     cb.remove()
#     ax2.coastlines(resolution='110m',linewidths=0.8)
# #     ax2.add_feature(cfeature.LAND,color='lightgrey')

#     ax2.set_xticks([60,120,180,240,300], crs=ccrs.PlateCarree())
#     ax2.set_xticklabels([60,120,180,-120,-60], color='black', weight='bold',size=22)
#     ax2.set_yticks([-80,-60,-40,-20,0,20,40,60,80], crs=ccrs.PlateCarree())
#     ax2.set_yticklabels([-80,-60,-40,-20,0,20,40,60,80], color='black', weight='bold',size=22)
#     ax2.yaxis.tick_left()

#     ax2.text(0.02,0.05, 'Time = %s'%ds['%s_ocn_anom'%varName].time.isel(time=timeindex).values,
#              fontsize=24, color='k', weight='bold', transform=ax2.transAxes)


#     lon_formatter = cticker.LongitudeFormatter()
#     lat_formatter = cticker.LatitudeFormatter()
#     ax2.xaxis.set_major_formatter(lon_formatter)
#     ax2.yaxis.set_major_formatter(lat_formatter)
# #     ax2.grid(linewidth=2, color='black', alpha=0.3, linestyle='--')
#     ax2.set_xlabel('')
#     ax2.set_ylabel('')
#     if nindex == 0:
#         ax2.set_title('Random pattern', color='black', weight='bold',size=35,pad=20)
#     else:
#         ax2.set_title('', color='black', weight='bold',size=35,pad=20)
# #     ax2.set_aspect('auto')

#     #### random pattern 5days ####
#     ax2 = fig1.add_axes([0+0.55*2,0-nindex*devy-0.4,1,0.4],projection=ccrs.PlateCarree(central_longitude=180))
#     im = ds['RandPatt5d'].isel(time=timeindex)\
#            .plot.pcolormesh(x='lon',
#                             y='lat',
#                             ax=ax2,
#                             levels=dlevel,
#                             extend='both',
#                             cmap='hot_r',
#                             transform=ccrs.PlateCarree(central_longitude=0))

#     cb=im.colorbar
#     cb.remove()
#     ax2.coastlines(resolution='110m',linewidths=0.8)
# #     ax2.add_feature(cfeature.LAND,color='lightgrey')

#     ax2.set_xticks([60,120,180,240,300], crs=ccrs.PlateCarree())
#     ax2.set_xticklabels([60,120,180,-120,-60], color='black', weight='bold',size=22)
#     ax2.set_yticks([-80,-60,-40,-20,0,20,40,60,80], crs=ccrs.PlateCarree())
#     ax2.set_yticklabels([-80,-60,-40,-20,0,20,40,60,80], color='black', weight='bold',size=22)
#     ax2.yaxis.tick_left()

#     ax2.text(0.02,0.05, 'Time = %s'%ds['%s_ocn_anom'%varName].time.isel(time=timeindex).values,
#              fontsize=24, color='k', weight='bold', transform=ax2.transAxes)


#     lon_formatter = cticker.LongitudeFormatter()
#     lat_formatter = cticker.LatitudeFormatter()
#     ax2.xaxis.set_major_formatter(lon_formatter)
#     ax2.yaxis.set_major_formatter(lat_formatter)
# #     ax2.grid(linewidth=2, color='black', alpha=0.3, linestyle='--')
#     ax2.set_xlabel('')
#     ax2.set_ylabel('')
#     if nindex == 0:
#         ax2.set_title('Random pattern 5days', color='black', weight='bold',size=35,pad=20)
#     else:
#         ax2.set_title('', color='black', weight='bold',size=35,pad=20)
# #     ax2.set_aspect('auto')

#     #### random pointwise ####
#     ax2 = fig1.add_axes([0+0.55*3,0-nindex*devy-0.4,1,0.4],projection=ccrs.PlateCarree(central_longitude=180))
#     im = ds['RandPt1d'].isel(time=timeindex)\
#            .plot.pcolormesh(x='lon',
#                             y='lat',
#                             ax=ax2,
#                             levels=dlevel,
#                             extend='both',
#                             cmap='hot_r',
#                             transform=ccrs.PlateCarree(central_longitude=0))

#     cb=im.colorbar
#     cb.remove()
#     ax2.coastlines(resolution='110m',linewidths=0.8)
# #     ax2.add_feature(cfeature.LAND,color='lightgrey')

#     ax2.set_xticks([60,120,180,240,300], crs=ccrs.PlateCarree())
#     ax2.set_xticklabels([60,120,180,-120,-60], color='black', weight='bold',size=22)
#     ax2.set_yticks([-80,-60,-40,-20,0,20,40,60,80], crs=ccrs.PlateCarree())
#     ax2.set_yticklabels([-80,-60,-40,-20,0,20,40,60,80], color='black', weight='bold',size=22)
#     ax2.yaxis.tick_left()

#     ax2.text(0.02,0.05, 'Time = %s'%ds['%s_ocn_anom'%varName].time.isel(time=timeindex).values,
#              fontsize=24, color='k', weight='bold', transform=ax2.transAxes)


#     lon_formatter = cticker.LongitudeFormatter()
#     lat_formatter = cticker.LatitudeFormatter()
#     ax2.xaxis.set_major_formatter(lon_formatter)
#     ax2.yaxis.set_major_formatter(lat_formatter)
# #     ax2.grid(linewidth=2, color='black', alpha=0.3, linestyle='--')
#     ax2.set_xlabel('')
#     ax2.set_ylabel('')
#     if nindex == 0:
#         ax2.set_title('Random pointwise', color='black', weight='bold',size=35,pad=20)
#     else:
#         ax2.set_title('', color='black', weight='bold',size=35,pad=20)
# #     ax2.set_aspect('auto')



# cbaxes=fig1.add_axes([0+0.4,0-nindex*devy-0.6,2,0.03])
# cbar=fig1.colorbar(im,cax=cbaxes,orientation='horizontal')
# cbar.set_ticks(dlevel)
# cbar.set_ticklabels(["%0.2f"%(n) for n in dlevel])
# cbar.ax.tick_params(labelsize=40,rotation=45)
# cbar.set_label(label='Surface temperature',size=45, labelpad=15)



# fig1.savefig('SST_locking_total_global.pdf', dpi=300, facecolor='w', edgecolor='w',
#                 orientation='portrait', format=None,
#                 transparent=False, bbox_inches="tight", pad_inches=None)
