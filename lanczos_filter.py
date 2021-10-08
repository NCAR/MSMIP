import numpy as np
import datetime
import xarray as xr

def lanczos_low_pass_weights(window, cutoff):
    """
    Calculate weights for a low pass Lanczos filter.

    Inputs:
    ================
    window: int
        The length of the filter window (odd number).

    cutoff: float
        The cutoff frequency(1/cut off time steps)

    """
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
#     sigma = 1.   # edit for testing to match with Charlotte ncl code
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    return w[1:-1]

def lanczos_low_pass(da_ts, window, cutoff, dim='time', opt='symm'):
    
    wgts = lanczos_low_pass_weights(window, cutoff)
    weight = xr.DataArray(wgts, dims=['window_dim'])
    
    if opt == 'symm':
        # create symmetric front 
        da_ts = da_ts.transpose('lat','lon','time')
        da_front = (xr.DataArray(da_ts.loc[
                                    dict(time=slice("%0.4i-01-01"%da_ts['time.year'][0],
                                                    "%0.4i-12-31"%da_ts['time.year'][0]))].values,
                                dims=['lat','lon','time'],
                                coords=dict(lat=da_ts.lat.values,
                                            lon=da_ts.lon.values,
                                            time=da_ts.loc[
                                                dict(time=slice("%0.4i-01-01"%da_ts['time.year'][0],
                                                                "%0.4i-12-31"%da_ts['time.year'][0]))].time.values
                                                                -datetime.timedelta(days=365)))
                   )
        da_front = da_front.reindex(time=list(reversed(da_front.time.values)))
        
        # create symmetric end
        da_end = (xr.DataArray(da_ts.loc[
                                  dict(time=slice("%0.4i-01-01"%da_ts['time.year'][-1],
                                                  "%0.4i-12-31"%da_ts['time.year'][-1]))].values,
                                dims=['lat','lon','time'],
                                coords=dict(lat=da_ts.lat.values,lon=da_ts.lon.values,
                                            time=da_ts.loc[
                                                dict(time=slice("%0.4i-01-01"%da_ts['time.year'][-1],
                                                                "%0.4i-12-31"%da_ts['time.year'][-1]))].time.values
                                                                +datetime.timedelta(days=365)))
                 )
        da_end = da_end.reindex(time=list(reversed(da_end.time.values)))
        
        da_symm = xr.concat([da_front,da_ts,da_end],dim='time')
        da_symm_filtered = da_symm.rolling({dim:window}, center=True).construct('window_dim').dot(weight)
        da_ts_filtered = da_symm_filtered.sel(time=da_ts.time)
        
    else:
        da_ts_filtered = da_ts.rolling({dim:window}, center=True).construct('window_dim').dot(weight)
    
    return da_ts_filtered
    
def lanczos_high_pass(da_ts, window, cutoff, dim='time'):
    
    da_ts_lowpass = lanczos_low_pass(da_ts, window, cutoff, dim='time')
    da_ts_filtered = da_ts-da_ts_lowpass
    
    return da_ts_filtered    

def lanczos_band_pass(da_ts, window, cutoff_low, cutoff_high, dim='time'):
    
    da_ts_filtered = lanczos_low_pass(da_ts, window, cutoff_high, dim='time')
    da_ts_filtered = lanczos_high_pass(da_ts_filtered, window, cutoff_low, dim='time')
    
    return da_ts_filtered