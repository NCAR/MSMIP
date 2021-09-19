import xesmf as xe
import xarray as xr

def ocn_mask(da_var):

    ds_mask = xr.open_dataset('data.nc')
    
    da_allbasin = ds_mask.isel(Z=0).basin
    da_omask = da_allbasin.where(da_allbasin.isnull(),other=1)

    # Regridding to the tracer points
    regridder_mask = xe.Regridder(da_omask,\
                                  da_var,\
                                  'bilinear',
                                  periodic=True)
    da_omask_regrid = regridder_mask(da_omask)
    
    return da_omask_regrid

    