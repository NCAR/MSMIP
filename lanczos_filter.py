from typing import Hashable
import numpy as np
from pandas import DateOffset
import xarray as xr


def lanczos_low_pass_weights(window: int, cutoff: float) -> np.ndarray:
    """
    Calculate weights for a low pass Lanczos filter.
    Adapted from https://github.com/liv0505/Lanczos-Filter

    Duchon C. E. (1979) Lanczos Filtering in One and Two Dimensions.
    Journal of Applied Meteorology, Vol 18, pp 1016-1022.

    Parameters
    ----------
    window: int
        The length of the filter window (odd number).

    cutoff: float
        The cutoff frequency (1/cut off time steps)

    Returns
    -------
        w: numpy.ndarray
            array of Lanczos filter weights of size ``window``
    """
    if window % 2 != 1:
        raise ValueError("window width must be odd")
    w = np.zeros(window)
    n = window // 2 + 1
    w[n - 1] = 2.0 * cutoff
    k = np.arange(1, n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    norm = np.sin(2.0 * np.pi * cutoff * k) / (np.pi * k)
    w[: n - 1] = (norm * sigma)[::-1]
    w[n:] = norm * sigma
    return w


def lanczos_low_pass(
    da_ts: xr.DataArray,
    window: int,
    cutoff: float,
    dim: Hashable = "time",
    opt: str = "symm",
) -> xr.DataArray:
    """
    perform low-pass filtering of a timeseries as an :py:class:``xarray.DataArray`` using
    a Lanczos filter

    Parameters
    ----------
    da_ts : xarray.DataArray
        timeseries data to filter
    window : int
        the width of the filter (should be an odd number)
    cutoff : float
        the cutoff frequency for filtering
    dim : Hashable, optional
        the name of the temporal dimension of the data to filter, by default "time"
    opt : str, optional
        use "symm" to do symmetric filtering, otherwise filtering will be asymmetric, by
        default "symm"

    Returns
    -------
    xarray.DataArray
        the filtered timeseries data
    """

    wgts = lanczos_low_pass_weights(window, cutoff)
    weight = xr.DataArray(wgts, dims=["window_dim"])

    if opt == "symm":
        # create symmetric front
        da_ts = da_ts.transpose(..., dim)
        da_front = da_ts[..., da_ts[dim].dt.year == da_ts[dim].dt.year[0]]
        try:
            front_times = da_front[dim].to_index().to_datetimeindex()
        except AttributeError:
            front_times = da_front[dim].to_index()
        da_front = da_front.assign_coords(time=front_times - DateOffset(years=1))
        da_front = da_front.reindex({dim: da_front[dim][::-1]})

        # create symmetric end
        da_end = da_ts[..., da_ts[dim].dt.year == da_ts[dim].dt.year[-1]]
        try:
            end_times = da_end[dim].to_index().to_datetimeindex()
        except AttributeError:
            end_times = da_end[dim].to_index()
        da_end = da_end.assign_coords(time=end_times + DateOffset(years=1))
        da_end = da_end.reindex({dim: da_end[dim][::-1]})

        da_symm = xr.concat([da_front, da_ts, da_end], dim=dim)
        da_symm_filtered = (
            da_symm.rolling({dim: window}, center=True)
            .construct("window_dim")
            .dot(weight)
        )
        da_ts_filtered = da_symm_filtered.sel({dim: da_ts[dim]})

    else:
        da_ts_filtered = (
            da_ts.rolling({dim: window}, center=True)
            .construct("window_dim")
            .dot(weight)
        )

    return da_ts_filtered


def lanczos_high_pass(
    da_ts: xr.DataArray,
    window: int,
    cutoff: float,
    dim: Hashable = "time",
    opt: str = "symm",
) -> xr.DataArray:
    """
    perform high-pass filtering of a timeseries as an :py:class:``xarray.DataArray`` using
    a Lanczos filter

    Parameters
    ----------
    da_ts : xarray.DataArray
        timeseries data to filter
    window : int
        the width of the filter (should be an odd number)
    cutoff : float
        the cutoff frequency for filtering
    dim : Hashable, optional
        the name of the temporal dimension of the data to filter, by default "time"
    opt : str, optional
        use "symm" to do symmetric filtering, otherwise filtering will be asymmetric, by
        default "symm"

    Returns
    -------
    xarray.DataArray
        the filtered timeseries data
    """

    da_ts_lowpass = lanczos_low_pass(da_ts, window, cutoff, dim=dim, opt=opt)
    da_ts_filtered = da_ts.transpose(..., dim) - da_ts_lowpass

    return da_ts_filtered


def lanczos_band_pass(
    da_ts: xr.DataArray,
    window: int,
    cutoff_low: float,
    cutoff_high: float,
    dim: Hashable = "time",
    opt: str = "symm",
) -> xr.DataArray:
    """
    perform band-pass filtering of a timeseries as a :py:class:``xarray.DataArray`` using
    a Lanczos filter

    Parameters
    ----------
    da_ts : xarray.DataArray
        timeseries data to filter
    window : int
        the width of the filter (should be an odd number)
    cutoff_low : float
        the low cutoff frequency for filtering
    cutoff_high : float
        the high cutoff frequency for filtering
    dim : Hashable, optional
        the name of the temporal dimension of the data to filter, by default "time"
    opt : str, optional
        use "symm" to do symmetric filtering, otherwise filtering will be asymmetric, by
        default "symm"

    Returns
    -------
    xarray.DataArray
        the filtered timeseries data
    """

    da_ts_filtered = lanczos_low_pass(da_ts, window, cutoff_high, dim=dim, opt=opt)
    da_ts_filtered = lanczos_high_pass(
        da_ts_filtered, window, cutoff_low, dim=dim, opt=opt
    )

    return da_ts_filtered
