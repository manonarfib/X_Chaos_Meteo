import xarray as xr
from weatherbench2.regions import SliceRegion
from .config import ERA5_ZARR_PATH, YEARS, CHUNKS, INPUT_VARS, TARGET_VAR


import dask.array as da




def open_era5_region(path=ERA5_ZARR_PATH, years=YEARS):
    ds = xr.open_zarr(path, consolidated=True)
    ds = ds.sel(time=ds.time.dt.year.isin([int(y) for y in years]))


    europe_region = SliceRegion(
    lat_slice=slice(75, 35),
    lon_slice=[slice(347.5, 360), slice(0, 42.5)]
    )
    ds_europe, _ = europe_region.apply(ds, xr.ones_like(ds.isel(time=0)))


    available_inputs = [v for v in INPUT_VARS if v in ds_europe.data_vars]
    ds_inputs = ds_europe[available_inputs]
    ds_target = ds_europe[TARGET_VAR]


    ds_inputs = ds_inputs.chunk(CHUNKS)
    ds_target = ds_target.chunk(CHUNKS)


    return ds_inputs, ds_target




def compute_norms(ds_inputs):
    mean = {}
    std = {}
    for v in ds_inputs.data_vars:
        s = ds_inputs[v]
        mean[v] = s.mean(dim=("time", "latitude", "longitude")).compute().item()
        std[v] = s.std(dim=("time", "latitude", "longitude")).compute().item()
    return mean, std