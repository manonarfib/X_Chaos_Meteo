import xarray as xr
from weatherbench2.regions import SliceRegion
from .config import ERA5_ZARR_PATH, YEARS, CHUNKS, INPUT_VARS, TARGET_VAR

import dask.array as da


def open_era5_region(path=ERA5_ZARR_PATH, years=YEARS,
                     selected_levels=[500, 700, 925]):

    ds = xr.open_zarr(path, consolidated=True)
    if years != "all":
        ds = ds.sel(time=ds.time.dt.year.isin([int(y) for y in years]))

    europe_region = SliceRegion(
        lat_slice=slice(75, 35),
        lon_slice=[slice(347.5, 360), slice(0, 42.5)]
    )
    ds_europe, _ = europe_region.apply(ds, xr.ones_like(ds.isel(time=0)))

    available_inputs = [v for v in INPUT_VARS if v in ds_europe.data_vars]

    vars_with_levels = [v for v in available_inputs if "level" in ds_europe[v].dims]
    vars_without_levels = [v for v in available_inputs if "level" not in ds_europe[v].dims]


    ds_levels_expanded = expand_level_vars(ds_europe, vars_with_levels, selected_levels)
    ds_no_levels = ds_europe[vars_without_levels]

    ds_inputs = xr.merge([ds_no_levels, ds_levels_expanded])
    ds_target = ds_europe[TARGET_VAR]

    ds_inputs = ds_inputs.chunk(CHUNKS)
    ds_target = ds_target.chunk(CHUNKS)

    return ds_inputs, ds_target


def expand_level_vars(ds, vars_with_levels, selected_levels):
    """
    Pour chaque variable dans vars_with_levels, sélectionne certains levels et
    crée une variable par level sous forme var_level (sans dimension 'level').
    
    selected_levels : e.g. [500, 700, 850] (Pa hPa)
    """
    new_vars = {}

    for var in vars_with_levels:
        if var not in ds.data_vars:
            continue

        da_var = ds[var]

        # Vérifier que la variable a un level
        if "level" not in da_var.dims:
            continue

        for lev in selected_levels:
            # Sélectionner le niveau exact
            if lev not in da_var.level.values:
                raise ValueError(f"Level {lev} not found in variable {var}")

            # Extraire la variable au level choisi
            var_lev = da_var.sel(level=lev)

            # Nouveau nom : var_500, var_700, etc.
            new_name = f"{var}_{lev}"
            new_vars[new_name] = var_lev.drop_vars("level")

    return xr.Dataset(new_vars)




def compute_norms(ds_inputs):
    mean = {}
    std = {}
    for v in ds_inputs.data_vars:
        s = ds_inputs[v]
        mean[v] = s.mean(dim=("time", "latitude", "longitude")).compute().item()
        std[v] = s.std(dim=("time", "latitude", "longitude")).compute().item()
    return mean, std