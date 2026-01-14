import xarray as xr
from numcodecs import Blosc
from dask.diagnostics import ProgressBar
import os

SRC_ZARR = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
OUT_ZARR = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"

# TIME_RANGE = ("1980-01-01", "1980-01-0")
DTYPE = "float32"
# TIME_BLOCKS = [
    # ("1980-01-01", "1980-01-07"),
#     ("1980-01-05", "1980-01-10"),
    # ]

# import xarray as xr
# import numpy as np

# ds = xr.open_zarr(OUT_ZARR)

# times = ds.time.values
# n = len(times)

# chunk = 32
# n_full = (n // chunk) * chunk   # nombre de pas de temps valides
# last_valid_time = times[n_full - 1]

# print("Total times:", n)
# print("Full times:", n_full)
# print("Dernier timestamp valide:", last_valid_time)


from datetime import datetime, timedelta

def generate_time_blocks(start_date, end_date, days_per_block=8):
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)

    blocks = []
    current = start

    while current < end:
        next_block = current + timedelta(days=days_per_block-1)
        if next_block > end:
            break  # on ne crée PAS de bloc incomplet
        blocks.append((
            current.strftime("%Y-%m-%d"),
            next_block.strftime("%Y-%m-%d")
        ))
        current = next_block + timedelta(days=1)

    return blocks

# pour le test :
TIME_BLOCKS = generate_time_blocks(
    start_date="2020-01-01",
    # end_date="2018-01-01",
    end_date="2022-01-01",
    days_per_block=8
)

# pour le val :
# TIME_BLOCKS = generate_time_blocks(
#     start_date="2018-01-01",
#    end_date="2020-01-01",
#     days_per_block=8
# )

# TIME_BLOCKS = generate_time_blocks(
#     # start_date="1980-01-01",
#     start_date = "2007-09-16",
#     # end_date="2018-01-01",
#     end_date="2018-01-01",
#     days_per_block=8
# )

INPUT_VARS = [
    "tp_6h",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "10m_wind_speed",
    "2m_temperature",
    "boundary_layer_height",
    "mean_sea_level_pressure",
    "total_cloud_cover",
    "volumetric_soil_water_layer_1",
    "u1000", "u925", "u700", "u500",
    "v1000", "v925", "v700", "v500",
    "t1000", "t925", "t700", "t500",
    "rh1000", "rh925", "rh700", "rh500",
    "gws1000", "gws925", "gws700", "gws500",
    "vv1000", "vv925", "vv700", "vv500",
]

TARGET_VAR = "tp_6h"

# ======================
# WRITE OPTIMIZED ZARR
# ======================

first = True
if os.path.exists(OUT_ZARR):
                ds_out = xr.open_zarr(OUT_ZARR)
                existing_times = set(ds_out.time.values)
                print(existing_times)
# breakpoint()
for start, end in TIME_BLOCKS:
    print(f"Processing {start} → {end}")

    ds = xr.open_dataset(SRC_ZARR, engine="zarr", chunks={})
    ds = ds.sel(time=slice(start, end))
    # print("ds selectionné")
    # Europe
    ds_west = ds.sel(longitude=slice(347.5, 360), latitude=slice(72, 35))
    ds_east = ds.sel(longitude=slice(0, 42.5), latitude=slice(72, 35))
    ds = xr.concat([ds_west, ds_east], dim="longitude")
    # print("Europe ok")
    
    DEFAULT_VARS = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "10m_wind_speed",
        "2m_temperature",
        "boundary_layer_height",
        "mean_sea_level_pressure",
        "total_cloud_cover",
        "volumetric_soil_water_layer_1"
    ]
    
    tp_6h = (ds["total_precipitation_6hr"] * 1000.0).rename("tp_6h")
    
    ds2d = ds[DEFAULT_VARS]

    # Variables statiques
    lsm = ds["land_sea_mask"].broadcast_like(tp_6h).rename("lsm")
    geo = ds["geopotential_at_surface"].broadcast_like(tp_6h).rename("geo")
    soil = ds["soil_type"].broadcast_like(tp_6h).rename("soil")
    
    # Variables 3D à certains niveaux
    u1000 = ds["u_component_of_wind"].sel(level=1000).drop_vars("level").rename("u1000")
    u925 = ds["u_component_of_wind"].sel(level=925).drop_vars("level").rename("u925")
    u700 = ds["u_component_of_wind"].sel(level=700).drop_vars("level").rename("u700")
    u500 = ds["u_component_of_wind"].sel(level=500).drop_vars("level").rename("u500")
    v1000 = ds["v_component_of_wind"].sel(level=1000).drop_vars("level").rename("v1000")
    v925 = ds["v_component_of_wind"].sel(level=925).drop_vars("level").rename("v925")
    v700 = ds["v_component_of_wind"].sel(level=700).drop_vars("level").rename("v700")
    v500 = ds["v_component_of_wind"].sel(level=500).drop_vars("level").rename("v500")
    t1000 = ds["temperature"].sel(level=1000).drop_vars("level").rename("t1000")
    t925 = ds["temperature"].sel(level=925).drop_vars("level").rename("t925")
    t700 = ds["temperature"].sel(level=700).drop_vars("level").rename("t700")
    t500 = ds["temperature"].sel(level=500).drop_vars("level").rename("t500")
    rh1000 = ds["relative_humidity"].sel(level=1000).drop_vars("level").rename("rh1000")
    rh925 = ds["relative_humidity"].sel(level=925).drop_vars("level").rename("rh925")
    rh700 = ds["relative_humidity"].sel(level=700).drop_vars("level").rename("rh700")
    rh500 = ds["relative_humidity"].sel(level=500).drop_vars("level").rename("rh500")
    gws1000 = ds["geostrophic_wind_speed"].sel(level=1000).drop_vars("level").rename("gws1000")
    gws925 = ds["geostrophic_wind_speed"].sel(level=925).drop_vars("level").rename("gws925") 
    gws700 = ds["geostrophic_wind_speed"].sel(level=700).drop_vars("level").rename("gws700")
    gws500 = ds["geostrophic_wind_speed"].sel(level=500).drop_vars("level").rename("gws500")     
    vv1000 = ds["vertical_velocity"].sel(level=1000).drop_vars("level").rename("vv1000") 
    vv925 = ds["vertical_velocity"].sel(level=925).drop_vars("level").rename("vv925") 
    vv700 = ds["vertical_velocity"].sel(level=700).drop_vars("level").rename("vv700") 
    vv500 = ds["vertical_velocity"].sel(level=500).drop_vars("level").rename("vv500") 

    # Concaténer dans un seul dataset
    ds = xr.merge([tp_6h, ds2d, lsm, geo, soil, u1000, u925, u700, u500, v1000, v925, v700, v500, t1000, t925, t700, t500, rh1000, rh925, rh700, rh500, gws1000, gws925, gws700, gws500, vv1000, vv925, vv700, vv500])
    
    # ds["tp_6h"] = ds["total_precipitation_6hr"] * 1000.0

    # empilement
    X = (
        ds[INPUT_VARS]
        .to_array(dim="channel")
        .transpose("time", "channel", "latitude", "longitude")
        .astype("float32")
    )
    H = X.sizes["latitude"]
    W = X.sizes["longitude"]
    
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    encoding = {
        "X": {
            # "chunks": (32, len(INPUT_VARS), H, W),
            "compressor": compressor,
        },
        "Y": {
            # "chunks": (32, H, W),
            "compressor": compressor,
        },
    }
    
    X = X.chunk({"time": 32, "channel": len(INPUT_VARS), "latitude": H, "longitude": W})
    print("X ok")
    Y = ds[TARGET_VAR].astype("float32")
    Y = Y.chunk({"time": 32, "latitude": H, "longitude": W})

    ds = xr.Dataset({"X": X, "Y": Y})

    
    # with ProgressBar():
    if first:
        print("Nombre de pas de temps :", ds.sizes["time"])
        ds.to_zarr(
            OUT_ZARR,
            mode="w",
            encoding=encoding
        )
        first = False
        
    else:
        # if os.path.exists(OUT_ZARR):
        #     max_existing_time = ds_out.time.max().values
        #     print("Dernier timestamp existant :", max_existing_time)

        #     existing_times = set(ds_out.time.values)
        #     new_times = set(ds.time.values)

        #     overlap = existing_times & new_times
        #     # print(overlap)
        #     if overlap:
        #         raise ValueError(f"Overlap temporel détecté : {len(overlap)} timestamps")

        ds.to_zarr(
            OUT_ZARR,
            mode="a",
            append_dim="time"
        )
    print("ds_to_zarr ok")

print("Dataset complet prêt")