"""
Pipeline de préparation de datasets ERA5 (WeatherBench2) pour précipitation en Europe.

- Source : Zarr public Google Cloud WeatherBench2 / ERA5
- Cible : total_precipitation_6hr (mm / 6h)
- Entrées : u_component_of_wind, v_component_of_wind
- Domaine : Europe (lon [-12.5, 42.5], lat [72, 35] en coordonnées géographiques)
- Split temporel : train / val / test

Ce script laisse xarray/dask gérer le lazy loading. La réduction de taille réelle
se fait si l'on écrit les datasets en Zarr/NetCDF local.
"""

import xarray as xr
from typing import Tuple


# --------- CONFIG GLOBALE ---------

WB2_URL = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

# Domaine Europe (coordonnées géographiques, lon en [-180, 180])
LON_MIN_EU = -12.5
LON_MAX_EU = 42.5
LAT_MAX_EU = 72.0   # ERA5 a la latitude décroissante, donc slice(72, 35)
LAT_MIN_EU = 35.0

# Split temporel
TRAIN_START = "1980-01-01"
TRAIN_END   = "2015-12-31"
VAL_START   = "2016-01-01"
VAL_END     = "2018-12-31"
TEST_START  = "2019-01-01"
TEST_END    = "2020-12-31"


# --------- FONCTIONS UTILITAIRES ---------

def open_weatherbench2_era5(url: str = WB2_URL) -> xr.Dataset:
    """
    Ouvre le store Zarr ERA5 de WeatherBench2 en lazy.
    """
    ds = xr.open_zarr(url, consolidated=True)
    return ds


def recenter_longitudes(ds: xr.Dataset) -> xr.Dataset:
    """
    Convertit les longitudes de [0, 360) vers [-180, 180] et trie l'axe.

    Important pour que le slicing Europe + Cartopy / projections géographiques
    soient cohérents.
    """
    ds = ds.assign_coords(
        longitude=(((ds.longitude + 180) % 360) - 180)
    ).sortby("longitude")
    return ds


def subset_europe(ds: xr.Dataset) -> xr.Dataset:
    """
    Restreint le dataset à l'Europe (lon [-12.5, 42.5], lat [72, 35]).

    Attention : dans ERA5, la latitude décroît du Nord vers le Sud,
    donc on garde slice(LAT_MAX, LAT_MIN).
    """
    ds_eu = ds.sel(
        longitude=slice(LON_MIN_EU, LON_MAX_EU),
        latitude=slice(LAT_MAX_EU, LAT_MIN_EU),
    )
    return ds_eu


def select_variables(ds: xr.Dataset) -> xr.Dataset:
    """
    Ne garde que les variables nécessaires (entrées + cible).

    Cible :
        - total_precipitation_6hr  -> converti en mm / 6h

    Entrées :
        - u_component_of_wind      (3D : time, level, lat, lon)
        - v_component_of_wind
    """
    required_vars = [
        "total_precipitation_6hr",
        "u_component_of_wind",
        "v_component_of_wind",
    ]

    missing = [v for v in required_vars if v not in ds.variables]
    if missing:
        raise KeyError(f"Variables manquantes dans le dataset : {missing}")

    ds_sel = ds[required_vars].copy()

    # conversion m/6h -> mm/6h pour la cible (plus lisible)
    ds_sel["total_precipitation_6hr"] = ds_sel["total_precipitation_6hr"] * 1000.0
    ds_sel["total_precipitation_6hr"].attrs["units"] = "mm/6h"

    return ds_sel


def split_train_val_test(ds: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Découpe le dataset en train / val / test selon les dates définies en haut du fichier.
    """
    ds_train = ds.sel(time=slice(TRAIN_START, TRAIN_END))
    ds_val   = ds.sel(time=slice(VAL_START,   VAL_END))
    ds_test  = ds.sel(time=slice(TEST_START,  TEST_END))

    return ds_train, ds_val, ds_test


# --------- MAIN PIPELINE ---------

def build_datasets(
    out_prefix: str = "era5_europe_precip"
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Pipeline complet :
      1) ouverture WB2/ERA5
      2) recentrage longitudes
      3) découpage Europe
      4) sélection variables (features + cible)
      5) split train/val/test

    Args:
        save_to_disk: si True, écrit les datasets en Zarr dans le cwd.
        out_prefix: préfixe des dossiers Zarr créés.

    Returns:
        (ds_train, ds_val, ds_test)
    """
    print("Ouverture du dataset WeatherBench2 / ERA5...")
    ds = open_weatherbench2_era5()

    print("Recentrage des longitudes en [-180, 180]...")
    ds = recenter_longitudes(ds)

    print("Découpage du domaine Europe...")
    ds = subset_europe(ds)

    print("Sélection des variables (u, v, total_precipitation_6hr)...")
    ds = select_variables(ds)

    print("Split train / val / test...")
    ds_train, ds_val, ds_test = split_train_val_test(ds)

    print("Taille train :", {k: int(v) for k, v in ds_train.sizes.items()})
    print("Taille val   :", {k: int(v) for k, v in ds_val.sizes.items()})
    print("Taille test  :", {k: int(v) for k, v in ds_test.sizes.items()})

    # if save_to_disk:
    #     print("Écriture des datasets en Zarr local...")
    #     ds_train.to_zarr(f"{out_prefix}_train.zarr", mode="w", consolidated=True)
    #     ds_val.to_zarr(f"{out_prefix}_val.zarr",   mode="w", consolidated=True)
    #     ds_test.to_zarr(f"{out_prefix}_test.zarr", mode="w", consolidated=True)
    #     print("Sauvegarde terminée.")

    return ds_train, ds_val, ds_test


if __name__ == "__main__":
    # Exécution simple : on construit les trois datasets sans les sauvegarder
    build_datasets()