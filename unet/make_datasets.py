import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
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
TRAIN_END = "2015-12-31"
VAL_START = "2016-01-01"
VAL_END = "2018-12-31"
TEST_START = "2019-01-01"
TEST_END = "2020-12-31"

# --------- FONCTIONS UTILITAIRES ---------


def open_weatherbench2_era5(url: str = WB2_URL) -> xr.Dataset:
    """
    Ouvre le store Zarr ERA5 de WeatherBench2 en lazy.
    """
    ds = xr.open_zarr(url, consolidated=True, chunks='auto')
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
    Ne garde que les variables nécessaires (entrées + cible), en créant des
    variables dérivées au bon niveau :

    Cible :
        - total_precipitation_6hr -> 'tp' (mm/6h), dims: (time, lat, lon)

    Entrées :
        - 2m_temperature                -> 'temp_2m'             (time, lat, lon)
        - 10m_u_component_of_wind       -> 'u_comp_10'           (time, lat, lon)
        - 10m_v_component_of_wind       -> 'v_comp_10'           (time, lat, lon)
        - land_sea_mask                 -> 'land_sea_mask'       (lat, lon)
        - geopotential_at_surface       -> 'geo_surf'            (lat, lon)  
        - boundary_layer_height         -> 'bound_lay_height'    (time, lat, lon)    
        - geostrophic_wind_speed        -> 'geo_wind_speed'      (time, level, lat, lon)
        - mean_sea_level_pressure       -> 'mean_sea_lvl_press'  (time, lat, lon)
        - relative_humidity             -> 'rel_hum'             (time, level, lat, lon)
        - soil_type                     -> 'soil'                (time, lat, lon)
        - temperature                   -> 'temp'                (time, level, lat, lon)
        - vertical_velocity             -> 'vertical_velo'       (time, level, lat, lon)
        - total_cloud_cover             -> 'cloud_cov'           (time, lat, lon)
        - u_component_of_wind           -> 'u_comp'              (time, level, lat, lon)
        - v_component_of_wind           -> 'v_comp'              (time, level, lat, lon)
        - volumetric_soil_water_layer_1 -> 'vol_soil_layer1'     (time, lat, lon)  
    """

    # Variables sources nécessaires dans le dataset d'origine
    required_vars = [
        "total_precipitation_6hr",
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "land_sea_mask",
        "geopotential_at_surface",
        "boundary_layer_height",
        "geostrophic_wind_speed",
        "mean_sea_level_pressure",
        "relative_humidity",
        "soil_type",
        "temperature",
        "vertical_velocity",
        "total_cloud_cover",
        "u_component_of_wind",
        "v_component_of_wind",
        "volumetric_soil_water_layer_1"
    ]

    missing = [v for v in required_vars if v not in ds.variables]
    if missing:
        raise KeyError(f"Variables manquantes dans le dataset : {missing}")

    # --- Cible : précipitations sur 6h en mm/6h ---
    tp_6h = ds["total_precipitation_6hr"] * 1000.0
    tp_6h = tp_6h.rename("tp_6h")
    tp_6h.attrs["units"] = "mm/6h"

    # --- Entrées sans niveau vertical ---
    t_2m = ds["2m_temperature"].rename("t_2m")  # en K
    u_10 = ds["10m_u_component_of_wind"].rename("u_10")
    v_10 = ds["10m_v_component_of_wind"].rename("v_10")
    land_sea_mask = ds["land_sea_mask"].rename("land_sea_mask")
    geo_surf = ds["geopotential_at_surface"].rename("geo_surf")
    bound_lay_height = ds["boundary_layer_height"].rename("bound_lay_height")
    mean_sea_lvl_press = ds["mean_sea_level_pressure"].rename(
        "mean_sea_lvl_press")
    soil = ds["soil_type"].rename("soil")
    cloud_cov = ds["total_cloud_cover"].rename("cloud_cov")
    vol_soil_layer1 = ds["volumetric_soil_water_layer_1"].rename(
        "vol_soil_layer1")

    # --- Entrées avec niveau vertical -> on sélectionne level ---
    geo_wind_speed_925 = ds["geostrophic_wind_speed"].sel(
        level=925).rename("geo_wind_speed_925")
    geo_wind_speed_700 = ds["geostrophic_wind_speed"].sel(
        level=700).rename("geo_wind_speed_700")
    geo_wind_speed_500 = ds["geostrophic_wind_speed"].sel(
        level=500).rename("geo_wind_speed_500")
    rel_hum_925 = ds["relative_humidity"].sel(level=925).rename("rel_hum_925")
    rel_hum_700 = ds["relative_humidity"].sel(level=700).rename("rel_hum_700")
    rel_hum_500 = ds["relative_humidity"].sel(level=500).rename("rel_hum_500")
    temp_925 = ds["temperature"].sel(level=925).rename("temp_925")
    temp_700 = ds["temperature"].sel(level=700).rename("temp_700")
    temp_500 = ds["temperature"].sel(level=500).rename("temp_500")
    vertical_velo_925 = ds["vertical_velocity"].sel(
        level=925).rename("vertical_velo_925")
    vertical_velo_700 = ds["vertical_velocity"].sel(
        level=700).rename("vertical_velo_700")
    vertical_velo_500 = ds["vertical_velocity"].sel(
        level=500).rename("vertical_velo_500")
    u_comp_925 = ds["u_component_of_wind"].sel(level=925).rename("u_comp_925")
    u_comp_700 = ds["u_component_of_wind"].sel(level=700).rename("u_comp_700")
    u_comp_500 = ds["u_component_of_wind"].sel(level=500).rename("u_comp_500")
    v_comp_925 = ds["v_component_of_wind"].sel(level=925).rename("v_comp_925")
    v_comp_700 = ds["v_component_of_wind"].sel(level=700).rename("v_comp_700")
    v_comp_500 = ds["v_component_of_wind"].sel(level=500).rename("v_comp_500")

    # Dataset final "propre"
    ds_sel = xr.Dataset(
        data_vars={
            "tp_6h": tp_6h,
            "t_2m": t_2m,
            "u_comp_10": u_10,
            "v_comp_10": v_10,
            "land_sea_mask": land_sea_mask,
            "geo_surf": geo_surf,
            "bound_lay_height": bound_lay_height,
            "geo_wind_speed_925": geo_wind_speed_925,
            "geo_wind_speed_700": geo_wind_speed_700,
            "geo_wind_speed_500": geo_wind_speed_500,
            "mean_sea_lvl_press": mean_sea_lvl_press,
            "rel_hum_925": rel_hum_925,
            "rel_hum_700": rel_hum_700,
            "rel_hum_500": rel_hum_500,
            "soil": soil,
            "temp_925": temp_925,
            "temp_700": temp_700,
            "temp_500": temp_500,
            "vertical_velo_925": vertical_velo_925,
            "vertical_velo_700": vertical_velo_700,
            "vertical_velo_500": vertical_velo_500,
            "cloud_cov": cloud_cov,
            "u_comp_925": u_comp_925,
            "u_comp_700": u_comp_700,
            "u_comp_500": u_comp_500,
            "v_comp_925": v_comp_925,
            "v_comp_700": v_comp_700,
            "v_comp_500": v_comp_500,
            "vol_soil_layer1": vol_soil_layer1
        },
        coords={
            "time": ds.time,
            "latitude": ds.latitude,
            "longitude": ds.longitude,
        },
    )

    return ds_sel


def split_train_val_test(ds: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Découpe le dataset en train / val / test selon les dates définies en haut du fichier.
    """
    ds_train = ds.sel(time=slice(TRAIN_START, TRAIN_END))
    ds_val = ds.sel(time=slice(VAL_START,   VAL_END))
    ds_test = ds.sel(time=slice(TEST_START,  TEST_END))

    return ds_train, ds_val, ds_test


class ERA5Dataset(Dataset):
    """
    Dataset PyTorch

    - Entrée : séquence X de shape (T_in, C_in, H, W)
        T_in = n_input_steps (par ex. 9)
        C_in = nb de variables d'entrée (toutes sauf tp_6h)
    - Cible : y = tp_6h à un temps futur (lead_steps * 6h), shape (H, W)

    ds : xarray.Dataset avec dimensions (time, latitude, longitude)
         et variables :
           - 'tp_6h' (target)
           - plusieurs variables features (entrées)
    """

    def __init__(
        self,
        ds: xr.Dataset,
        input_vars: List[str],
        target_var: str = "tp_6h",
        # nombre de pas temporels en entrée (par ex. 9 => -48h...0h)
        n_input_steps: int = 9,
        # lead time (en indices de pas de temps : 1 => +6h)
        lead_steps: int = 1,
    ):
        super().__init__()

        self.ds = ds
        self.input_vars = input_vars
        self.target_var = target_var
        self.n_input_steps = n_input_steps
        self.lead_steps = lead_steps

        times = ds.time.values
        n_time = len(times)

        t_min = n_input_steps - 1
        t_max = n_time - 1 - lead_steps

        if t_max <= t_min:
            raise ValueError(
                f"Pas assez de pas de temps pour n_input_steps={n_input_steps} "
                f"et lead_steps={lead_steps} (n_time={n_time})"
            )

        self.valid_time_indices = np.arange(t_min, t_max + 1)

        # Juste pour sanity check : mémoriser tailles spatiales
        self.H = ds.dims["latitude"]
        self.W = ds.dims["longitude"]

    def __len__(self):
        return len(self.valid_time_indices)

    def __getitem__(self, idx):
        """
        Retourne :
            X : (T_in, C_in, H, W)
            y : (H, W)
        """
        t0 = int(self.valid_time_indices[idx])

        # index du temps le plus ancien
        t_start = t0 - (self.n_input_steps - 1)
        # slice exclusif, donc [t_start, ..., t0]
        t_end = t0 + 1
        t_target = t0 + self.lead_steps          # instant futur pour la cible

        # --- Entrée X : features sur la fenêtre temporelle ---
        ds_in = self.ds[self.input_vars].isel(time=slice(t_start, t_end))
        # dims : time, latitude, longitude pour chaque variable

        # On empile les variables en dimension "channel"
        # -> DataArray (channel, time, lat, lon)
        da_in = ds_in.to_array("channel")

        # On réordonne en (time, channel, lat, lon)
        da_in = da_in.transpose("time", "channel", "latitude", "longitude")

        # Conversion en numpy puis torch (déclenche un compute pour cette petite slice)
        x_np = da_in.values.astype("float32")  # (T_in, C_in, H, W)
        X = torch.from_numpy(x_np)

        # --- Cible y : tp_6h à t_target ---
        da_out = self.ds[self.target_var].isel(time=t_target)  # (lat, lon)
        y_np = da_out.values.astype("float32")  # (H, W)
        y = torch.from_numpy(y_np)

        return X, y
