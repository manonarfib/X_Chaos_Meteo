# Utiliser comme ça :
    # dataset_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_validation.zarr"
    # T, lead = 8, 1
    # batch_size = 2

    # dataset = ERA5Dataset(dataset_path, T=T, lead=lead)
    # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

import torch
from torch.utils.data import DataLoader, Dataset
import xarray as xr
import pandas as pd
import numpy as np


class ERA5Dataset(Dataset):        
    def __init__(self, path, T=8, lead=1, last_n_years=None):
        ds = xr.open_zarr(path, chunks=None)
        
        # invalid times : times with nan during the download (merci le DCE 😭)
        invalid_times = (
        pd.date_range(start="2007-09-05 18:00:00", end="2007-09-17 12:00:00", freq="6H")
        .append(
            pd.date_range(start="2001-10-11 00:00:00", end="2001-10-27 18:00:00", freq="6H")))
        ds = ds.sel(time=~ds.time.isin(invalid_times))
        
        self.X = ds["X"]
        self.Y = ds["Y"]
        self.T = T
        self.lead = lead
        self.nt = self.X.sizes["time"]
        
        # last_n_years : pour sélectionner les n dernières années du dataset, plutôt pour débuger
        if last_n_years is not None:
            # Date de fin du dataset
            end_date = pd.to_datetime(self.X.time.values[-1])
            start_date = end_date - pd.DateOffset(years=last_n_years-1)
            
            # Indices correspondant
            times = pd.to_datetime(self.X.time.values)
            mask = (times >= start_date)
            self.start_idx = mask.argmax()  # premier True
            self.nt = self.nt - self.start_idx
            
            print(f"Filtré pour les {last_n_years} dernières années : {times[self.start_idx]} -> {times[-1]}")
        else:
            self.start_idx = 0

    def __len__(self):
        return self.nt - self.T - self.lead + 1

    def __getitem__(self, i):
        i = i + self.start_idx
        x = torch.from_numpy(
            self.X.isel(time=slice(i, i + self.T)).values
        )
        y = torch.from_numpy(
            self.Y.isel(time=i + self.T + self.lead - 1).values
        )
        y = torch.clamp(y, min=0.0) # clamp pour ne pas avoir de precip négatives (existe dans ERA5 dû à de l'interpolation)
        # On retourne i pour connaître la date
        return x, y, i


