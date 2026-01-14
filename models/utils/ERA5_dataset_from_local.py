# S'appelle comme ça :
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
    # def __init__(self, path, T=8, lead=1):
        # ds = xr.open_zarr(path, chunks=None) 
        # # ds = xr.open_zarr("/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr")
        # print(ds.time.min().values, ds.time.max().values)
        # print(len(ds.time))
        # self.X = ds["X"]
        # self.Y = ds["Y"]
        # self.T = T
        # self.lead = lead
        # self.nt = self.X.sizes["time"]
        
    def __init__(self, path, T=8, lead=1, last_n_years=None):
        ds = xr.open_zarr(path, chunks=None)
        
        invalid_times = pd.date_range(
            start="2007-09-05 18:00:00",
            end="2007-09-17 12:00:00",
            freq="6H"
        )
        ds = ds.sel(time=~ds.time.isin(invalid_times))
        
        self.X = ds["X"]
        self.Y = ds["Y"]
        self.T = T
        self.lead = lead
        self.nt = self.X.sizes["time"]
        
        if last_n_years is not None:
            # Date de fin du dataset
            end_date = pd.to_datetime(self.X.time.values[-1])
            start_date = end_date - pd.DateOffset(years=last_n_years-1, months=5)
            
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
        # lecture contiguë sur l’axe time
        i = i + self.start_idx
        # date = pd.to_datetime(self.X.time.values[i])
        x = torch.from_numpy(
            self.X.isel(time=slice(i, i + self.T)).values
        )
        y = torch.from_numpy(
            self.Y.isel(time=i + self.T + self.lead - 1).values
        )
        y = torch.clamp(y, min=0.0)
        return x, y, i

    def get_date(self, batch_idx, t_idx=0):
        """
        Retourne la date correspondant à un batch et un index temporel dans ce batch.
        batch_idx : indice de départ dans le dataset (i dans __getitem__)
        t_idx : index temporel à l'intérieur du batch (0 à T-1)
        """
        time_index = batch_idx + t_idx
        date = self.X.time.isel(time=time_index).values
        return np.datetime_as_string(date)
