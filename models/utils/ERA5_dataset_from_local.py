# Use like this :
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
    def __init__(self, path, T=8, lead=1, last_n_years=None, mean="models/utils/era5_mean.npy", std="models/utils/era5_std.npy"):
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
        
        if mean is not None and std is not None:
            mean = np.load(mean)
            std  = np.load(std)
            if isinstance(mean, np.ndarray):
                mean = torch.from_numpy(mean)
            if isinstance(std, np.ndarray):
                std = torch.from_numpy(std)
            self.mean = mean
            self.std = std
        else:
            self.mean = None
            self.std = None
        
        # last_n_years : to select the last n years of the dataset, use for debugging preferably
        if last_n_years is not None:
            # End date of the datset
            end_date = pd.to_datetime(self.X.time.values[-1])
            start_date = end_date - pd.DateOffset(years=last_n_years-1)
            
            # Corresponding indices
            times = pd.to_datetime(self.X.time.values)
            mask = (times >= start_date)
            self.start_idx = mask.argmax()  # first True
            self.nt = self.nt - self.start_idx
            
            print(f"Filtered for the last {last_n_years} years : {times[self.start_idx]} -> {times[-1]}")
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
        
        if self.mean is not None and self.std is not None:
            # x shape: (T, C, H, W)
            x = (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        
        y = torch.clamp(y, min=0.0) # clamp so there's no negative precipitations (they exist in ERA5 because of interpolation)
        # Return i to know what dates are in the batch
        return x, y, i


