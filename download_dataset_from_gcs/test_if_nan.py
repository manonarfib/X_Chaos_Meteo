import torch
import time
from ERA5_dataset_from_local import ERA5Dataset
from torch.utils.data import DataLoader
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device utilisé : {device}")

train_dataset_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_train.zarr"
# val_dataset_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_validation.zarr"
T, lead = 8, 1
batch_size = 16

train_dataset = ERA5Dataset(train_dataset_path, T=T, lead=lead)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0) 

i=0        
for X, y, idx in train_loader:
    print(i)
    if torch.isnan(X).any():
        date = pd.to_datetime(train_dataset.X.time.values[idx])
        print(f"NaN détecté à la date {date} in X")
        
    if torch.isnan(y).any():
        date = pd.to_datetime(train_dataset.Y.time.values[idx])
        print(f"NaN détecté à la date {date} in y")
    i+=1
