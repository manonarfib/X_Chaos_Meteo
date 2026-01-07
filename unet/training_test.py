import numpy as np
import torch
import torch.nn as nn
from model import WFUNet_with_train
from make_datasets_t import build_datasets, ERA5Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

np.random.seed(1)
torch.manual_seed(2)
torch.cuda.manual_seed_all(2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Hyperparameters ---- #
lags = 12
lat = 96
long = 96
feats = 1
feats_out = 1
filters = 16
dropout = 0.5
batch_size = 2
epochs = 200
learning_rate = 1e-3
# choisir entre w_mse_and_w_dice, w_mse, w_dice or mse
loss_type = "w_mse_and_w_dice"

# ---- Datasets ---- #
print("[STEP] build_datasets() ...")
ds_train, ds_val, ds_test = build_datasets()
print(f"[DONE] build_datasets()")

all_vars = list(ds_train.data_vars.keys())
input_vars = [v for v in all_vars if v != "tp_6h"]
target_var = "tp_6h"
print("\nVariables d'entrée utilisées (features) :")
for v in input_vars:
    print("  -", v)
print("Variable cible :", target_var)

print("\n[STEP] Création des Datasets PyTorch")
train_dataset = ERA5Dataset(
    ds=ds_train,
    input_vars=input_vars,
    target_var=target_var,
    n_input_steps=9,
    lead_steps=1,
)
val_dataset = ERA5Dataset(
    ds=ds_val,
    input_vars=input_vars,
    target_var=target_var,
    n_input_steps=9,
    lead_steps=1,
)
print(f"[DONE] Dataset PyTorch créé")
print(f"Nombre d'exemples dans train_dataset")

print("\n[STEP] Création des DataLoaders...")
batch_size = 2
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # à 0 pour éviter les soucis avec xarray/dask
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # à 0 pour éviter les soucis avec xarray/dask
)
print(f"[DONE] DataLoaders créés")
num_batches = len(train_loader)
print(f"Nombre de batches par epoch : {num_batches}")


# ---- Model ---- #
model = WFUNet_with_train(lags, lat, long, feats, feats_out,
                          filters, dropout).to(device)

# ---- Optimizer / Scheduler ---- #
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(
    optimizer, factor=0.5, patience=10, min_lr=1e-4, verbose=True)

# ---- Train ---- #
train_losses, val_losses = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=epochs,
    loss_type=loss_type,
    device=device,
    save_path="best_model.pt"
)
