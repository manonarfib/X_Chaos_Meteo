import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import argparse

from models.unet.model_without_collapse import WFUNet_with_train
from models.utils.ERA5_dataset_from_local import ERA5Dataset


if __name__=="__main__":

    np.random.seed(1)
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Hyperparameters ---- #
    n_input_steps=8 #On prend le temps courant et les 8 time steps précédents (48h)
    lead_steps=1 #On prédit pour 6h après
    lags=n_input_steps
    # feats = 1 #nombre de variables d'entrée
    feats_out = 1 #nombre de variables de sortie (1 pour nous car veut seulement prédire les précipitations)
    filters = 32
    dropout = 0
    batch_size = 8
    epochs = 3
    learning_rate = 1e-3
    # choisir entre w_mse_and_w_dice, w_mse, w_dice or mse
    loss_type = "wmse"
    weight_update_interval=120 #maj poids tous les mois
    val_loss_calculation_interval=4*720 #calcul loss tous les 2 ans

    print("\n[STEP] Création des DataLoaders...")
    t3 = time.time()
    dataset_train_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_train.zarr"
    dataset_train = ERA5Dataset(dataset_train_path, T=n_input_steps, lead=lead_steps)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0) 
    print("dataset train ok")
    input_vars = list(dataset_train.X.coords["channel"].values)
    print(f"number of input vars : {len(input_vars)}")

    dataset_val_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_validation.zarr"
    dataset_val = ERA5Dataset(dataset_val_path, T=n_input_steps, lead=lead_steps)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=0) 
    print("dataset val ok")

    print(f"[DONE] DataLoaders créés en {time.time() - t3:.1f} s")
    num_batches = len(train_loader)
    # num_batches = 180 # 6 mois de données si batch size = 2, env 1h30/2h pour 1 epoch
    # num_batches = 10
    print(f"Nombre de batches par epoch : {num_batches}")

    print("Warming up the dataloader")
    X,y, idx_ = next(iter(train_loader))  # wait until dataloader is ready
    print(X.shape, y.shape)
    feats=X.shape[2]
    lat=X.shape[3]
    long=X.shape[4]
    print("Dataloader warmed up !")

    # ---- Model ---- #
    model = WFUNet_with_train(lags, lat, long, feats, feats_out,
                              batch_size, filters, dropout).to(device)

    # ---- Optimizer / Scheduler ---- #
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',       # on cherche à réduire la loss
        factor=0.5,       # le LR sera multiplié par 0.5
        patience=3,       # on attend 2 évaluations sans amélioration
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="checkpoints",
        help="Chemin où sauvegarder les checkpoints"
    )
    args = parser.parse_args()

    save_path = args.save_path

    print("n_input_steps=",n_input_steps)
    print("lead_steps=", lead_steps)
    print("lags=", lags)
    print("feats_out=", feats_out)
    print("filters=", filters)
    print("dropout=", dropout)
    print("batch_size=", batch_size)
    print("epochs=", epochs)
    print("learning_rate=",learning_rate)
    print("loss_type=", loss_type)
    print("weight_update_interval=", weight_update_interval)
    print("val_loss_calculation_interval=", val_loss_calculation_interval)   

    # ---- Train ---- #
    train_losses, val_losses = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        loss_type=loss_type,
        device=device,
        weight_update_interval=weight_update_interval,
        val_loss_calculation_interval=val_loss_calculation_interval,
        save_path=save_path
    )