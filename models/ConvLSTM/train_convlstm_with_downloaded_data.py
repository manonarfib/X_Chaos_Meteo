import time
import os
import csv
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.utils.losses import WeightedMSELoss, WeightedDiceRegressionLoss
from models.utils.ERA5_dataset_from_local import ERA5Dataset
from models.ConvLSTM.convlstm import PrecipConvLSTM


# Configuration

@dataclass
class Config:
    # Data
    train_dataset_path: str = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_train.zarr"
    val_dataset_path: str = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_validation.zarr"
    T: int = 8
    lead: int = 1
    batch_size: int = 16
    
    # Steps
    number_of_days_for_gradient_acc = 30 # 1 mois
    number_of_days_before_eval = 730 # 365*2 = 2 ans

    # Training
    n_epochs: int = 3
    lr: float = 1e-3
    # loss : "mse", "w_mse", "w_dice", "w_mse_and_w_dice"
    loss_type: str = "mse"

    # Model
    hidden_channels: Tuple[int, int] = (32, 64)
    kernel_size: int = 3

    # Logging / checkpoint
    checkpoint_dir: str = "checkpoints"
    train_csv: str = "checkpoints/train_log.csv"
    val_csv: str = "checkpoints/validation_log.csv"

    # Misc
    seed: int = 42


# Utils

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")
    return device


# Loss

def compute_loss(output, target, loss_type="w_mse_and_w_dice"):
    if loss_type in ["w_mse", "w_dice", "w_mse_and_w_dice"]:
        weight = torch.where(
            target > 0.1,
            torch.tensor(5.0, device=target.device),
            torch.tensor(1.0, device=target.device),
        )
    else:
        weight = None

    if loss_type == "w_mse_and_w_dice":
        loss_mse = WeightedMSELoss()(output, target, weight)
        loss_dice = WeightedDiceRegressionLoss()(output, target, weight)
        return 0.7 * loss_mse + 0.3 * loss_dice

    if loss_type == "w_dice":
        return WeightedDiceRegressionLoss()(output, target, weight)

    if loss_type == "w_mse":
        return WeightedMSELoss()(output, target, weight)

    return WeightedMSELoss()(output, target)


# Data

def create_dataloaders(cfg: Config):
    print("\n[STEP] Création du DataLoader...")
    t0 = time.time()

    train_dataset = ERA5Dataset(cfg.train_dataset_path, T=cfg.T, lead=cfg.lead)
    val_dataset = ERA5Dataset(cfg.val_dataset_path, T=cfg.T, lead=cfg.lead)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    input_vars = list(train_dataset.X.coords["channel"].values)
    print(f"number of input vars : {len(input_vars)}")
    print(f"Nombre de batches par epoch : {len(train_loader)}")
    print(f"[DONE] DataLoader créé en {time.time() - t0:.1f} s")

    # Warm-up
    X, y, _ = next(iter(train_loader))
    print(f"[WARMUP] X.shape : {X.shape}, y.shape : {y.shape}")

    return train_loader, val_loader, train_dataset, input_vars


# Model / Optim

def build_model(cfg: Config, input_channels: int, device: torch.device):
    print("\n[STEP] Initialisation du modèle PrecipConvLSTM...")
    t0 = time.time()

    model = PrecipConvLSTM(
        input_channels=input_channels,
        hidden_channels=list(cfg.hidden_channels),
        kernel_size=cfg.kernel_size,
    ).to(device)

    print(
        "Nombre de paramètres du modèle :",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    print(f"[DONE] Modèle initialisé en {time.time() - t0:.2f} s")
    return model


def build_optimizer(model, cfg: Config):
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )
    return optimizer, scheduler


# Checkpoints / logs

def init_csv(path, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def save_checkpoint(path, epoch, model, optimizer):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_last_checkpoint(path, model, optimizer, device):
    if not os.path.exists(path):
        print("[CHECKPOINT] Aucun checkpoint trouvé, entraînement depuis le début")
        return 1

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"[CHECKPOINT] Chargé depuis {path}, reprise à l'époque {start_epoch}")
    return start_epoch


# Validation

def run_validation(model, val_loader, device, loss_type):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y, _ in tqdm(val_loader, desc="Validation", leave=True):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_hat = model(X).squeeze(1)
            loss = compute_loss(y_hat, y, loss_type)
            total_loss += loss.item()

    return total_loss / len(val_loader)


# Training

def train(cfg: Config):
    set_seed(cfg.seed)
    device = get_device()

    train_loader, val_loader, train_dataset, input_vars = create_dataloaders(cfg)
    model = build_model(cfg, len(input_vars), device)
    optimizer, scheduler = build_optimizer(model, cfg)

    init_csv(cfg.train_csv, ["epoch", "batch_idx", "loss"])
    init_csv(cfg.val_csv, ["epoch", "batch_idx", "eval_type", "loss"])

    accumulation_steps = (cfg.number_of_days_for_gradient_acc*4) // cfg.batch_size
    evaluation_steps = (cfg.number_of_days_before_eval*4) // cfg.batch_size
    print(f"Accumulation steps : {accumulation_steps}")

    last_checkpoint = os.path.join(cfg.checkpoint_dir, "checkpoint_last.pt")
    start_epoch = load_last_checkpoint(last_checkpoint, model, optimizer, device)

    previous_val_loss = np.inf

    print("\n[TRAIN] Début de l'entraînement...")
    t_train = time.time()

    for epoch in range(start_epoch, cfg.n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        epoch_loss = 0.0
        acc_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.n_epochs}")
        for batch_idx, (X, y, i) in enumerate(pbar):
            print(f"event : training start, batch : {batch_idx}")
            t0 = time.time()

            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_hat = model(X).squeeze(1)
            loss = compute_loss(y_hat, y, cfg.loss_type)

            if torch.isnan(loss):
                date = pd.to_datetime(train_dataset.Y.time.values[i])
                print(f"NaN détecté à la date {date}")
                continue

            (loss / accumulation_steps).backward()
            epoch_loss += loss.item()
            acc_loss += loss / accumulation_steps

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

                with open(cfg.train_csv, "a", newline="") as f:
                    csv.writer(f).writerow([epoch, batch_idx, acc_loss.item()])

                acc_loss = 0.0

            if (batch_idx + 1) % evaluation_steps == 0 or (batch_idx + 1) == len(train_loader):
                val_loss = run_validation(model, val_loader, device, cfg.loss_type)

                if val_loss < previous_val_loss:
                    save_checkpoint(
                        f"{cfg.checkpoint_dir}/best_checkpoint_epoch{epoch}_batch{batch_idx}.pt",
                        epoch,
                        model,
                        optimizer,
                    )
                    previous_val_loss = val_loss

                with open(cfg.val_csv, "a", newline="") as f:
                    csv.writer(f).writerow([epoch, batch_idx, "val", val_loss])

                scheduler.step(val_loss)
                model.train()

            print(f"event : training batch end, duration : {time.time() - t0:.2f}s")

        avg_loss = epoch_loss / len(train_loader)
        print(f">>> Epoch {epoch} terminé - loss moyenne : {avg_loss:.4e}")

        save_checkpoint(f"{cfg.checkpoint_dir}/epoch{epoch}_full.pt", epoch, model, optimizer)
        save_checkpoint(last_checkpoint, epoch, model, optimizer)

    print(f"[TRAIN] Entraînement terminé en {time.time() - t_train:.1f}s")



if __name__ == "__main__":
    cfg = Config()
    train(cfg)
