import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os
import csv
import random
from tqdm import tqdm

from models.utils.losses import WeightedMSELoss, WeightedDiceRegressionLoss
from models.utils.ERA5_dataset_from_local import ERA5Dataset
from convlstm import PrecipConvLSTM

random.seed(42)

def compute_loss(output, target, loss_type="w_mse_and_w_dice"):
        # print("SHAPE TARGET", target.shape)
        if loss_type in ["w_mse", "w_dice", "w_mse_and_w_dice"]:
            weight = torch.where(
                target > 0.1,
                torch.tensor(5.0, device=target.device),
                torch.tensor(1.0, device=target.device)
            )
        else:
            weight = None

        if loss_type == "w_mse_and_w_dice":
            # print(weight.shape)
            criterion_mse = WeightedMSELoss()
            criterion_dice = WeightedDiceRegressionLoss()
            loss_mse = criterion_mse(output, target, weight)
            loss_dice = criterion_dice(output, target, weight)
            return 0.7 * loss_mse + 0.3 * loss_dice

        elif loss_type == "w_dice":
            criterion_dice = WeightedDiceRegressionLoss()
            return criterion_dice(output, target,weight)

        elif loss_type == "w_mse":
            criterion_mse = WeightedMSELoss()
            return criterion_mse(output, target,weight)

        else:  # mse only
            criterion_mse = WeightedMSELoss()
            return criterion_mse(output, target)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    print("\n[STEP] Création du DataLoader...")
    t3 = time.time()

    # train_loader, val_loader, test_loader = make_dataloaders(
    #     train_time_range=("1970-01-01", "2010-01-01"),
    #     val_time_range=("2016-01-01", "2016-01-04"),
    #     test_time_range=("2018-01-01", "2018-01-14"),
    # )

    train_dataset_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_train.zarr"
    val_dataset_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_validation.zarr"
    T, lead = 8, 1
    batch_size = 16

    train_dataset = ERA5Dataset(train_dataset_path, T=T, lead=lead)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
    
    # for i, (X, y) in enumerate(train_loader):
        # Vérifier NaN dans X
        # print(i)
        # nan_mask_X = torch.isnan(X)
        
        # if nan_mask_X.any():
        #     print("nan found !")

        # # Vérifier NaN dans y
        # nan_mask_y = torch.isnan(y)
        # if nan_mask_y.any():
        #     print("nan found in y!")
            
    # for X, y, idx in train_loader:
    #     print(idx)
    #     if torch.isnan(X).any():
    #         date = pd.to_datetime(train_dataset.X.time.values[idx])
    #         print(f"NaN détecté à la date {date}")
    
    val_dataset = ERA5Dataset(val_dataset_path, T=T, lead=lead)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) 
    print("dataset ok")
    input_vars = list(train_dataset.X.coords["channel"].values)
    print(f"number of input vars : {len(input_vars)}")

    print(f"[DONE] DataLoader créé en {time.time() - t3:.1f} s")
    num_batches = len(train_loader)
    # num_batches = 180 # 6 mois de données si batch size = 2, env 1h30/2h pour 1 epoch
    # num_batches = 10
    print(f"Nombre de batches par epoch : {num_batches}")

    print("Warming up the dataloader")
    X,y,idx_ = next(iter(train_loader))  # wait until dataloader is ready
    # print(X.shape, y.shape)
    def tensor_size_mb(t):
        return t.numel() * t.element_size() / 1024**2

    print("X shape:", X.shape)
    print(f"X size: {tensor_size_mb(X):.2f} MB")

    print("y shape:", y.shape)
    print(f"y size: {tensor_size_mb(y):.2f} MB")
    print(f"CUDA allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    print(f"CUDA reserved : {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    print("Dataloader warmed up !")

    # -------------------------------------------------
    # 6) Modèle ConvLSTM
    # -------------------------------------------------
    C_in = len(input_vars)
    print("\n[STEP] Initialisation du modèle PrecipConvLSTM...")
    t4 = time.time()
    model = PrecipConvLSTM(
        input_channels=C_in,
        hidden_channels=[32, 64],  # plus petit pour debug
        kernel_size=3,
    ).to(device)
    print(f"[DONE] Modèle initialisé en {time.time() - t4:.2f} s")
    print("Nombre de paramètres du modèle :",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    # -------------------------------------------------
    # 7) Loss & Optimizer
    # -------------------------------------------------
    criterion = nn.MSELoss() # voir pour une weighted loss de Manon
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',       # on cherche à réduire la loss
        factor=0.5,       # le LR sera multiplié par 0.5
        patience=3,       # on attend 3 évaluations sans amélioration
    )

    # -------------------------------------------------
    # 8) Boucle d'entraînement (sur les 2 semaines)
    # -------------------------------------------------
    n_epochs = 3  

    print("\n[TRAIN] Début de l'entraînement...")
    t_train_start = time.time()

    start_epoch = 1
    
    last_checkpoint = "checkpoints/checkpoint_last.pt"
    os.makedirs(os.path.dirname(last_checkpoint), exist_ok=True)
    
    if os.path.exists(last_checkpoint):
        checkpoint = torch.load(last_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"[CHECKPOINT] Chargé depuis {last_checkpoint}, reprise à l'époque {start_epoch}")
    else:
        print("[CHECKPOINT] Aucun checkpoint trouvé, entraînement depuis le début")

    csv_path_val = "checkpoints/validation_log.csv"
    os.makedirs("checkpoints", exist_ok=True)
    csv_header = ["epoch", "batch_idx", "eval_type", "loss"]

    if not os.path.exists(csv_path_val):
        with open(csv_path_val, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            
    csv_path_train = "checkpoints/train_log.csv"
    os.makedirs("checkpoints", exist_ok=True)
    csv_header = ["epoch", "batch_idx", "loss"]

    if not os.path.exists(csv_path_train):
        with open(csv_path_train, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    accumulation_steps = (120 // batch_size) # 1 mois
    evaluation_steps = 2920//batch_size # 2ans
    # accumulation_steps = 2
    print(f"The model weights will be updated every {accumulation_steps} batches.")
    
    previous_val_b_loss = np.inf
    val_batches_loss = 0

    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        epoch_loss = 0.0
        epoch_start = time.time()
        accumulation_step_loss = 0

        pbar = tqdm(train_loader, total=num_batches, desc=f"Epoch {epoch}/{n_epochs}", leave=True)
        for batch_idx, (X, y, i) in enumerate(pbar):
            batch_start = time.time()
            print(f"event : training start, batch : {batch_idx}")
        
            # X : (B, T_in, C_in, H, W)
            # y : (B, H, W)
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            
            # if torch.isnan(X).any():
            #     date = pd.to_datetime(train_dataset.X.time.values[i])
            #     print(f"NaN détecté à la date {date}")

            # optimizer.zero_grad()

            # Forward
            y_hat = model(X)        # (B, 1, H, W)
            # print(y_hat)
            y_hat = y_hat.squeeze(1)  # (B, H, W)
            # breakpoint()
            # Loss
            # loss = criterion(y_hat, y)
            # loss.backward()
            # optimizer.step()
            raw_loss = criterion(y_hat, y)
            if torch.isnan(raw_loss).any():
                date = pd.to_datetime(train_dataset.Y.time.values[i])
                print(f"NaN détecté à la date {date}")
                continue
            # raw_loss = compute_loss(y_hat, y, loss_type="w_mse_and_w_dice")
            (raw_loss / accumulation_steps).backward()
            epoch_loss += raw_loss.item()
            accumulation_step_loss += raw_loss / accumulation_steps
            print(raw_loss)
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                batch_time = time.time() - batch_start
                print(
                    f"[Epoch {epoch}/{n_epochs}] "
                    f"Batch {batch_idx+1}/{num_batches} "
                    f"- Loss: {accumulation_step_loss.item():.4e} "
                    f"- Batch time: {batch_time:.2f}s"
                )
                
                with open(csv_path_train, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, batch_idx, accumulation_step_loss])
                    
                accumulation_step_loss = 0
                
            if (batch_idx + 1) % evaluation_steps == 0 or (batch_idx + 1) == num_batches:
                model.eval()
                previous_val_b_loss = val_batches_loss
                val_batches_loss = 0.0
                n_val_batches = 0

                with torch.no_grad():
                    num_batches_val = len(val_loader)
                    pbar_val = tqdm(val_loader, total=num_batches_val, desc=f"Epoch {epoch}/{n_epochs}", leave=True)
                    for batch_idx, (Xv, yv) in enumerate(pbar_val):
                        Xv = Xv.to(device, non_blocking=True)
                        yv = yv.to(device, non_blocking=True)

                        yv_hat = model(Xv).squeeze(1)
                        val_loss = criterion(yv_hat, yv)

                        val_batches_loss += val_loss.item()
                        n_val_batches += 1

                val_batches_loss /= n_val_batches
                
                if val_batches_loss<previous_val_b_loss:
                    best_checkpoint = f"checkpoints/best_checkpoint_epoch{epoch}_batch_idx{batch_idx}.pt"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }, best_checkpoint)
                
                with open(csv_path_val, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, batch_idx, "val", val_batches_loss])
                
                model.train()
                scheduler.step(val_batches_loss)
                        

            # epoch_loss += loss.item()
            
            batch_end = time.time()
            print(f"event : training batch end, batch {batch_idx}, duration :{batch_end - batch_start}")


        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_batches
        pbar.set_postfix(loss=f"{epoch_loss:.3e}", avg=f"{avg_loss:.3e}", bt=f"{batch_time:.2f}s")
        print(
            f"\n>>> Epoch {epoch}/{n_epochs} terminée "
            f"- Loss moyen: {avg_loss:.4e} "
            f"- Temps epoch: {epoch_time:.1f}s\n"
        )
            
        epoch_checkpoint = f"checkpoints/epoch{epoch}_full.pt"
        torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, epoch_checkpoint)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, last_checkpoint)
        print(f"[CHECKPOINT] Sauvegardé à {epoch_checkpoint}")
            

    total_train_time = time.time() - t_train_start
    print(f"[TRAIN] Entraînement terminé en {total_train_time:.1f} s")

    # -------------------------------------------------
    # 9) Phase de prédiction sur quelques batches du train
    # -------------------------------------------------
    # print("\n[PRED] Prédiction sur quelques batches du train...")
    # model.eval()

    # with torch.no_grad():
    #     t_pred_start = time.time()
    #     for i, (X, y) in enumerate(train_loader):
    #         X = X.to(device)
    #         y = y.to(device)

    #         t_batch_pred_start = time.time()
    #         y_hat = model(X)           # (B, 1, H, W)
    #         y_hat = y_hat.squeeze(1)   # (B, H, W)
    #         t_batch_pred = time.time() - t_batch_pred_start

    #         # Afficher quelques stats sur la prédiction
    #         print(
    #             f"[PRED] Batch {i+1} "
    #             f"- y_hat shape: {tuple(y_hat.shape)} "
    #             f"- min: {float(y_hat.min()):.3f}, "
    #             f"max: {float(y_hat.max()):.3f}, "
    #             f"time: {t_batch_pred:.2f}s"
    #         )

    #         if i >= 2:
    #             # On s'arrête après 3 batches de prédiction pour debug
    #             break

    #     print(f"[PRED] Phase de prédiction terminée en {time.time() - t_pred_start:.1f} s")


if __name__ == "__main__":
    main()