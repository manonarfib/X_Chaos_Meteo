import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import csv

from make_datasets import build_datasets, ERA5Dataset
from convlstm import PrecipConvLSTM


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    # -------------------------------------------------
    # 1) Construction des datasets xarray (lazy)
    # -------------------------------------------------
    t0 = time.time()
    print("[STEP] build_datasets() ...")
    ds_train, ds_val, ds_test = build_datasets()
    print(f"[DONE] build_datasets() en {time.time() - t0:.1f} s")
    
    # -------------------------------------------------
    # 2) Restriction à 2 semaines sur le train
    #    (2 semaines de données 6h -> 14 jours * 4 = 56 pas de temps)
    #    -> On garde les 56 premiers timestamps
    # -------------------------------------------------
    print("[STEP] Sous-échantillonnage temporel du train à 2 semaines...")
    t1 = time.time()
    ds_train = ds_train.isel(time=slice(0, 56))  # 56 timestamps = 2 semaines
    print(f"[DONE] Sous-échantillonnage train en {time.time() - t1:.1f} s")
    print("Taille ds_train (2 semaines) :", dict(ds_train.sizes))

    # -------------------------------------------------
    # 3) Variables d'entrée / cible
    # -------------------------------------------------
    all_vars = list(ds_train.data_vars.keys())
    input_vars = [v for v in all_vars if v != "tp_6h"]
    target_var = "tp_6h"

    print("\nVariables d'entrée utilisées (features) :")
    for v in input_vars:
        print("  -", v)
    print("Variable cible :", target_var)

    # -------------------------------------------------
    # 4) Création du Dataset PyTorch
    #    n_input_steps = 5 -> historique de 5*6h = 30h
    #    lead_steps    = 1 -> prédiction à +6h
    # -------------------------------------------------
    print("\n[STEP] Création du Dataset PyTorch (ERA5Dataset)...")
    t2 = time.time()
    train_dataset = ERA5Dataset(
        ds=ds_train,
        input_vars=input_vars,
        target_var=target_var,
        n_input_steps=5,
        lead_steps=1,
    )
    val_dataset = ERA5Dataset(
        ds=ds_val,
        input_vars=input_vars,
        target_var=target_var,
        n_input_steps=5,
        lead_steps=1,
    )
    print(f"[DONE] Dataset PyTorch créé en {time.time() - t2:.1f} s")
    print(f"Nombre d'exemples dans train_dataset : {len(train_dataset)}")

    # -------------------------------------------------
    # 5) DataLoader
    # -------------------------------------------------
    print("\n[STEP] Création du DataLoader...")
    t3 = time.time()
    batch_size = 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # à 0 pour éviter les soucis avec xarray/dask
    )
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=0)
    print(f"[DONE] DataLoader créé en {time.time() - t3:.1f} s")
    num_batches = len(train_loader)
    print(f"Nombre de batches par epoch : {num_batches}")

    # -------------------------------------------------
    # 6) Modèle ConvLSTM
    # -------------------------------------------------
    C_in = len(input_vars)
    print("\n[STEP] Initialisation du modèle PrecipConvLSTM...")
    t4 = time.time()
    model = PrecipConvLSTM(
        input_channels=C_in,
        hidden_channels=[32, 16],  # plus petit pour debug
        kernel_size=3,
    ).to(device)
    print(f"[DONE] Modèle initialisé en {time.time() - t4:.2f} s")
    print("Nombre de paramètres du modèle :",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    # -------------------------------------------------
    # 7) Loss & Optimizer
    # -------------------------------------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # -------------------------------------------------
    # 8) Boucle d'entraînement (sur les 2 semaines)
    # -------------------------------------------------
    n_epochs = 2  # pour debug
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

    csv_path = "checkpoints/validation_log.csv"
    os.makedirs("checkpoints", exist_ok=True)
    csv_header = ["epoch", "batch_idx", "eval_type", "loss"]

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    week_len = 56
    n_weeks = ds_val.sizes["time"] // week_len
    week_indices = list(range(n_weeks))
    while True:
        w1, w2 = random.sample(week_indices, 2)
    mini_val = xr.concat([
        ds_val.isel(time=slice(w1 * week_len, (w1 + 1) * week_len)),
        ds_val.isel(time=slice(w2 * week_len, (w2 + 1) * week_len)),
    ], dim="time")
    
    mini_val_dataset = ERA5Dataset(
        ds=mini_val,
        input_vars=input_vars,
        target_var=target_var,
        n_input_steps=5,
        lead_steps=1,
    )
    mini_val_loader = DataLoader(mini_val_dataset, batch_size=batch_size, shuffle=False)

    # fréquence pour ≈ 40 checkpoints par epoch
    checkpoint_every = max(1, num_batches // 40)

    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (X, y) in enumerate(train_loader):
            batch_start = time.time()

            # X : (B, T_in, C_in, H, W)
            # y : (B, H, W)
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            # Forward
            y_hat = model(X)        # (B, 1, H, W)
            y_hat = y_hat.squeeze(1)  # (B, H, W)

            # Loss
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            if (batch_idx + 1) % checkpoint_every == 0:
                model.eval()
                mini_losses = []
                with torch.no_grad():
                    for Xv, yv in mini_val_loader:
                        Xv, yv = Xv.to(device), yv.to(device)
                        yv_hat = model(Xv).squeeze(1)
                        mini_losses.append(criterion(yv_hat, yv).item())

                mini_val_loss = float(np.mean(mini_losses))

                # log CSV
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, batch_idx + 1, "mini_val", mini_val_loss])

                # checkpoint
                cp_path = f"checkpoints/epoch{epoch}_batch{batch_idx+1}.pt"
                torch.save({
                    "epoch": epoch,
                    "batch": batch_idx + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, cp_path)

                print(f"[CHECKPOINT-INTER] Epoch {epoch} Batch {batch_idx+1} - mini_val_loss: {mini_val_loss:.4e}")
                model.train()


            # Logs intermédiaires (toutes les 5 batches par ex.)
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
                batch_time = time.time() - batch_start
                print(
                    f"[Epoch {epoch}/{n_epochs}] "
                    f"Batch {batch_idx+1}/{num_batches} "
                    f"- Loss: {loss.item():.4e} "
                    f"- Batch time: {batch_time:.2f}s"
                )

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_batches
        print(
            f"\n>>> Epoch {epoch}/{n_epochs} terminée "
            f"- Loss moyen: {avg_loss:.4e} "
            f"- Temps epoch: {epoch_time:.1f}s\n"
        )
        
        # ----- Validation -----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(val_loader):
                Xv, yv = X.to(device), yv.to(device)
                yv_hat = model(X).squeeze(1)
                val_loss = criterion(yv_hat, yv)
                val_losses.append(val_loss.item())

        print(f"Epoch {epoch}, val loss: {np.mean(val_losses):.6f}")
        
        val_loss_full = float(np.mean(val_losses))

        # log CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, -1, "full_val", val_loss_full])
            
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
    print("\n[PRED] Prédiction sur quelques batches du train...")
    model.eval()

    with torch.no_grad():
        t_pred_start = time.time()
        for i, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)

            t_batch_pred_start = time.time()
            y_hat = model(X)           # (B, 1, H, W)
            y_hat = y_hat.squeeze(1)   # (B, H, W)
            t_batch_pred = time.time() - t_batch_pred_start

            # Afficher quelques stats sur la prédiction
            print(
                f"[PRED] Batch {i+1} "
                f"- y_hat shape: {tuple(y_hat.shape)} "
                f"- min: {float(y_hat.min()):.3f}, "
                f"max: {float(y_hat.max()):.3f}, "
                f"time: {t_batch_pred:.2f}s"
            )

            if i >= 2:
                # On s'arrête après 3 batches de prédiction pour debug
                break

        print(f"[PRED] Phase de prédiction terminée en {time.time() - t_pred_start:.1f} s")


if __name__ == "__main__":
    main()
