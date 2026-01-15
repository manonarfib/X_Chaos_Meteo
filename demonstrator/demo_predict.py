import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.utils.ERA5_dataset_from_local import  ERA5Dataset
from models.ConvLSTM.convlstm import PrecipConvLSTM


import os
import matplotlib
matplotlib.use("Agg")  # backend non interactif (clusters)


def save_maps(y_true, y_pred, out_path, title_prefix=""):
    err = np.abs(y_pred - y_true)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im0 = axes[0].imshow(y_true)
    axes[0].set_title(f"{title_prefix}Truth tp_6h")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(y_pred)
    axes[1].set_title(f"{title_prefix}Pred tp_6h")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(err)
    axes[2].set_title(f"{title_prefix}|Error|")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] Saved: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_validation.zarr"
    T, lead = 8, 1
    batch_size = 8

    dataset = ERA5Dataset(dataset_path, T=T, lead=lead)    
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    input_vars = list(dataset.X.coords["channel"].values)
    C_in = len(input_vars)

    model = PrecipConvLSTM(
        input_channels=C_in,
        hidden_channels=[32, 16],
        kernel_size=3,
    ).to(device)

    ckpt_path = "x_chaos/checkpoints/epoch40_full.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint epoch={ckpt.get('epoch', 'unknown')} from {ckpt_path}")

    X, y = next(iter(val_loader))
    X = X.to(device)
    y = y.to(device)

    with torch.no_grad():
        y_hat = model(X).squeeze(1)  # (B,H,W)

    mse = nn.MSELoss()(y_hat, y).item()
    mae = torch.mean(torch.abs(y_hat - y)).item()
    print(f"Demo metrics - MSE: {mse:.6f} | MAE: {mae:.6f}")

    y_true = y.squeeze(0).detach().cpu().numpy()
    y_pred = y_hat.squeeze(0).detach().cpu().numpy()
    save_maps(
        y_true,
        y_pred,
        out_path="demo_outputs/sample0_maps.png",
        title_prefix="Val sample 0 - "
    )


if __name__ == "__main__":
    main()
