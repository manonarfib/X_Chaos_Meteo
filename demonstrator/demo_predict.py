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

    vmin = float(np.nanmin([y_true.min(), y_pred.min()]))
    vmax = float(np.nanmax([y_true.max(), y_pred.max()]))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im0 = axes[0].imshow(y_true, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{title_prefix}Truth tp_6h")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(y_pred, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{title_prefix}Pred tp_6h")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(err)  # erreur en échelle libre (ou fixe aussi)
    axes[2].set_title(f"{title_prefix}|Error|")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_boxplot(y_true, y_pred, out_path, title=""):
    """
    y_true, y_pred : np.ndarray de shape (H, W)
    """
    yt = y_true.flatten()
    yp = y_pred.flatten()

    # optionnel : enlever les NaNs
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.boxplot(
        [yt, yp],
        labels=["Truth", "Pred"],
        showfliers=True,
        patch_artist=True
    )

    ax.set_ylabel("Precipitation value")
    ax.set_title(title)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[FIG] Saved boxplot: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"
    T, lead = 8, 1
    batch_size = 8

    dataset = ERA5Dataset(dataset_path, T=T, lead=lead)    
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    input_vars = list(dataset.X.coords["channel"].values)
    C_in = len(input_vars)

    model = PrecipConvLSTM(
        input_channels=C_in,
        hidden_channels=[32, 64],
        kernel_size=3,
    ).to(device)

    ckpt_path = "checkpoints_w_mse/best_checkpoint_epoch1_batch364.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint epoch={ckpt.get('epoch', 'unknown')} from {ckpt_path}")

    batch = next(iter(val_loader))
    X, y = batch[0], batch[1]   # ignore le reste

    X = X.to(device, non_blocking=True).float()
    y = y.to(device, non_blocking=True).float()

    with torch.no_grad():
        y_hat = model(X).squeeze(1)  # (B,H,W)

    print("y min/max:", float(y.min()), float(y.max()))
    print("y_hat min/max:", float(y_hat.min()), float(y_hat.max()))

    
    mse = nn.MSELoss()(y_hat, y).item()
    mae = torch.mean(torch.abs(y_hat - y)).item()
    print(f"Demo metrics - MSE: {mse:.6f} | MAE: {mae:.6f}")

    y_true = y[0].detach().cpu().numpy()
    y_pred = y_hat[0].detach().cpu().numpy()
    save_maps(
        y_true,
        y_pred,
        out_path="demonstrator/demo_outputs/sample0_maps.png",
        title_prefix="Val sample 0 - "
    )
    save_boxplot(
        y_true,
        y_pred,
        out_path="demonstrator/demo_outputs/sample0_boxplot.png",
        title="Val sample 0 – distribution Truth vs Pred"
    )



if __name__ == "__main__":
    main()