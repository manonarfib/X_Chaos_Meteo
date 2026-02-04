import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # backend non interactif (clusters)
import matplotlib.pyplot as plt

from models.utils.ERA5_dataset_from_local import ERA5Dataset
from models.ConvLSTM.convlstm import PrecipConvLSTM
from models.unet.model_without_collapse import WFUNet_with_train


# ============================================================
# USER CONFIG
# ============================================================
MODEL_TYPE = "unet"  # "convlstm" or "unet"
LOSS_NAME = "mse"        # e.g. "mse", "weighted_mse", "dice_weighted"
CKPT_PATH = "checkpoints/run_140292/best_checkpoint_epoch3_batch_idx5759.pt"  # or ".../best_checkpoint_epoch1_batch528.pt"
LEAD = 1  # lead in 6h steps -> prediction at t_lead = LEAD*6 hours
SAMPLE_IDX = 982
DATASET_PATH = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"

T = 8
BATCH_SIZE = 16
CLIP_NEG_PRED = True  # clamp predictions to >=0
# ============================================================


def save_maps(y_true, y_pred, out_path, title_prefix=""):
    err = np.abs(y_pred - y_true)

    vmin = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
    vmax = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im0 = axes[0].imshow(y_true, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{title_prefix}Truth tp_6h")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(y_pred, vmin=vmin, vmax=vmax)
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
    print(f"[FIG] Saved maps: {out_path}")


def save_boxplot(y_true, y_pred, out_path, title=""):
    yt = y_true.flatten()
    yp = y_pred.flatten()

    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.boxplot(
        [yt, yp],
        tick_labels=["Truth", "Pred"],
        showfliers=True,
        patch_artist=True
    )

    ax.set_ylabel("Precipitation value (mm/6h)")
    ax.set_title(title)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[FIG] Saved boxplot: {out_path}")


def build_model(model_type: str, C_in: int, device: torch.device) -> torch.nn.Module:
    model_type = model_type.lower().strip()
    if model_type == "convlstm":
        model = PrecipConvLSTM(
            input_channels=C_in,
            hidden_channels=[32, 64],
            kernel_size=3,
        ).to(device)
        return model
    elif model_type == "unet":
        # signature in your commented line: WFUNet_with_train(8,149,221,33,1, 8,32,0)
        # If you change T/H/W/C_in etc., update these args accordingly.
        model = WFUNet_with_train(T, 149, 221, C_in, 1, 8, 32, 0).to(device)
        return model
    else:
        raise ValueError(f"Unknown MODEL_TYPE='{model_type}'. Use 'convlstm' or 'unet'.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Derived
    t_lead = LEAD * 6  # hours
    ckpt_stem = os.path.splitext(os.path.basename(CKPT_PATH))[0]  # "epoch3_full", "best_checkpoint_epoch1_batch528", ...

    dataset = ERA5Dataset(DATASET_PATH, T=T, lead=LEAD)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    input_vars = list(dataset.X.coords["channel"].values)
    C_in = len(input_vars)
    print(f"Input channels: {C_in}")

    model = build_model(MODEL_TYPE, C_in, device)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint epoch={ckpt.get('epoch', 'unknown')} from {CKPT_PATH}")

    # Get one specific sample from dataset
    X, y, *_ = dataset[SAMPLE_IDX]
    X = X.unsqueeze(0).to(device).float()  # (1,T,C,H,W)
    y = y.unsqueeze(0).to(device).float()  # (1,H,W)

    with torch.no_grad():
        y_hat = model(X)
        # convlstm returns (B,1,H,W); unet might return (B,1,H,W) too -> handle both
        if y_hat.ndim == 4 and y_hat.shape[1] == 1:
            y_hat = y_hat.squeeze(1)  # (B,H,W)

        if CLIP_NEG_PRED:
            y_hat = torch.clamp(y_hat, min=0.0)

    print("y min/max:", float(y.min()), float(y.max()))
    print("y_hat min/max:", float(y_hat.min()), float(y_hat.max()))

    mse = nn.MSELoss()(y_hat, y).item()
    mae = torch.mean(torch.abs(y_hat - y)).item()
    print(f"Demo metrics - MSE: {mse:.6f} | MAE: {mae:.6f}")

    y_true = y[0].detach().cpu().numpy()
    y_pred = y_hat[0].detach().cpu().numpy()

    out_dir = f"demonstrator/demo_outputs/sample{SAMPLE_IDX}"
    maps_path = f"{out_dir}/{ckpt_stem}_maps_{LOSS_NAME}_prediction_{t_lead}h.png"
    box_path = f"{out_dir}/{ckpt_stem}_boxplot_{LOSS_NAME}_prediction_{t_lead}h.png"

    save_maps(
        y_true,
        y_pred,
        out_path=maps_path,
        title_prefix=f"Test sample {SAMPLE_IDX} - "
    )
    save_boxplot(
        y_true,
        y_pred,
        out_path=box_path,
        title=f"Test sample {SAMPLE_IDX} – distribution Truth vs Pred"
    )


if __name__ == "__main__":
    main()
