import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # backend non interactif (clusters)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from models.utils.ERA5_dataset_from_local import ERA5Dataset
from models.ConvLSTM.convlstm import PrecipConvLSTM
from models.unet.model_without_collapse import WFUNet_with_train


# ============================================================
# USER CONFIG
# ============================================================
MODEL_TYPE = "convlstm"  # "convlstm" or "unet"
LOSS_NAME = "mse"        # e.g. "mse", "weighted_mse", "dice_weighted"
CKPT_PATH = "checkpoints_mse_48h/epoch3_full.pt"  # or ".../best_checkpoint_epoch1_batch528.pt"
LEAD = 8  # lead in 6h steps -> prediction at t_lead = LEAD*6 hours
SAMPLE_IDX = 982
DATASET_PATH = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"

T = 8
BATCH_SIZE = 16
CLIP_NEG_PRED = True  # clamp predictions to >=0
# ============================================================

def _cmap_with_white_bad(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color="white")
    return cmap

def _apply_mask(arr2d: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    out = arr2d.astype(np.float32).copy()
    out[~mask2d.astype(bool)] = np.nan
    return out

def save_maps_europe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    title_prefix: str = "",
    region=(-12.5, 42.5, 35, 72),
    error_mode: str = "abs",   # "abs" or "signed"
):
    """
    3 panels with Cartopy background:
      1) Truth (Blues)
      2) Pred  (Blues)
      3) Error (Reds for abs, bwr for signed)
    White outside Europe (mask from finite values).
    """

    # Mask: white outside Europe
    # (If your dataset already has NaNs outside Europe, this works immediately)
    europe_mask = np.isfinite(y_true) | np.isfinite(y_pred)

    yt = _apply_mask(y_true, europe_mask)
    yp = _apply_mask(y_pred, europe_mask)

    if error_mode == "abs":
        err = np.abs(yp - yt)
        err_cmap = _cmap_with_white_bad("Reds")
        err_title = "|Error|"
        err_vmin, err_vmax = None, None
    elif error_mode == "signed":
        err = (yp - yt)
        err_cmap = _cmap_with_white_bad("bwr")
        err_title = "Pred - Truth"
        m = np.nanmax(np.abs(err))
        err_vmin, err_vmax = -m, m
    else:
        raise ValueError("error_mode must be 'abs' or 'signed'")

    # Shared scale for truth/pred
    vmin = float(np.nanmin([np.nanmin(yt), np.nanmin(yp)]))
    vmax = float(np.nanmax([np.nanmax(yt), np.nanmax(yp)]))

    blues = _cmap_with_white_bad("Blues")

    lon_min, lon_max, lat_min, lat_max = region
    extent = [lon_min, lon_max, lat_min, lat_max]
    proj = ccrs.PlateCarree()

    # IMPORTANT: Cartopy + layout -> use constrained_layout (avoid tight_layout warnings)
    fig = plt.figure(figsize=(16, 5), constrained_layout=True)
    ax0 = fig.add_subplot(1, 3, 1, projection=proj)
    ax1 = fig.add_subplot(1, 3, 2, projection=proj)
    ax2 = fig.add_subplot(1, 3, 3, projection=proj)
    axes = [ax0, ax1, ax2]

    for ax in axes:
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.4)
        gl.right_labels = False
        gl.top_labels = False

    im0 = ax0.imshow(yt, cmap=blues, vmin=vmin, vmax=vmax, origin="upper", extent=extent, transform=proj)
    ax0.set_title(f"{title_prefix}Truth tp_6h")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)

    im1 = ax1.imshow(yp, cmap=blues, vmin=vmin, vmax=vmax, origin="upper", extent=extent, transform=proj)
    ax1.set_title(f"{title_prefix}Pred tp_6h")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

    im2 = ax2.imshow(err, cmap=err_cmap, vmin=err_vmin, vmax=err_vmax, origin="upper", extent=extent, transform=proj)
    ax2.set_title(f"{title_prefix}{err_title}")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
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

    save_maps_europe(
        y_true,
        y_pred,
        out_path=maps_path,
        title_prefix=f"Test sample {SAMPLE_IDX} - ",
        error_mode="abs",   # ou "signed"
    )

    save_boxplot(
        y_true,
        y_pred,
        out_path=box_path,
        title=f"Test sample {SAMPLE_IDX} – distribution Truth vs Pred"
    )


if __name__ == "__main__":
    main()
