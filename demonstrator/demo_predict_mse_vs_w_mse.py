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
CKPT_PATH_MSE = "checkpoints_mse/epoch3_full.pt"
CKPT_PATH_WMSE = "checkpoints_w_mse/epoch3_full.pt"
LEAD = 1  # lead in 6h steps -> prediction at t_lead = LEAD*6 hours
SAMPLE_IDX = 250
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

def save_maps_europe_3(
    y_true: np.ndarray,
    y_pred_mse: np.ndarray,
    y_pred_wmse: np.ndarray,
    out_path: str,
    title_prefix: str = "",
    region=(-12.5, 42.5, 35, 72),
    tp_cmap_name: str = "cividis",   # mieux que Blues pour petites pluies
):
    """
    3 panels with Cartopy background:
      1) Truth
      2) Pred (MSE)
      3) Pred (W_MSE)
    White outside Europe.
    Shared color scale across all three.
    """
    # Mask: white outside Europe
    europe_mask = np.isfinite(y_true) | np.isfinite(y_pred_mse) | np.isfinite(y_pred_wmse)

    yt  = _apply_mask(y_true, europe_mask)
    ym  = _apply_mask(y_pred_mse, europe_mask)
    ywm = _apply_mask(y_pred_wmse, europe_mask)

    # Shared scale for fair comparison
    vmin = float(np.nanmin([np.nanmin(yt), np.nanmin(ym), np.nanmin(ywm)]))
    vmax = float(np.nanmax([np.nanmax(yt), np.nanmax(ym), np.nanmax(ywm)]))

    tp_cmap = _cmap_with_white_bad(tp_cmap_name)

    lon_min, lon_max, lat_min, lat_max = region
    extent = [lon_min, lon_max, lat_min, lat_max]
    proj = ccrs.PlateCarree()

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

    im0 = ax0.imshow(yt,  cmap=tp_cmap, vmin=vmin, vmax=vmax, origin="upper", extent=extent, transform=proj)
    ax0.set_title(f"{title_prefix}Truth tp_6h")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)

    im1 = ax1.imshow(ym,  cmap=tp_cmap, vmin=vmin, vmax=vmax, origin="upper", extent=extent, transform=proj)
    ax1.set_title(f"{title_prefix}Pred (MSE) tp_6h")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

    im2 = ax2.imshow(ywm, cmap=tp_cmap, vmin=vmin, vmax=vmax, origin="upper", extent=extent, transform=proj)
    ax2.set_title(f"{title_prefix}Pred (W_MSE) tp_6h")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[FIG] Saved maps: {out_path}")


def save_boxplot_3(y_true, y_pred_mse, y_pred_wmse, out_path, title=""):
    yt = y_true.flatten()
    ym = y_pred_mse.flatten()
    yw = y_pred_wmse.flatten()

    mask = np.isfinite(yt) & np.isfinite(ym) & np.isfinite(yw)
    yt, ym, yw = yt[mask], ym[mask], yw[mask]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(
        [yt, ym, yw],
        tick_labels=["Truth", "MSE", "W_MSE"],
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

    t_lead = LEAD * 6  # hours

    dataset = ERA5Dataset(DATASET_PATH, T=T, lead=LEAD)
    input_vars = list(dataset.X.coords["channel"].values)
    C_in = len(input_vars)
    print(f"Input channels: {C_in}")

    # Sample
    X, y, *_ = dataset[SAMPLE_IDX]
    X = X.unsqueeze(0).to(device).float()  # (1,T,C,H,W)
    y = y.unsqueeze(0).to(device).float()  # (1,H,W)

    # --- Model MSE ---
    model_mse = build_model(MODEL_TYPE, C_in, device)
    ckpt_mse = torch.load(CKPT_PATH_MSE, map_location=device)
    model_mse.load_state_dict(ckpt_mse["model_state_dict"])
    model_mse.eval()

    # --- Model W_MSE ---
    model_wmse = build_model(MODEL_TYPE, C_in, device)
    ckpt_wmse = torch.load(CKPT_PATH_WMSE, map_location=device)
    model_wmse.load_state_dict(ckpt_wmse["model_state_dict"])
    model_wmse.eval()

    # Predict
    with torch.no_grad():
        y_hat_mse = model_mse(X)
        if y_hat_mse.ndim == 4 and y_hat_mse.shape[1] == 1:
            y_hat_mse = y_hat_mse.squeeze(1)

        y_hat_wmse = model_wmse(X)
        if y_hat_wmse.ndim == 4 and y_hat_wmse.shape[1] == 1:
            y_hat_wmse = y_hat_wmse.squeeze(1)

        if CLIP_NEG_PRED:
            y_hat_mse = torch.clamp(y_hat_mse, min=0.0)
            y_hat_wmse = torch.clamp(y_hat_wmse, min=0.0)

    # Metrics (optional)
    mse_mse = nn.MSELoss()(y_hat_mse, y).item()
    mse_wmse = nn.MSELoss()(y_hat_wmse, y).item()
    print(f"MSE model -> MSE: {mse_mse:.6f}")
    print(f"W_MSE model -> MSE: {mse_wmse:.6f}")

    y_true = y[0].detach().cpu().numpy()
    y_pred_mse = y_hat_mse[0].detach().cpu().numpy()
    y_pred_wmse = y_hat_wmse[0].detach().cpu().numpy()

    # Output paths (new folder)
    ckpt_mse_stem = os.path.splitext(os.path.basename(CKPT_PATH_MSE))[0]
    ckpt_wmse_stem = os.path.splitext(os.path.basename(CKPT_PATH_WMSE))[0]

    out_dir = f"demonstrator/demo_compare_outputs/sample{SAMPLE_IDX}/lead{t_lead}h"
    maps_path = f"{out_dir}/maps_truth_vs_mse_vs_wmse_{ckpt_mse_stem}_AND_{ckpt_wmse_stem}.png"
    box_path  = f"{out_dir}/boxplot_truth_vs_mse_vs_wmse_{ckpt_mse_stem}_AND_{ckpt_wmse_stem}.png"

    save_maps_europe_3(
        y_true=y_true,
        y_pred_mse=y_pred_mse,
        y_pred_wmse=y_pred_wmse,
        out_path=maps_path,
        title_prefix=f"Test sample {SAMPLE_IDX} ({t_lead}h) - ",
        tp_cmap_name="Blues", 
    )

    save_boxplot_3(
        y_true=y_true,
        y_pred_mse=y_pred_mse,
        y_pred_wmse=y_pred_wmse,
        out_path=box_path,
        title=f"Sample {SAMPLE_IDX} ({t_lead}h) – Truth vs MSE vs W_MSE"
    )

if __name__ == "__main__":
    main()