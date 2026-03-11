#!/usr/bin/env python3
"""
Noise robustness by variable for ConvLSTM / UNet precipitation model.

Goal:
- perturb only ONE input variable at a time
- add spatially correlated Gaussian noise
- compare prediction before vs after perturbation
- rank variables by prediction sensitivity

Assumptions:
- ERA5Dataset returns X of shape (T,C,H,W)
- model expects (B,T,C,H,W)
- model outputs (B,1,H,W) or (B,H,W)
"""

import os
import json
import numpy as np
import torch
import matplotlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

from models.utils.ERA5_dataset_from_local import ERA5Dataset
from models.ConvLSTM.convlstm import PrecipConvLSTM
from models.unet.model_without_collapse import WFUNet_with_train


# ============================================================
# USER CONFIG
# ============================================================
MODEL_TYPE = "convlstm"   # "convlstm" or "unet"
LOSS_NAME  = "MSE"
CKPT_PATH  = "checkpoints/convlstm/mse/epoch3_full.pt"

DATASET_PATH = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"
SAMPLE_IDX = 250
T = 8
LEAD = 1

# Variable-noise experiment
NOISE_STD = 0.05

# smoothing in time and space
SMOOTHING_SIGMA_TIME = 1.0
SMOOTHING_SIGMA_SPACE = 2.0

SEED = 0

# Visualisation
TOP_K_VARS = 5
REGION = (-12.5, 42.5, 35, 72)  # lon_min, lon_max, lat_min, lat_max

# ============================================================


# ----------------------------
# Helpers
# ----------------------------
def lead_to_str(lead: int) -> str:
    return f"{lead * 6}h"


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cmap_with_white_bad(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color="white")
    return cmap


def _apply_mask(arr2d: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    out = arr2d.astype(np.float32).copy()
    out[~mask2d.astype(bool)] = np.nan
    return out


def build_model(model_type: str, C_in: int, T: int, device: torch.device) -> torch.nn.Module:
    model_type = model_type.lower().strip()
    if model_type == "convlstm":
        return PrecipConvLSTM(
            input_channels=C_in,
            hidden_channels=[32, 64],
            kernel_size=3,
        ).to(device)
    elif model_type == "unet":
        return WFUNet_with_train(T, 149, 221, C_in, 1, 8, 32, 0).to(device)
    else:
        raise ValueError(f"Unknown model_type={model_type}")


@torch.no_grad()
def predict_map(model, X):
    """
    Returns prediction as (B,H,W)
    """
    y_hat = model(X)

    if y_hat.dim() == 4:
        if y_hat.shape[1] == 1:
            y_hat = y_hat[:, 0]
        else:
            raise ValueError(f"Unexpected 4D output shape: {tuple(y_hat.shape)}")
    elif y_hat.dim() == 3:
        pass
    else:
        raise ValueError(f"Unexpected output shape: {tuple(y_hat.shape)}")

    return y_hat


def make_spatiotemporally_correlated_noise(
    shape,
    std,
    smoothing_sigma_time=1.0,
    smoothing_sigma_space=2.0,
):
    """
    shape = (B,T,1,H,W) or (B,T,C,H,W)

    Creates Gaussian noise smoothed over:
    - time dimension T
    - spatial dimensions H,W

    No smoothing across batch or channel dimensions.
    """
    if len(shape) != 5:
        raise ValueError(f"Expected 5D shape, got {shape}")

    noise_np = np.random.randn(*shape).astype(np.float32)
    out_np = np.empty_like(noise_np)

    for b in range(shape[0]):
        for c in range(shape[2]):
            # here noise_np[b, :, c] has shape (T,H,W)
            out_np[b, :, c] = gaussian_filter(
                noise_np[b, :, c],
                sigma=(smoothing_sigma_time, smoothing_sigma_space, smoothing_sigma_space)
            )

    current_std = out_np.std()
    if current_std > 1e-8:
        out_np = out_np / current_std * std
    else:
        out_np[:] = 0.0

    return out_np


def add_noise_to_single_variable(
    X,
    var_idx,
    std,
    smoothing_sigma_time=1.0,
    smoothing_sigma_space=2.0,
):
    """
    X: (B,T,C,H,W)
    Only variable var_idx is perturbed, with noise correlated
    in time and space.
    """
    X_pert = X.clone()

    noise_np = make_spatiotemporally_correlated_noise(
        shape=(X.shape[0], X.shape[1], 1, X.shape[3], X.shape[4]),
        std=std,
        smoothing_sigma_time=smoothing_sigma_time,
        smoothing_sigma_space=smoothing_sigma_space,
    )

    noise = torch.tensor(noise_np, dtype=X.dtype, device=X.device)
    X_pert[:, :, var_idx:var_idx+1, :, :] += noise
    return X_pert


def compute_metrics(base_pred, pert_pred):
    """
    base_pred, pert_pred: (B,H,W)
    """
    diff = pert_pred - base_pred
    abs_diff = diff.abs()

    return {
        "mean_abs_change": float(abs_diff.mean().item()),
        "max_abs_change": float(abs_diff.max().item()),
        "mse_change": float((diff ** 2).mean().item()),
    }


def save_barplot(results, out_path, title="", value_key="mean_abs_change", top_k=15):
    values = np.asarray([r[value_key] for r in results])
    labels = [r["variable"] for r in results]

    idx = np.argsort(-values)[:top_k]
    v = values[idx][::-1]
    l = [labels[i] for i in idx][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(v)), v, color="mediumseagreen")
    ax.set_yticks(range(len(v)))
    ax.set_yticklabels(l)
    ax.set_xlabel(value_key)
    ax.set_title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] Saved: {out_path}")


def save_triptych_europe(base_pred, pert_pred, europe_mask, out_path, title="",
                         region=(-12.5, 42.5, 35, 72)):
    base_m = _apply_mask(base_pred, europe_mask)
    pert_m = _apply_mask(pert_pred, europe_mask)
    diff_m = np.abs(pert_m - base_m)

    blues = _cmap_with_white_bad("Blues")
    reds = _cmap_with_white_bad("Reds")

    lon_min, lon_max, lat_min, lat_max = region
    extent = [lon_min, lon_max, lat_min, lat_max]
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(20, 6))
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

    im0 = ax0.imshow(base_m, cmap=blues, origin="upper", extent=extent, transform=proj)
    ax0.set_title("Base prediction")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)

    im1 = ax1.imshow(pert_m, cmap=blues, origin="upper", extent=extent, transform=proj)
    ax1.set_title("Prediction after variable noise")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

    im2 = ax2.imshow(diff_m, cmap=reds, origin="upper", extent=extent, transform=proj)
    ax2.set_title("|Difference|")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)

    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] Saved: {out_path}")


def save_results_json(results, out_path):
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[JSON] Saved: {out_path}")


# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- data ----
    dataset = ERA5Dataset(DATASET_PATH, T=T, lead=LEAD)
    input_vars = list(dataset.X.coords["channel"].values)
    C_in = len(input_vars)
    print("C_in:", C_in)

    # ---- model ----
    model = build_model(MODEL_TYPE, C_in=C_in, T=T, device=device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model={MODEL_TYPE} | loss={LOSS_NAME} | ckpt={CKPT_PATH}")

    # ---- output dir ----
    lead_str = lead_to_str(LEAD)
    out_dir = (
        f"explainability/noise/noise_by_variable/"
        f"model_{MODEL_TYPE}/{LOSS_NAME.lower()}_{lead_str}/sample{SAMPLE_IDX}"
    )
    os.makedirs(out_dir, exist_ok=True)
    print("[OUT_DIR]", out_dir)

    # ---- sample ----
    X, y, *_ = dataset[SAMPLE_IDX]
    X = X.unsqueeze(0).to(device).float()   # (1,T,C,H,W)
    print("X shape:", tuple(X.shape))

    europe_mask = np.isfinite(X[0, 0, 0].detach().cpu().numpy())

    # ---- base prediction ----
    with torch.no_grad():
        base_pred = predict_map(model, X)   # (1,H,W)
    base_map = base_pred[0].detach().cpu().numpy()

    # ---- loop over variables ----
    results = []

    for var_idx, var_name in enumerate(input_vars):
        print(f"\n[VAR] {var_idx:02d} | {var_name}")

        X_pert = add_noise_to_single_variable(
            X,
            var_idx=var_idx,
            std=NOISE_STD,
            smoothing_sigma_time=SMOOTHING_SIGMA_TIME,
            smoothing_sigma_space=SMOOTHING_SIGMA_SPACE,
        )

        with torch.no_grad():
            pert_pred = predict_map(model, X_pert)

        metrics = compute_metrics(base_pred, pert_pred)
        metrics["variable"] = str(var_name)
        metrics["var_idx"] = int(var_idx)
        metrics["noise_std"] = float(NOISE_STD)

        results.append(metrics)
        print(metrics)

    # ---- save full results ----
    save_results_json(results, os.path.join(out_dir, "variable_noise_results.json"))

    # ---- global ranking ----
    save_barplot(
        results,
        out_path=os.path.join(out_dir, "variable_noise_barplot_mean_abs_change.png"),
        title=(
            f"Prediction sensitivity to variable-wise correlated noise\n"
            f"model={MODEL_TYPE} loss={LOSS_NAME} sample={SAMPLE_IDX} std={NOISE_STD}"
        ),
        value_key="mean_abs_change",
        top_k=len(results),
    )

    # ---- top-k triptychs ----
    results_sorted = sorted(results, key=lambda x: x["mean_abs_change"], reverse=True)
    top_results = results_sorted[:TOP_K_VARS]

    for rank, r in enumerate(top_results, start=1):
        var_idx = r["var_idx"]
        var_name = r["variable"]

        X_pert = add_noise_to_single_variable(
            X,
            var_idx=var_idx,
            std=NOISE_STD,
            smoothing_sigma_time=SMOOTHING_SIGMA_TIME,
            smoothing_sigma_space=SMOOTHING_SIGMA_SPACE,
        )

        with torch.no_grad():
            pert_pred = predict_map(model, X_pert)

        pert_map = pert_pred[0].detach().cpu().numpy()

        save_triptych_europe(
            base_pred=base_map,
            pert_pred=pert_map,
            europe_mask=europe_mask,
            out_path=os.path.join(out_dir, f"top{rank:02d}_{var_name}_triptych.png"),
            title=(
                f"Top-{rank} sensitive variable: {var_name}\n"
                f"noise std={NOISE_STD} | model={MODEL_TYPE} | loss={LOSS_NAME} | sample={SAMPLE_IDX}"
            ),
            region=REGION,
        )

    print("\n[DONE] Outputs written to:", out_dir)


if __name__ == "__main__":
    main()