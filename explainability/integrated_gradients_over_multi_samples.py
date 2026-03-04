#!/usr/bin/env python3
"""
Integrated Gradients demo + clean visualisations:
- variable importance (top-k)
- time importance
- per-variable attribution maps (top-5 variables)
- tp_6h input at t=7 in Blues with Europe mask (white outside)
- attribution in Reds (white outside)
- contour (top-5% attribution) over tp map for readability

Assumptions:
- ERA5Dataset returns X of shape (T,C,H,W) and y of shape (H,W) or (1,H,W)
- Model expects X as (B,T,C,H,W) and outputs (B,1,H,W) (or (B,H,W))
"""

import os
import time
import numpy as np
import torch
import matplotlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.utils.ERA5_dataset_from_local import ERA5Dataset
from models.ConvLSTM.convlstm import PrecipConvLSTM
from models.unet.model_without_collapse import WFUNet_with_train

# ============================================================
# USER CONFIG (no argparse, no inference)
# ============================================================
MODEL_TYPE = "convlstm"  # "convlstm" or "unet"
LOSS_NAME  = "MSE"       # purely for output folder naming
CKPT_PATH  = "checkpoints/convlstm/mse/epoch3_full.pt"

DATASET_PATH = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"
SAMPLE_IDX = 250
T = 8
LEAD = 1

# Aggregate attribution settings
DO_AGG = True
N_SAMPLES_AGG = 100
SEED = 0

# IG settings
METHOD = "ig"            # "ig" or "gradxinput"
IG_STEPS = 30
BASELINE_MODE = "zeros"  # "zeros" or "mean_over_space_time"
REGION_QUANTILE = 0.90

# Viz settings
T_VIEW = 7
CONTOUR_Q = 0.95
TOP_K_VARS = 5
# ============================================================

# ----------------------------
# Plot helpers
# ----------------------------
TIME_STEP_HOURS = 6  # ERA5 resolution

def rel_time_label(t_idx: int, t_ref: int) -> str:
    """
    t_ref = index du temps de référence (ex: t_view=7) qui correspond à 't' (0h).
    t_idx=t_ref   -> 't'
    t_idx=t_ref-1 -> 't-6h'
    t_idx=t_ref-2 -> 't-12h'
    ...
    """
    dh = (t_ref - t_idx) * TIME_STEP_HOURS
    return "t" if dh == 0 else f"t-{dh}h"

def lead_to_str(lead: int) -> str:
    """
    Your convention: lead=1 corresponds to 6h.
    If later you use lead in 6h-steps, this generalizes nicely.
    """
    return f"{lead * 6}h"

def _cmap_with_white_bad(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color="white")  # NaNs will render white
    return cmap


def _apply_mask(arr2d: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    out = arr2d.astype(np.float32).copy()
    out[~mask2d.astype(bool)] = np.nan
    return out


def save_barplot(values, labels, out_path, title="", top_k=15):
    values = np.asarray(values)
    idx = np.argsort(-values)[:top_k]
    v = values[idx][::-1]
    l = [labels[i] for i in idx][::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(v)), v, color='mediumseagreen')
    ax.set_yticks(range(len(v)))
    ax.set_yticklabels(l)
    ax.set_title(title)
    ax.set_xlabel("Importance (sum abs attribution)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] Saved: {out_path}")


def save_lineplot(values, out_path, title="", xlabel="t index (0..T-1)", ylabel="Importance (sum abs attribution)"):
    values = np.asarray(values)
    T = len(values)
    t_ref = T - 1  # on suppose que le dernier index correspond à 't' (comme ton t_view=7 quand T=8)
    hours = -TIME_STEP_HOURS * (t_ref - np.arange(T))  # [-42, -36, ..., -6, 0] si T=8

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hours, values, marker="o", color='mediumseagreen')
    ax.set_title(title)
    ax.set_xlabel("Lead time relative to prediction (hours)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5)

    # ticks explicites
    ax.set_xticks(hours)
    ax.set_xticklabels(["t" if h == 0 else f"t{int(h)}h" for h in hours])

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] Saved: {out_path}")


def save_barplot_mean_std(mean_vals, std_vals, labels, out_path, title="", top_k=15):
    mean_vals = np.asarray(mean_vals)
    std_vals = np.asarray(std_vals)

    idx = np.argsort(-mean_vals)[:top_k]
    m = mean_vals[idx][::-1]
    s = std_vals[idx][::-1]
    l = [labels[i] for i in idx][::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(m)), m, xerr=s, color='mediumseagreen')
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(l)
    ax.set_title(title)
    ax.set_xlabel("Importance (mean ± std over samples)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] Saved: {out_path}")


def save_lineplot_mean_std(mean_vals, std_vals, out_path, title="", xlabel="t index", ylabel="Importance"):
    mean_vals = np.asarray(mean_vals)
    std_vals = np.asarray(std_vals)

    T = len(mean_vals)
    t_ref = T - 1
    hours = -TIME_STEP_HOURS * (t_ref - np.arange(T))


    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hours, mean_vals, marker="o", color='mediumseagreen')
    ax.fill_between(hours, mean_vals - std_vals, mean_vals + std_vals, alpha=0.25)

    ax.set_title(title)
    ax.set_xlabel("Lead time relative to prediction (hours)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5)

    ax.set_xticks(hours)
    ax.set_xticklabels(["t" if h == 0 else f"t{int(h)}h" for h in hours])


    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] Saved: {out_path}")

def plot_tp_attr_contour(
    tp_map: np.ndarray,
    attr_map: np.ndarray,
    europe_mask: np.ndarray,
    out_path: str,
    title: str,
    tp_title: str,
    attr_title: str,
    contour_q: float = 0.95,
    region=(-12.5, 42.5, 35, 72),  # (lon_min, lon_max, lat_min, lat_max)
):
    """
    3 panels:
      1) tp (Blues) + coastlines/borders
      2) attribution (Reds) + coastlines/borders
      3) tp + contour(top-5% attribution) + coastlines/borders

    White outside Europe (mask).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tp_m = _apply_mask(tp_map, europe_mask)
    at_m = _apply_mask(attr_map, europe_mask)

    blues = _cmap_with_white_bad("Blues")
    reds = _cmap_with_white_bad("Reds")

    lon_min, lon_max, lat_min, lat_max = region
    extent = [lon_min, lon_max, lat_min, lat_max]
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(20, 9))
    ax0 = fig.add_subplot(1, 3, 1, projection=proj)
    ax1 = fig.add_subplot(1, 3, 2, projection=proj)
    ax2 = fig.add_subplot(1, 3, 3, projection=proj)
    axes = [ax0, ax1, ax2]

    # Add map features (the “Europe map underneath”)
    for ax in axes:
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.4)
        gl.right_labels = False
        gl.top_labels = False

    # --- Panel 1: TP ---
    im0 = ax0.imshow(tp_m, cmap=blues, origin="upper", extent=extent, transform=proj)
    ax0.set_title(tp_title)
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)

    # --- Panel 2: Attribution ---
    im1 = ax1.imshow(at_m, cmap=reds, origin="upper", extent=extent, transform=proj)
    ax1.set_title(attr_title)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

    # --- Panel 3: TP + contour attribution ---
    ax2.imshow(tp_m, cmap=blues, origin="upper", extent=extent, transform=proj)

    flat = at_m[np.isfinite(at_m)]
    if flat.size > 0:
        thr = np.quantile(flat, contour_q)
        contour_mask = np.zeros_like(at_m, dtype=np.float32)
        contour_mask[np.isfinite(at_m) & (at_m >= thr)] = 1.0

        # --- FIX orientation: build lon/lat grid and contour in geocoords ---
        lon_min, lon_max, lat_min, lat_max = region
        lons = np.linspace(lon_min, lon_max, contour_mask.shape[1])
        lats = np.linspace(lat_max, lat_min, contour_mask.shape[0])  # IMPORTANT: lat_max -> lat_min
        Lon, Lat = np.meshgrid(lons, lats)

        ax2.contour(
            Lon, Lat, contour_mask,
            levels=[0.5],
            colors="red",        # <- red contour
            linewidths=1.2,
            transform=proj,
        )

    ax2.set_title(f"{tp_title} + top-{int((1-contour_q)*100)}% attr contour")

    fig.suptitle(title) 
    plt.tight_layout() 
    plt.savefig(out_path, dpi=200, bbox_inches="tight") 
    plt.close(fig)
    print(f"[FIG] Saved: {out_path}")

def plot_tp_var_times_attr_contour(
    tp_map: np.ndarray,
    var_maps: list,           # list of 2D arrays [t=7, t=6, t=5]
    var_name: str,
    attr_map: np.ndarray,
    europe_mask: np.ndarray,
    out_path: str,
    title: str,
    tp_title: str,
    attr_title: str,
    var_titles: list,         # ["var t=7", "var t=6", "var t=5"]
    contour_q: float = 0.95,
    region=(-12.5, 42.5, 35, 72),
    var_cmap: str = "cividis",
):
    """
    Layout:
      Col 1: TP(t=7)
      Col 2: Variable at t=7,6,5 stacked vertically
      Col 3: Attribution (for variable)
      Col 4: TP(t=7) + contour(top-5% attribution)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Masks (NaN outside Europe)
    tp_m = _apply_mask(tp_map, europe_mask)
    var_ms = [_apply_mask(v, europe_mask) for v in var_maps]
    at_m = _apply_mask(attr_map, europe_mask)

    blues = _cmap_with_white_bad("Blues")
    reds = _cmap_with_white_bad("Reds")
    varcmap = _cmap_with_white_bad(var_cmap)

    lon_min, lon_max, lat_min, lat_max = region
    extent = [lon_min, lon_max, lat_min, lat_max]
    proj = ccrs.PlateCarree()

    # GridSpec: 3 rows, 4 columns
    # Col 0: tp spans 3 rows
    # Col 1: var t=7 / t=6 / t=5 (3 rows)
    # Col 2: attribution spans 3 rows
    # Col 3: tp+contour spans 3 rows
    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(nrows=3, ncols=4, width_ratios=[1.0, 1.0, 1.0, 1.0], wspace=0.25, hspace=0.15)

    ax_tp   = fig.add_subplot(gs[:, 0], projection=proj)
    ax_v7   = fig.add_subplot(gs[0, 1], projection=proj)
    ax_v6   = fig.add_subplot(gs[1, 1], projection=proj)
    ax_v5   = fig.add_subplot(gs[2, 1], projection=proj)
    ax_attr = fig.add_subplot(gs[:, 2], projection=proj)
    ax_tp_c = fig.add_subplot(gs[:, 3], projection=proj)

    axes = [ax_tp, ax_v7, ax_v6, ax_v5, ax_attr, ax_tp_c]

    # Map features
    for ax in axes:
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
        gl = ax.gridlines(draw_labels=False, linestyle="--", linewidth=0.35)
        gl.right_labels = False
        gl.top_labels = False

    # --- Col 1: TP ---
    im_tp = ax_tp.imshow(tp_m, cmap=blues, origin="upper", extent=extent, transform=proj)
    ax_tp.set_title(tp_title)
    plt.colorbar(im_tp, ax=ax_tp, fraction=0.046, pad=0.02)

    # --- Col 2: Variable times ---
    ims = []
    for ax, vm, vt in zip([ax_v7, ax_v6, ax_v5], var_ms, var_titles):
        imv = ax.imshow(vm, cmap=varcmap, origin="upper", extent=extent, transform=proj)
        ax.set_title(vt)
        ims.append(imv)

    # One shared colorbar for the 3 var maps (using the first one)
    plt.colorbar(ims[0], ax=[ax_v7, ax_v6, ax_v5], fraction=0.046, pad=0.02)

    # --- Col 3: Attribution ---
    im_at = ax_attr.imshow(at_m, cmap=reds, origin="upper", extent=extent, transform=proj)
    ax_attr.set_title(attr_title)
    plt.colorbar(im_at, ax=ax_attr, fraction=0.046, pad=0.02)

    # --- Col 4: TP + contour ---
    ax_tp_c.imshow(tp_m, cmap=blues, origin="upper", extent=extent, transform=proj)

    flat = at_m[np.isfinite(at_m)]
    if flat.size > 0:
        thr = np.quantile(flat, contour_q)
        contour_mask = np.zeros_like(at_m, dtype=np.float32)
        contour_mask[np.isfinite(at_m) & (at_m >= thr)] = 1.0

        # contour in lon/lat to avoid inversion
        lons = np.linspace(lon_min, lon_max, contour_mask.shape[1])
        lats = np.linspace(lat_max, lat_min, contour_mask.shape[0])
        Lon, Lat = np.meshgrid(lons, lats)

        ax_tp_c.contour(
            Lon, Lat, contour_mask,
            levels=[0.5],
            colors="red",
            linewidths=1.2,
            transform=proj,
        )

    ax_tp_c.set_title(f"{tp_title} + top-{int((1-contour_q)*100)}% attr contour")

    fig.suptitle(f"{title}\nVariable: {var_name}", y=0.98) 
    plt.tight_layout() 
    plt.savefig(out_path, dpi=200, bbox_inches="tight") 
    plt.close(fig) 
    print(f"[FIG] Saved: {out_path}")

# ----------------------------
# Attribution helpers
# ----------------------------

def build_model(model_type: str, C_in: int, T: int, device: torch.device) -> torch.nn.Module:
    model_type = model_type.lower().strip()
    if model_type == "convlstm":
        return PrecipConvLSTM(
            input_channels=C_in,
            hidden_channels=[32, 64],
            kernel_size=3,
        ).to(device)
    elif model_type == "unet":
        # garde exactement tes params UNet utilisés ailleurs
        return WFUNet_with_train(T, 149, 221, C_in, 1, 8, 32, 0).to(device)
    else:
        raise ValueError(f"Unknown model_type={model_type}")
    
def find_precip_channel(input_vars):
    """
    Try to locate precip channel in input variables.
    Adjust candidates if needed.
    """
    candidates = ["tp_6h", "tp", "total_precipitation", "precip", "precipitation"]
    for cand in candidates:
        for i, name in enumerate(input_vars):
            if name == cand or cand in name:
                return i, name
    return None, None


@torch.no_grad()
def make_baseline(x, mode="zeros"):
    """
    x: (B,T,C,H,W)
    """
    if mode == "zeros":
        return torch.zeros_like(x)
    if mode == "mean_over_space_time":
        # constant per-channel baseline: mean over (T,H,W)
        b = x.mean(dim=(1, 3, 4), keepdim=True)  # (B,1,C,1,1)
        return b.expand_as(x).clone()
    raise ValueError(f"Unknown baseline mode: {mode}")


def integrated_gradients(model, x, baseline, steps=30, target="region_sum", region_mask=None):
    """
    x: (B,T,C,H,W)
    baseline: (B,T,C,H,W)
    region_mask: (B,1,H,W) if target="region_sum"
    returns attr: (B,T,C,H,W)
    """
    assert x.shape == baseline.shape
    B, T, C, H, W = x.shape

    if target == "region_sum" and region_mask is None:
        region_mask = torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype)

    x = x.float()
    baseline = baseline.float()
    total_grad = torch.zeros_like(x, dtype=torch.float32)

    for s in range(1, steps + 1):
        alpha = s / steps
        x_alpha = baseline + alpha * (x - baseline)
        x_alpha.requires_grad_(True)

        y_hat = model(x_alpha)
        if y_hat.dim() == 3:
            y_hat = y_hat.unsqueeze(1)

        if target == "mean":
            S = y_hat.mean()
        elif target == "region_sum":
            S = (y_hat * region_mask).sum()
        else:
            raise ValueError(f"Unknown target: {target}")

        model.zero_grad(set_to_none=True)
        if x_alpha.grad is not None:
            x_alpha.grad.zero_()
        S.backward()

        total_grad += x_alpha.grad.detach()

    avg_grad = total_grad / float(steps)
    attr = (x - baseline) * avg_grad
    return attr


def grad_x_input(model, x, target="region_sum", region_mask=None):
    """
    Faster alternative (1 backward). Useful for quick iteration.
    x: (B,T,C,H,W)
    returns attr: (B,T,C,H,W)
    """
    B, T, C, H, W = x.shape
    if target == "region_sum" and region_mask is None:
        region_mask = torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype)

    x_req = x.clone().detach().float().requires_grad_(True)
    y_hat = model(x_req)
    if y_hat.dim() == 3:
        y_hat = y_hat.unsqueeze(1)

    if target == "mean":
        S = y_hat.mean()
    elif target == "region_sum":
        S = (y_hat * region_mask).sum()
    else:
        raise ValueError(f"Unknown target: {target}")

    model.zero_grad(set_to_none=True)
    S.backward()
    return x_req.grad.detach() * x_req.detach()

def compute_importance_over_random_samples(
    model,
    dataset,
    input_vars,
    device,
    n_samples=100,
    seed=0,
    method="ig",
    steps=30,
    baseline_mode="zeros",
    region_quantile=0.90,
):
    """
    Returns:
      var_mean, var_std  (C,)
      time_mean, time_std (T,)
      chosen_indices (n_samples,)
    """
    rng = np.random.default_rng(seed)
    N = len(dataset)
    chosen = rng.choice(N, size=min(n_samples, N), replace=False)

    C = len(input_vars)
    T = dataset.T if hasattr(dataset, "T") else None  # fallback
    # We'll infer T from first sample
    X0, *_ = dataset[int(chosen[0])]
    T = X0.shape[0]

    var_all = []
    time_all = []

    model.eval()

    for k, idx in enumerate(chosen, start=1):
        X, y, *_ = dataset[int(idx)]
        X = X.unsqueeze(0).to(device).float()  # (1,T,C,H,W)

        # predict -> define region mask from prediction
        with torch.no_grad():
            y_hat = model(X)
            if y_hat.dim() == 3:
                y_hat = y_hat.unsqueeze(1)  # (1,1,H,W)

            pred_map = y_hat[0, 0]
            thresh = torch.quantile(pred_map.flatten(), region_quantile)
            region = (pred_map >= thresh).float()
            region_mask = region.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # attribution
        if method == "ig":
            baseline = make_baseline(X, mode=baseline_mode)
            attr = integrated_gradients(
                model=model,
                x=X,
                baseline=baseline,
                steps=steps,
                target="region_sum",
                region_mask=region_mask,
            )
        elif method == "gradxinput":
            attr = grad_x_input(
                model=model,
                x=X,
                target="region_sum",
                region_mask=region_mask,
            )
        else:
            raise ValueError("method must be 'ig' or 'gradxinput'")

        attr_abs = attr.abs()  # (1,T,C,H,W)

        # var importance: sum over T,H,W -> (C,)
        var_imp = attr_abs.sum(dim=(1, 3, 4))[0].detach().cpu().numpy()
        # time importance: sum over C,H,W -> (T,)
        time_imp = attr_abs.sum(dim=(2, 3, 4))[0].detach().cpu().numpy()

        var_all.append(var_imp)
        time_all.append(time_imp)

        if k % 10 == 0:
            print(f"[AGG] {k}/{len(chosen)} samples processed")

    var_all = np.stack(var_all, axis=0)   # (N,C)
    time_all = np.stack(time_all, axis=0) # (N,T)

    return (
        var_all.mean(axis=0), var_all.std(axis=0),
        time_all.mean(axis=0), time_all.std(axis=0),
        chosen
    )

# ----------------------------
# Main
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_type = MODEL_TYPE.lower().strip()

    # ---- data ----
    dataset = ERA5Dataset(DATASET_PATH, T=T, lead=LEAD)
    input_vars = list(dataset.X.coords["channel"].values)
    C_in = len(input_vars)
    print("C_in:", C_in)

    # ---- model ----
    model = build_model(model_type, C_in=C_in, T=T, device=device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model={model_type} | loss={LOSS_NAME} | ckpt={CKPT_PATH}")

    # ---- output dir ----
    lead_str = lead_to_str(LEAD)
    model_tag = model_type
    loss_tag = LOSS_NAME.lower()
    out_dir = f"explainability/ig_outputs/model_{model_tag}/{loss_tag}_{lead_str}"
    os.makedirs(out_dir, exist_ok=True)
    print("[OUT_DIR]", out_dir)

    # ---- aggregate importance ----
    if DO_AGG:
        agg_dir = os.path.join(out_dir, f"aggregate_{METHOD}_{N_SAMPLES_AGG}")
        os.makedirs(agg_dir, exist_ok=True)

        v_mean, v_std, t_mean, t_std, chosen_idx = compute_importance_over_random_samples(
            model=model,
            dataset=dataset,
            input_vars=input_vars,
            device=device,
            n_samples=N_SAMPLES_AGG,
            seed=SEED,
            method=METHOD,
            steps=IG_STEPS,
            baseline_mode=BASELINE_MODE,
            region_quantile=REGION_QUANTILE,
        )

        np.save(os.path.join(agg_dir, "chosen_indices.npy"), chosen_idx)

        save_barplot(
            v_mean,
            labels=input_vars,
            out_path=os.path.join(agg_dir, "var_importance_mean.png"),
            title=f"{METHOD.upper()} variable importance (MEAN over {len(chosen_idx)} samples)\nmodel={model_tag} loss={LOSS_NAME} lead={lead_str}",
            top_k=20,
        )
        save_barplot_mean_std(
            v_mean, v_std,
            labels=input_vars,
            out_path=os.path.join(agg_dir, "var_importance_mean_std.png"),
            title=f"{METHOD.upper()} variable importance (mean ± std over {len(chosen_idx)} samples)\nmodel={model_tag} loss={LOSS_NAME} lead={lead_str}",
            top_k=20,
        )
        save_lineplot(
            t_mean,
            out_path=os.path.join(agg_dir, "time_importance_mean.png"),
            title=f"{METHOD.upper()} time importance (MEAN over {len(chosen_idx)} samples)\nmodel={model_tag} loss={LOSS_NAME} lead={lead_str}",
            xlabel="t index in input window (0..T-1, past→present)",
            ylabel="Importance (mean sum abs attribution)",
        )
        save_lineplot_mean_std(
            t_mean, t_std,
            out_path=os.path.join(agg_dir, "time_importance_mean_std.png"),
            title=f"{METHOD.upper()} time importance (mean ± std over {len(chosen_idx)} samples)\nmodel={model_tag} loss={LOSS_NAME} lead={lead_str}",
            xlabel="t index in input window (0..T-1, past→present)",
            ylabel="Importance (sum abs attribution)",
        )

        print("[AGG DONE] Aggregate plots written to:", agg_dir)
    else:
        print("[AGG SKIPPED] DO_AGG=False")

    # ---- sample ----
    sample_idx = SAMPLE_IDX
    X, y, *_ = dataset[sample_idx]
    X = X.unsqueeze(0).to(device).float()  # (1,T,C,H,W)
    y = y.unsqueeze(0).to(device).float()
    if y.dim() == 3:
        y = y.unsqueeze(1)

    # ---- predict ----
    with torch.no_grad():
        y_hat = model(X)
        if y_hat.dim() == 3:
            y_hat = y_hat.unsqueeze(1)

    print("y_hat shape:", tuple(y_hat.shape))

    # ---- define region ----
    with torch.no_grad():
        pred_map = y_hat[0, 0]
        thresh = torch.quantile(pred_map.flatten(), REGION_QUANTILE)
        region = (pred_map >= thresh).float()
        region_mask = region.unsqueeze(0).unsqueeze(0)

    # ---- attribution ----
    if METHOD == "ig":
        baseline = make_baseline(X, mode=BASELINE_MODE)
        t0 = time.time()
        attr = integrated_gradients(
            model=model,
            x=X,
            baseline=baseline,
            steps=IG_STEPS,
            target="region_sum",
            region_mask=region_mask,
        )
        print(f"Computed IG in {time.time()-t0:.2f}s (steps={IG_STEPS}, baseline={BASELINE_MODE})")
    elif METHOD == "gradxinput":
        t0 = time.time()
        attr = grad_x_input(model=model, x=X, target="region_sum", region_mask=region_mask)
        print(f"Computed grad×input in {time.time()-t0:.2f}s")
    else:
        raise ValueError("METHOD must be 'ig' or 'gradxinput'")

    attr_abs = attr.abs()

    # ---- summaries ----
    var_importance = attr_abs.sum(dim=(1, 3, 4))[0].detach().cpu().numpy()
    time_importance = attr_abs.sum(dim=(2, 3, 4))[0].detach().cpu().numpy()

    sample_dir = os.path.join(out_dir, f"sample{sample_idx}")
    os.makedirs(sample_dir, exist_ok=True)

    # ---- plots: importance ----
    save_barplot(
        var_importance,
        labels=input_vars,
        out_path=os.path.join(sample_dir, f"{METHOD}_var_importance.png"),
        title=f"{METHOD.upper()} variable importance (sum abs over T,H,W)\nSample {sample_idx} | model={model_tag} loss={LOSS_NAME} lead={lead_str}",
        top_k=15,
    )
    save_lineplot(
        time_importance,
        out_path=os.path.join(sample_dir, f"{METHOD}_time_importance.png"),
        title=f"{METHOD.upper()} time importance (sum abs over C,H,W)\nSample {sample_idx} | model={model_tag} loss={LOSS_NAME} lead={lead_str}",
        xlabel="t index in input window (0..T-1, past→present)",
        ylabel="Importance (sum abs attribution)",
    )

    # ---- get tp input at t_view ----
    tp_idx, tp_name = find_precip_channel(input_vars)
    if tp_idx is None:
        raise RuntimeError("No precip channel found in input_vars. Update find_precip_channel().")

    tp_in = X[0, T_VIEW, tp_idx].detach().cpu().numpy()  # (H,W)
    europe_mask = np.isfinite(tp_in)

    # ---- top-k variable visualisations ----
    top_idx = np.argsort(-var_importance)[:TOP_K_VARS]

    spatial_all = attr_abs.sum(dim=(1, 2))[0].detach().cpu().numpy()
    plot_tp_attr_contour(
        tp_map=tp_in,
        attr_map=spatial_all,
        europe_mask=europe_mask,
        out_path=os.path.join(sample_dir, f"{METHOD}_ALL_tp_t{T_VIEW}_contour.png"),
        title=f"Sample {sample_idx} | ALL vars (sum over T,C) | method={METHOD} | model={model_tag} loss={LOSS_NAME}",
        tp_title=f"{tp_name} input (t={T_VIEW})",
        attr_title=f"{METHOD.upper()} abs attribution (sum over T,C)",
        contour_q=CONTOUR_Q,
    )

    for rank, c_idx in enumerate(top_idx, start=1):
        var_name = input_vars[c_idx]
        attr_c = attr_abs[0, :, c_idx].sum(dim=0).detach().cpu().numpy()  # (H,W)

        t_idxs = [T_VIEW, T_VIEW - 1, T_VIEW - 2]
        var_t0 = X[0, t_idxs[0], c_idx].detach().cpu().numpy()
        var_t1 = X[0, t_idxs[1], c_idx].detach().cpu().numpy()
        var_t2 = X[0, t_idxs[2], c_idx].detach().cpu().numpy()

        t_ref = T_VIEW
        var_titles = [
            f"{var_name} input ({rel_time_label(t_idxs[0], t_ref)})",
            f"{var_name} input ({rel_time_label(t_idxs[1], t_ref)})",
            f"{var_name} input ({rel_time_label(t_idxs[2], t_ref)})",
        ]

        plot_tp_var_times_attr_contour(
            tp_map=tp_in,
            var_maps=[var_t0, var_t1, var_t2],
            var_name=var_name,
            attr_map=attr_c,
            europe_mask=europe_mask,
            out_path=os.path.join(sample_dir, f"{METHOD}_top{rank}_{var_name}_tp_t{T_VIEW}.png"),
            title=f"Sample {sample_idx} | Top-{rank} variable: {var_name} | method={METHOD} | model={model_tag} loss={LOSS_NAME}",
            tp_title=f"{tp_name} input (t)",
            attr_title=f"{METHOD.upper()} abs attribution for {var_name} (sum over T)",
            var_titles=var_titles,
            contour_q=CONTOUR_Q,
            var_cmap="cividis",
        )

    print("[DONE] Outputs written to:", out_dir)


if __name__ == "__main__":
    main()