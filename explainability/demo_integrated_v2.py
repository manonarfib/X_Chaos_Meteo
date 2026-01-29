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


# ----------------------------
# Plot helpers
# ----------------------------
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
    ax.barh(range(len(v)), v)
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
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(values)), values, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5)
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

    fig = plt.figure(figsize=(15, 4))
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


# ----------------------------
# Main
# ----------------------------
def main():
    # ---- config ----
    dataset_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"
    ckpt_path = "checkpoints_mse/epoch3_full.pt"

    sample_idx = 982
    T = 8
    lead = 1

    # Visualisation choices
    t_view = 7  # show tp at time index 7 (last input step)
    contour_q = 0.95  # top-5% contour
    top_k_vars = 5

    # Attribution choices
    method = "ig"  # "ig" or "gradxinput"
    steps = 30     # used only if method="ig"
    baseline_mode = "zeros"  # "zeros" or "mean_over_space_time"

    # target: explain high predicted precip regions
    region_quantile = 0.90  # region = top 10% predicted pixels

    # ---- device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- data ----
    dataset = ERA5Dataset(dataset_path, T=T, lead=lead)
    input_vars = list(dataset.X.coords["channel"].values)
    C_in = len(input_vars)
    print("C_in:", C_in)

    # ---- model ----
    model = PrecipConvLSTM(
        input_channels=C_in,
        hidden_channels=[32, 64],
        kernel_size=3,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint epoch={ckpt.get('epoch', 'unknown')} from {ckpt_path}")

    # ---- sample ----
    X, y, *_ = dataset[sample_idx]
    X = X.unsqueeze(0).to(device).float()  # (1,T,C,H,W)
    y = y.unsqueeze(0).to(device).float()
    if y.dim() == 3:
        y = y.unsqueeze(1)  # (1,1,H,W)

    # ---- predict ----
    with torch.no_grad():
        y_hat = model(X)
        if y_hat.dim() == 3:
            y_hat = y_hat.unsqueeze(1)

    B, _, H, W = y_hat.shape
    print("y_hat shape:", tuple(y_hat.shape))

    # ---- define region to explain (top-10% predicted precip) ----
    with torch.no_grad():
        pred_map = y_hat[0, 0]  # (H,W)
        thresh = torch.quantile(pred_map.flatten(), region_quantile)
        region = (pred_map >= thresh).float()
        region_mask = region.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # ---- attribution ----
    if method == "ig":
        baseline = make_baseline(X, mode=baseline_mode)
        t0 = time.time()
        attr = integrated_gradients(
            model=model,
            x=X,
            baseline=baseline,
            steps=steps,
            target="region_sum",
            region_mask=region_mask,
        )
        print(f"Computed IG in {time.time()-t0:.2f}s (steps={steps}, baseline={baseline_mode})")
    elif method == "gradxinput":
        t0 = time.time()
        attr = grad_x_input(
            model=model,
            x=X,
            target="region_sum",
            region_mask=region_mask,
        )
        print(f"Computed grad×input in {time.time()-t0:.2f}s")
    else:
        raise ValueError("method must be 'ig' or 'gradxinput'")

    attr_abs = attr.abs()  # (1,T,C,H,W)

    # ---- summaries: variable/time importance ----
    var_importance = attr_abs.sum(dim=(1, 3, 4))[0].detach().cpu().numpy()  # sum over T,H,W -> (C,)
    time_importance = attr_abs.sum(dim=(2, 3, 4))[0].detach().cpu().numpy()  # sum over C,H,W -> (T,)

    # ---- output dir ----
    out_dir = "explainability/ig_outputs_mse_viz"
    os.makedirs(out_dir, exist_ok=True)

    # ---- plots: importance ----
    save_barplot(
        var_importance,
        labels=input_vars,
        out_path=os.path.join(out_dir, f"sample{sample_idx}_{method}_var_importance.png"),
        title=f"{method.upper()} variable importance (sum abs over T,H,W)\nSample {sample_idx}",
        top_k=15,
    )
    save_lineplot(
        time_importance,
        out_path=os.path.join(out_dir, f"sample{sample_idx}_{method}_time_importance.png"),
        title=f"{method.upper()} time importance (sum abs over C,H,W)\nSample {sample_idx}",
        xlabel="t index in input window (0..T-1, past→present)",
        ylabel="Importance (sum abs attribution)",
    )

    # ---- get tp input at t=7 ----
    tp_idx, tp_name = find_precip_channel(input_vars)
    if tp_idx is None:
        raise RuntimeError(
            "No precip channel found in input_vars. "
            "Update find_precip_channel() with the exact name in your dataset."
        )

    tp_in = X[0, t_view, tp_idx].detach().cpu().numpy()  # (H,W)

    # ---- europe mask: safest fallback = finite values of tp ----
    # If your dataset uses NaN outside-Europe, this yields the desired white mask.
    europe_mask = np.isfinite(tp_in)

    # ---- top-k variable visualisations ----
    top_idx = np.argsort(-var_importance)[:top_k_vars]

    # Also produce one global spatial attribution (sum over T,C) if you want
    spatial_all = attr_abs.sum(dim=(1, 2))[0].detach().cpu().numpy()
    plot_tp_attr_contour(
        tp_map=tp_in,
        attr_map=spatial_all,
        europe_mask=europe_mask,
        out_path=os.path.join(out_dir, f"sample{sample_idx}_{method}_ALL_tp_t{t_view}_contour.png"),
        title=f"Sample {sample_idx} | ALL vars (sum over T,C) | method={method}",
        tp_title=f"{tp_name} input (t={t_view})",
        attr_title=f"{method.upper()} abs attribution (sum over T,C)",
        contour_q=contour_q,
    )

    # Per-variable top-k maps (sum over time for each variable)
    for rank, c_idx in enumerate(top_idx, start=1):
        var_name = input_vars[c_idx]

        # Attribution de cette variable (sum over T)
        attr_c = attr_abs[0, :, c_idx].sum(dim=0).detach().cpu().numpy()  # (H,W)

        # Variable à différents temps: t=7,6,5
        var_t7 = X[0, 7, c_idx].detach().cpu().numpy()
        var_t6 = X[0, 6, c_idx].detach().cpu().numpy()
        var_t5 = X[0, 5, c_idx].detach().cpu().numpy()

        plot_tp_var_times_attr_contour(
            tp_map=tp_in,
            var_maps=[var_t7, var_t6, var_t5],
            var_name=var_name,
            attr_map=attr_c,
            europe_mask=europe_mask,
            out_path=os.path.join(out_dir, f"sample{sample_idx}_{method}_top{rank}_{var_name}_tp_t{t_view}.png"),
            title=f"Sample {sample_idx} | Top-{rank} variable: {var_name} | method={method}",
            tp_title=f"{tp_name} input (t={t_view})",
            attr_title=f"{method.upper()} abs attribution for {var_name} (sum over T)",
            var_titles=[f"{var_name} input (t=7)", f"{var_name} input (t=6)", f"{var_name} input (t=5)"],
            contour_q=contour_q,
            var_cmap="cividis",  # or "cividis"
        )


    print("[DONE] Outputs written to:", out_dir)


if __name__ == "__main__":
    main()
