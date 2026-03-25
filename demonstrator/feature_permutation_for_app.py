import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from models.utils.ERA5_dataset_from_local import ERA5Dataset
from models.utils.evaluate_model import build_model
from explainability.features_permutation.features_permutation import permutation_importance_batch
from explainability.integrated_gradients.integrated_gradients_over_multi_samples import integrated_gradients_with_progress, make_baseline, lead_to_str, find_precip_channel, rel_time_label, _apply_mask, _cmap_with_white_bad
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import streamlit as st

# MODEL_TYPE = "unet" 
# LEAD=1
# T=8
# MAX_LEAD=1
# WITHOUT_PRECIP=False
# BATCH_SIZE=16
# DATASET_PATH = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"
# CKPT_PATH = 

def barplot_mean_std(mean_vals, std_vals, labels, out_path, title="", top_k=15): 
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
    return fig

def lineplot_mean_std(mean_vals, std_vals, out_path,
                           title="",
                           xlabel="Temps",
                           ylabel="Importance",
                           xtick_labels=None):

    mean_vals = np.asarray(mean_vals)
    std_vals = np.asarray(std_vals)
    x = np.arange(len(mean_vals))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, mean_vals, marker="o", color='mediumseagreen')
    ax.fill_between(x,
                    mean_vals - std_vals,
                    mean_vals + std_vals,
                    alpha=0.25,
                    color='mediumseagreen')
    if xtick_labels is not None:
        assert len(xtick_labels) == len(x), "xtick_labels doit avoir la même longueur que mean_vals"
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    return fig

def feature_permutation_for_one_sample(MODEL_TYPE, CKPT_PATH, DATASET_PATH, LEAD, T, MAX_LEAD, IDX, WITHOUT_PRECIP=False, BATCH_SIZE=16, progress_bar = None, status_text = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = ERA5Dataset(DATASET_PATH, T=T, lead=LEAD, without_precip=WITHOUT_PRECIP, max_lead=MAX_LEAD)
    input_vars = list(test_dataset.X.coords["channel"].values)
    C_in = len(input_vars)

    model = build_model(MODEL_TYPE, C_in, T, device, max_lead=MAX_LEAD)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metric = mean_squared_error

    C = C_in

    all_importances = []   # (N_images, T*C)

    X,y,_ = test_dataset[IDX]
    _, _, H, W = X.shape
    X_flat = X.reshape(T*C, H*W).T
    y_flat = y.reshape(-1)

    imp, base = permutation_importance_batch(
        model, X_flat, y_flat, metric,
        T=T, C=C, H=H, W=W,
        batch_size_features=16,
        n_repeats=5,
        progress_bar=progress_bar,
        status_text=status_text
    )

    all_importances.append(imp)

    all_importances = np.stack(all_importances)   # shape = (N, T*C)

    N = all_importances.shape[0]
    imp_tc = imp.reshape(T, C)
    imp_var = imp_tc.mean(axis=0)

    mean_var = imp_var
    std_var = np.zeros_like(imp_var)

    labels_var = input_vars

    barplot = barplot_mean_std(
        mean_var,
        std_var,
        labels_var,
        f"explainability/features_permutation/figures/importance_per_variable_{MODEL_TYPE}_new.png",
        title="Permutation importance — aggregated per variable",
        top_k=20
    )

    imp_time = imp_tc.mean(axis=1)   # (N, T)

    mean_time = imp_time.mean(axis=0)
    std_time  = imp_time.std(axis=0)
    time_labels = ["t-42h", "t-36h", "t-30h", "t-24h", "t-18h", "t-12h", "t-6h", "t"]

    lineplot = lineplot_mean_std(
        mean_time,
        std_time,
        f"explainability/features_permutation/figures/importance_per_time_{MODEL_TYPE}_new.png",
        title="Permutation importance — aggregated per timestep",
        xlabel="Time",
        ylabel="ΔMSE",
        xtick_labels=time_labels
    )

    return barplot, lineplot


def ig_barplot(values, labels, title="", top_k=15):
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
    return fig


def ig_lineplot(values, title="", xlabel="t index (0..T-1)", ylabel="Importance (sum abs attribution)"):
    values = np.asarray(values)
    T = len(values)
    t_ref = T - 1 
    hours = -6 * (t_ref - np.arange(T))  # [-42, -36, ..., -6, 0] si T=8

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
    return fig


def plot_tp_attr_contour(
    tp_map: np.ndarray,
    attr_map: np.ndarray,
    europe_mask: np.ndarray,
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
    return fig


def plot_tp_var_times_attr_contour(
    tp_map: np.ndarray,
    var_maps: list,           # list of 2D arrays [t=7, t=6, t=5]
    var_name: str,
    attr_map: np.ndarray,
    europe_mask: np.ndarray,
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
    return fig

@st.cache_resource
def integrated_gradients_for_one_sample(MODEL_TYPE, CKPT_PATH, DATASET_PATH, LEAD, T, MAX_LEAD, IDX, WITHOUT_PRECIP=False, BATCH_SIZE=16, _progress_bar = None, _status_text=None):

    METHOD = "ig"
    LOSS_NAME= "MSE"
    model_tag = MODEL_TYPE
    lead_str = lead_to_str(LEAD)
    REGION_QUANTILE = 0.9
    BASELINE_MODE = "zeros"
    IG_STEPS = 30
    T_VIEW = 7
    CONTOUR_Q = 0.95
    TOP_K_VARS = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ERA5Dataset(DATASET_PATH, T=T, lead=LEAD, without_precip=WITHOUT_PRECIP, max_lead=MAX_LEAD)
    input_vars = list(dataset.X.coords["channel"].values)
    C_in = len(input_vars)

    model = build_model(MODEL_TYPE, C_in, T, device, max_lead=MAX_LEAD)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- sample ----
    sample_idx = IDX
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

    baseline = make_baseline(X, mode=BASELINE_MODE)
    attr = integrated_gradients_with_progress(
        model=model,
        x=X,
        baseline=baseline,
        steps=IG_STEPS,
        target="region_sum",
        region_mask=region_mask,
        progress_bar=_progress_bar,
        status_text=_status_text
    )
 
    attr_abs = attr.abs()

    # ---- summaries ----
    var_importance = attr_abs.sum(dim=(1, 3, 4))[0].detach().cpu().numpy()
    time_importance = attr_abs.sum(dim=(2, 3, 4))[0].detach().cpu().numpy()

    # ---- plots: importance ----
    barplot = ig_barplot(
        var_importance,
        labels=input_vars,
        title=f"{METHOD.upper()} variable importance (sum abs over T,H,W)\nSample {sample_idx} | model={model_tag} loss={LOSS_NAME} lead={lead_str}",
        top_k=15,
    )

    lineplot = ig_lineplot(
        time_importance,
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
        title=f"Sample {sample_idx} | ALL vars (sum over T,C) | method={METHOD} | model={model_tag} loss={LOSS_NAME}",
        tp_title=f"{tp_name} input (t={T_VIEW})",
        attr_title=f"{METHOD.upper()} abs attribution (sum over T,C)",
        contour_q=CONTOUR_Q,
    )

    var_figs = []
    var_names = []
    for rank, c_idx in enumerate(top_idx, start=1):
        var_name = input_vars[c_idx]
        var_names.append(var_name)
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

        fig = plot_tp_var_times_attr_contour(
            tp_map=tp_in,
            var_maps=[var_t0, var_t1, var_t2],
            var_name=var_name,
            attr_map=attr_c,
            europe_mask=europe_mask,
            title=f"Sample {sample_idx} | Top-{rank} variable: {var_name} | method={METHOD} | model={model_tag} loss={LOSS_NAME}",
            tp_title=f"{tp_name} input (t)",
            attr_title=f"{METHOD.upper()} abs attribution for {var_name} (sum over T)",
            var_titles=var_titles,
            contour_q=CONTOUR_Q,
            var_cmap="cividis",
        )   
        var_figs.append(fig) 

    return barplot, lineplot, var_figs, var_names