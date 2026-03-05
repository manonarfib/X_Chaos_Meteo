import os
import numpy as np
import torch
import torch.nn as nn
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
MODEL_TYPE = "unet"  # "convlstm" or "unet"

# CKPT_PATHS = {
#     "MSE": "checkpoints/convlstm/mse/epoch3_full.pt",
#     "W_MSE": "checkpoints/convlstm/w_mse/epoch3_full.pt",
#     "MSE+W_DICE": "checkpoints/convlstm/mse_and_w_dice/epoch3_full.pt",
#     "ADV_TORRENTIAL": "checkpoints/convlstm/advanced_torrential/epoch3_full.pt",
# }
CKPT_PATHS = {
    "MSE": "checkpoints/run_144949/best_checkpoint_epoch5_batch_idx6479.pt",
    "W_MSE": "checkpoints/run_148318/best_checkpoint_epoch4_batch_idx1439.pt",
}

LEAD = 1  # lead in 6h steps -> prediction at t_lead = LEAD*6 hours
SAMPLE_IDX = 250
DATASET_PATH = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"

T = 8
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


def save_maps_europe_multi(
    y_true: np.ndarray,
    preds_dict: dict,   # {label: pred_2d}
    out_path: str,
    title_prefix: str = "",
    region=(-12.5, 42.5, 35, 72),
    tp_cmap_name: str = "cividis",
):
    """
    Multi panels with Cartopy background:
      - Truth
      - Pred for each label in preds_dict (order preserved)
    White outside Europe.
    Shared color scale across all panels.
    """
    labels = list(preds_dict.keys())
    preds = [preds_dict[k] for k in labels]

    # Mask: white outside Europe (anything finite among all panels)
    europe_mask = np.isfinite(y_true)
    for p in preds:
        europe_mask = europe_mask | np.isfinite(p)

    yt = _apply_mask(y_true, europe_mask)
    masked_preds = [_apply_mask(p, europe_mask) for p in preds]

    # Shared scale
    all_arrays = [yt] + masked_preds
    vmin = float(np.nanmin([np.nanmin(a) for a in all_arrays]))
    vmax = float(np.nanmax([np.nanmax(a) for a in all_arrays]))

    tp_cmap = _cmap_with_white_bad(tp_cmap_name)

    lon_min, lon_max, lat_min, lat_max = region
    extent = [lon_min, lon_max, lat_min, lat_max]
    proj = ccrs.PlateCarree()

    ncols = 3
    n_panels = 1 + len(labels)  # truth + predictions
    nrows = int(np.ceil(n_panels / ncols))

    fig = plt.figure(figsize=(5.6 * ncols, 4.6 * nrows), constrained_layout=True)

    def _setup_ax(ax):
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.4)
        gl.right_labels = False
        gl.top_labels = False

    # Panel 0: truth
    ax0 = fig.add_subplot(nrows, ncols, 1, projection=proj)
    _setup_ax(ax0)
    im0 = ax0.imshow(yt, cmap=tp_cmap, vmin=vmin, vmax=vmax, origin="upper", extent=extent, transform=proj)
    ax0.set_title(f"{title_prefix}Truth tp_6h")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)

    # Panels: preds
    for i, (lab, mp) in enumerate(zip(labels, masked_preds), start=2):
        ax = fig.add_subplot(nrows, ncols, i, projection=proj)
        _setup_ax(ax)
        im = ax.imshow(mp, cmap=tp_cmap, vmin=vmin, vmax=vmax, origin="upper", extent=extent, transform=proj)
        ax.set_title(f"{title_prefix}Pred ({lab}) tp_6h")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[FIG] Saved maps: {out_path}")


def save_boxplot_multi(y_true: np.ndarray, preds_dict: dict, out_path: str, title: str = ""):
    labels = list(preds_dict.keys())

    yt = y_true.flatten()
    preds_flat = [preds_dict[k].flatten() for k in labels]

    # finite mask across all
    mask = np.isfinite(yt)
    for pf in preds_flat:
        mask = mask & np.isfinite(pf)

    data = [yt[mask]] + [pf[mask] for pf in preds_flat]
    tick_labels = ["Truth"] + labels

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.boxplot(
        data,
        tick_labels=tick_labels,
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
        model = WFUNet_with_train(T, 149, 221, C_in, 1, 8, 32, 0).to(device)
        return model
    else:
        raise ValueError(f"Unknown MODEL_TYPE='{model_type}'. Use 'convlstm' or 'unet'.")


def _load_model_and_predict(model_type, C_in, device, ckpt_path, X, clip_neg=True):
    model = build_model(model_type, C_in, device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        y_hat = model(X)
        if y_hat.ndim == 4 and y_hat.shape[1] == 1:
            y_hat = y_hat.squeeze(1)
        if clip_neg:
            y_hat = torch.clamp(y_hat, min=0.0)
    return y_hat


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

    # Predict for each checkpoint
    preds = {}
    for label, ckpt_path in CKPT_PATHS.items():
        print(f"[LOAD] {label}: {ckpt_path}")
        y_hat = _load_model_and_predict(
            MODEL_TYPE, C_in, device, ckpt_path, X, clip_neg=CLIP_NEG_PRED
        )
        preds[label] = y_hat[0].detach().cpu().numpy()

        # Optional: compute plain MSE vs truth for logging
        mse_val = nn.MSELoss()(y_hat, y).item()
        print(f"  -> MSE vs truth: {mse_val:.6f}")

    y_true = y[0].detach().cpu().numpy()

    # Output paths (new folder)
    model_tag = MODEL_TYPE.lower().strip()
    out_dir = f"inference/compare_predict_maps_outputs/model_{model_tag}/sample{SAMPLE_IDX}"
    maps_path = f"{out_dir}/maps_truth_vs_" + "_vs_".join([k.lower() for k in CKPT_PATHS.keys()]) + ".png"
    box_path  = f"{out_dir}/boxplot_truth_vs_" + "_vs_".join([k.lower() for k in CKPT_PATHS.keys()]) + ".png"

    save_maps_europe_multi(
        y_true=y_true,
        preds_dict=preds,
        out_path=maps_path,
        title_prefix=f"Test sample {SAMPLE_IDX} ({t_lead}h) - ",
        tp_cmap_name="Blues",
    )

    save_boxplot_multi(
        y_true=y_true,
        preds_dict=preds,
        out_path=box_path,
        title=f"Sample {SAMPLE_IDX} ({t_lead}h) – Truth vs " + " vs ".join(CKPT_PATHS.keys())
    )


if __name__ == "__main__":
    main()
