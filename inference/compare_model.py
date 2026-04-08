import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from models.utils.ERA5_dataset_from_local import ERA5Dataset
from models.ConvLSTM.convlstm import PrecipConvLSTM
from models.unet.model_without_collapse import WFUNet_with_train


# ============================================================
# USER CONFIG
# ============================================================
CONVLSTM_CKPT = "checkpoints/convlstm/mse/epoch3_full.pt"
UNET_CKPT     = "checkpoints/unet/best_mse_true.pt"

# Best threshold-mixture rule found on TRAIN
TAU = 0.5
ALPHA_LOW = 1.0
ALPHA_HIGH = 0.4

LEAD = 1
SAMPLE_IDX = 250
DATASET_PATH = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"

T = 8
MAX_LEAD = 1
CLIP_NEG_PRED = True
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
    labels = list(preds_dict.keys())
    preds = [preds_dict[k] for k in labels]

    europe_mask = np.isfinite(y_true)
    for p in preds:
        europe_mask = europe_mask | np.isfinite(p)

    yt = _apply_mask(y_true, europe_mask)
    masked_preds = [_apply_mask(p, europe_mask) for p in preds]

    all_arrays = [yt] + masked_preds
    vmin = float(np.nanmin([np.nanmin(a) for a in all_arrays]))
    vmax = float(np.nanmax([np.nanmax(a) for a in all_arrays]))

    tp_cmap = _cmap_with_white_bad(tp_cmap_name)

    lon_min, lon_max, lat_min, lat_max = region
    extent = [lon_min, lon_max, lat_min, lat_max]
    proj = ccrs.PlateCarree()

    ncols = 2
    n_panels = 1 + len(labels)  # truth + predictions
    nrows = int(np.ceil(n_panels / ncols))

    fig = plt.figure(figsize=(6.0 * ncols, 4.8 * nrows), constrained_layout=True)

    def _setup_ax(ax):
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.4)
        gl.right_labels = False
        gl.top_labels = False

    # Truth
    ax0 = fig.add_subplot(nrows, ncols, 1, projection=proj)
    _setup_ax(ax0)
    im0 = ax0.imshow(
        yt, cmap=tp_cmap, vmin=vmin, vmax=vmax,
        origin="upper", extent=extent, transform=proj
    )
    ax0.set_title(f"{title_prefix}Truth tp_6h")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)

    # Predictions
    for i, (lab, mp) in enumerate(zip(labels, masked_preds), start=2):
        ax = fig.add_subplot(nrows, ncols, i, projection=proj)
        _setup_ax(ax)
        im = ax.imshow(
            mp, cmap=tp_cmap, vmin=vmin, vmax=vmax,
            origin="upper", extent=extent, transform=proj
        )
        ax.set_title(f"{title_prefix}{lab} tp_6h")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[FIG] Saved maps: {out_path}")


def save_boxplot_multi(y_true: np.ndarray, preds_dict: dict, out_path: str, title: str = ""):
    labels = list(preds_dict.keys())

    yt = y_true.flatten()
    preds_flat = [preds_dict[k].flatten() for k in labels]

    mask = np.isfinite(yt)
    for pf in preds_flat:
        mask = mask & np.isfinite(pf)

    data = [yt[mask]] + [pf[mask] for pf in preds_flat]
    tick_labels = ["Truth"] + labels

    fig, ax = plt.subplots(figsize=(9, 7))
    bp = ax.boxplot(
        data,
        tick_labels=tick_labels,
        showfliers=True,
        patch_artist=True
    )

    ax.set_ylabel("Precipitation value (mm/6h)")
    ax.set_title(title)
    # ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[FIG] Saved boxplot: {out_path}")


def build_model(model_type: str, C_in: int, device: torch.device, max_lead: int = 1) -> torch.nn.Module:
    model_type = model_type.lower().strip()

    if model_type == "convlstm":
        model = PrecipConvLSTM(
            input_channels=C_in,
            hidden_channels=[32, 64],
            kernel_size=3,
            output_size=max_lead
        ).to(device)
        return model

    elif model_type == "unet":
        model = WFUNet_with_train(T, 149, 221, C_in, max_lead, 8, 32, 0).to(device)
        return model

    else:
        raise ValueError(f"Unknown model_type='{model_type}'")


def _predict_model(model, X, max_lead=1, clip_neg=True):
    with torch.no_grad():
        y_hat = model(X)

        if max_lead == 1:
            # expected shape (B,1,H,W) -> (B,H,W)
            if y_hat.ndim == 4 and y_hat.shape[1] == 1:
                y_hat = y_hat.squeeze(1)
        else:
            # if multi-lead, adapt as needed
            y_hat = y_hat.squeeze(1)
            y_hat = y_hat[:, -1, :, :]

        if clip_neg:
            y_hat = torch.clamp(y_hat, min=0.0)

    return y_hat


def load_model_and_predict(model_type, ckpt_path, C_in, device, X, max_lead=1, clip_neg=True):
    model = build_model(model_type, C_in, device, max_lead=max_lead)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[LOAD] {model_type} from {ckpt_path}")
    y_hat = _predict_model(model, X, max_lead=max_lead, clip_neg=clip_neg)
    return y_hat


def build_threshold_mixture(pred_conv, pred_unet, tau, alpha_low, alpha_high, clip_neg=True):
    """
    pred_conv, pred_unet: tensors (B,H,W)
    """
    mean_pred = 0.5 * (pred_conv + pred_unet)

    alpha_map = torch.where(
        mean_pred < tau,
        torch.full_like(mean_pred, float(alpha_low)),
        torch.full_like(mean_pred, float(alpha_high)),
    )

    pred_mix = alpha_map * pred_conv + (1.0 - alpha_map) * pred_unet

    if clip_neg:
        pred_mix = torch.clamp(pred_mix, min=0.0)

    return pred_mix


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

    # Individual predictions
    y_hat_conv = load_model_and_predict(
        model_type="convlstm",
        ckpt_path=CONVLSTM_CKPT,
        C_in=C_in,
        device=device,
        X=X,
        max_lead=MAX_LEAD,
        clip_neg=CLIP_NEG_PRED
    )

    y_hat_unet = load_model_and_predict(
        model_type="unet",
        ckpt_path=UNET_CKPT,
        C_in=C_in,
        device=device,
        X=X,
        max_lead=MAX_LEAD,
        clip_neg=CLIP_NEG_PRED
    )

    # Mixture prediction
    y_hat_mix = build_threshold_mixture(
        pred_conv=y_hat_conv,
        pred_unet=y_hat_unet,
        tau=TAU,
        alpha_low=ALPHA_LOW,
        alpha_high=ALPHA_HIGH,
        clip_neg=CLIP_NEG_PRED
    )

    # Logging metrics
    mse_loss = nn.MSELoss()
    print(f"ConvLSTM MSE vs truth: {mse_loss(y_hat_conv, y).item():.6f}")
    print(f"UNet     MSE vs truth: {mse_loss(y_hat_unet, y).item():.6f}")
    print(f"Mixture  MSE vs truth: {mse_loss(y_hat_mix, y).item():.6f}")

    y_true = y[0].detach().cpu().numpy()
    pred_conv_np = y_hat_conv[0].detach().cpu().numpy()
    pred_unet_np = y_hat_unet[0].detach().cpu().numpy()
    pred_mix_np = y_hat_mix[0].detach().cpu().numpy()

    preds = {
        "ConvLSTM": pred_conv_np,
        "UNet": pred_unet_np,
        f"Mixture(tau={TAU}, aL={ALPHA_LOW}, aH={ALPHA_HIGH})": pred_mix_np,
    }

    out_dir = f"inference/compare_3models/sample{SAMPLE_IDX}"

    maps_path = f"{out_dir}/maps_truth_vs_convlstm_vs_unet_vs_mixture.png"
    box_path  = f"{out_dir}/boxplot_truth_vs_convlstm_vs_unet_vs_mixture.png"

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
        title=(
            f"Sample {SAMPLE_IDX} ({t_lead}h) – "
            f"Truth vs ConvLSTM vs UNet vs Mixture "
            f"(tau={TAU}, alpha_low={ALPHA_LOW}, alpha_high={ALPHA_HIGH})"
        )
    )


if __name__ == "__main__":
    main()