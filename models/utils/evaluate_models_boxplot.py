import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from torch.utils.data import DataLoader, Subset

from models.utils.ERA5_dataset_from_local import ERA5Dataset
from models.ConvLSTM.convlstm import PrecipConvLSTM
from models.unet.model_without_collapse import WFUNet_with_train


# ============================================================
# USER CONFIG
# ============================================================
CONVLSTM_CKPT = "checkpoints/convlstm/mse/epoch3_full.pt"
UNET_CKPT     = "checkpoints/unet/best_mse_true.pt"

LEAD = 1
SAMPLE_IDX = 13
DATASET_PATH = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"

T = 8
MAX_LEAD = 1
CLIP_NEG_PRED = True
# ============================================================


def save_boxplot_multi(y_true: np.ndarray, preds_dict: dict, out_path: str, title: str = ""):

    labels = list(preds_dict.keys())

    yt = y_true.flatten()
    preds_flat = [preds_dict[k].flatten() for k in labels]

    mask = np.isfinite(yt)
    for pf in preds_flat:
        mask = mask & np.isfinite(pf)

    data = [yt[mask]] + [pf[mask] for pf in preds_flat]
    tick_labels = ["Truth"] + labels

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.boxplot(
        data,
        tick_labels=tick_labels,
        showfliers=True,
        patch_artist=True
    )

    ax.set_ylabel("Precipitation value (mm/6h)")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

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



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    DATASET_PATH = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_validation.zarr"
    BATCH_SIZE = 8

    dataset = ERA5Dataset(
        DATASET_PATH,
        T=T,
        lead=LEAD,
        max_lead=MAX_LEAD
    )

    loader = DataLoader(
        Subset(dataset, range(100)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    input_vars = list(dataset.X.coords["channel"].values)
    C_in = len(input_vars)

    # =========================================================
    # LOAD MODELS
    # =========================================================
    conv_model = build_model("convlstm", C_in, device, MAX_LEAD)
    ckpt_conv = torch.load(CONVLSTM_CKPT, map_location=device)
    conv_model.load_state_dict(ckpt_conv["model_state_dict"])
    conv_model.eval()

    unet_model = build_model("unet", C_in, device, MAX_LEAD)
    ckpt_unet = torch.load(UNET_CKPT, map_location=device)
    unet_model.load_state_dict(ckpt_unet["model_state_dict"])
    unet_model.eval()

    # =========================================================
    # STORAGE
    # =========================================================
    all_truth = []
    all_conv = []
    all_unet = []

    # =========================================================
    # INFERENCE LOOP
    # =========================================================
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch, *_) in enumerate(loader):

            print(f"Batch {batch_idx+1}/{len(loader)}")

            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            # Predictions
            y_hat_conv = _predict_model(
                conv_model,
                X_batch,
                max_lead=MAX_LEAD,
                clip_neg=CLIP_NEG_PRED
            )

            y_hat_unet = _predict_model(
                unet_model,
                X_batch,
                max_lead=MAX_LEAD,
                clip_neg=CLIP_NEG_PRED
            )

            # CPU numpy
            y_np = y_batch.cpu().numpy()
            conv_np = y_hat_conv.cpu().numpy()
            unet_np = y_hat_unet.cpu().numpy()

            # Flatten batch
            all_truth.append(y_np.flatten())
            all_conv.append(conv_np.flatten())
            all_unet.append(unet_np.flatten())

    # =========================================================
    # CONCAT GLOBAL
    # =========================================================
    all_truth = np.concatenate(all_truth)
    all_conv = np.concatenate(all_conv)
    all_unet = np.concatenate(all_unet)

    preds = {
        "ConvLSTM": all_conv,
        "UNet": all_unet,
    }

    # =========================================================
    # BOXPLOT
    # =========================================================
    out_dir = "inference/global_validation_boxplots"
    os.makedirs(out_dir, exist_ok=True)

    save_boxplot_multi(
        y_true=all_truth,
        preds_dict=preds,
        out_path=f"{out_dir}/boxplot_validation_global.png",
        title="Test set - Truth vs ConvLSTM vs UNet"
    )

if __name__ == "__main__":
    main()