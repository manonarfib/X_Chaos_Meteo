import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from models.utils.ERA5_dataset_from_local import ERA5Dataset
from models.ConvLSTM.convlstm import PrecipConvLSTM
from models.unet.model_without_collapse import WFUNet_with_train


# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

torch.backends.cudnn.benchmark = True

LEAD = 1
T = 8
BATCH_SIZE = 8
MAX_LEAD = 1
WITHOUT_PRECIP = False

TRAIN_DATASET_PATH = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_train.zarr"
VAL_DATASET_PATH = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_validation.zarr"

USE_TRAIN_SUBSAMPLE = True
TRAIN_SUBSAMPLE_YEARS = 2
RANDOM_SEED = 0

CONVLSTM_CKPT = "checkpoints/convlstm/mse/epoch3_full.pt"
UNET_CKPT = "checkpoints/unet/best_mse_true.pt"

THRESHOLDS = [0.1, 1.0, 5.0, 10.0]
EPS = 1e-8

# Sauvegarde optionnelle
SAVE_ALPHA_MAP = True
ALPHA_OUT_PATH = "models/mixture/alpha_map_val.npy"

# ============================================================
# HELPER
# ============================================================
def build_random_train_sampler(dataset, n_years, seed=42):
    # 4 pas de temps par jour (6h), approx 365 jours/an
    n_samples = min(len(dataset), n_years * 365 * 4)

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n_samples, replace=False)

    return SubsetRandomSampler(indices.tolist())

# ============================================================
# DATA
# ============================================================
train_dataset = ERA5Dataset(
    TRAIN_DATASET_PATH,
    T=T,
    lead=LEAD,
    without_precip=WITHOUT_PRECIP,
    max_lead=MAX_LEAD
)

if USE_TRAIN_SUBSAMPLE:
    train_sampler = build_random_train_sampler(
        train_dataset,
        n_years=TRAIN_SUBSAMPLE_YEARS,
        seed=RANDOM_SEED
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
else:
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

val_dataset = ERA5Dataset(
    VAL_DATASET_PATH,
    T=T,
    lead=LEAD,
    without_precip=WITHOUT_PRECIP,
    max_lead=MAX_LEAD
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

input_vars = list(val_dataset.X.coords["channel"].values)
C_in = len(input_vars)


# ============================================================
# MODEL
# ============================================================
def build_model(model_type: str, c_in: int, device: torch.device, max_lead: int):
    model_type = model_type.lower().strip()

    if model_type == "convlstm":
        return PrecipConvLSTM(
            input_channels=c_in,
            hidden_channels=[32, 64],
            kernel_size=3,
            output_size=max_lead
        ).to(device)

    elif model_type == "unet":
        return WFUNet_with_train(T, 149, 221, c_in, max_lead, 8, 32, 0).to(device)

    else:
        raise ValueError(f"Unknown model_type='{model_type}'")


def load_model(model_type: str, ckpt_path: str, c_in: int, device: torch.device, max_lead: int):
    model = build_model(model_type, c_in, device, max_lead)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {model_type} checkpoint from {ckpt_path} | epoch={ckpt.get('epoch', 'unknown')}")
    return model


# ============================================================
# HELPERS
# ============================================================
def prepare_target(y_batch, max_lead, device):
    if max_lead == 1:
        return y_batch.to(device, non_blocking=True).float()
    else:
        return y_batch[:, -1, :, :].to(device, non_blocking=True).float()


def predict_model(model, x_batch, max_lead):
    y_hat = model(x_batch)

    if max_lead == 1:
        y_hat = y_hat.squeeze(1)   # (B,H,W)
    else:
        y_hat = y_hat.squeeze(1)   # (B,L,H,W) probable
        y_hat = y_hat[:, -1, :, :] # dernier lead

    return torch.clamp(y_hat, min=0.0)


def init_metric_accumulators(thresholds):
    return {
        "mse_sum": 0.0,
        "mae_sum": 0.0,
        "num_pixels": 0,
        "tp_tot": {th: 0 for th in thresholds},
        "fp_tot": {th: 0 for th in thresholds},
        "fn_tot": {th: 0 for th in thresholds},
        "tn_tot": {th: 0 for th in thresholds},
    }


def update_metrics(acc, y_hat, y_true, thresholds):
    diff = y_hat - y_true
    acc["mse_sum"] += (diff * diff).sum().item()
    acc["mae_sum"] += diff.abs().sum().item()
    acc["num_pixels"] += y_true.numel()

    for th in thresholds:
        pred_bin = y_hat >= th
        true_bin = y_true >= th

        acc["tp_tot"][th] += torch.logical_and(pred_bin, true_bin).sum().item()
        acc["fp_tot"][th] += torch.logical_and(pred_bin, ~true_bin).sum().item()
        acc["fn_tot"][th] += torch.logical_and(~pred_bin, true_bin).sum().item()
        acc["tn_tot"][th] += torch.logical_and(~pred_bin, ~true_bin).sum().item()


def finalize_metrics(acc, thresholds):
    mse = acc["mse_sum"] / acc["num_pixels"]
    mae = acc["mae_sum"] / acc["num_pixels"]

    csi = {}
    hss = {}
    pod = {}
    far = {}

    for th in thresholds:
        a = acc["tp_tot"][th]
        b = acc["fp_tot"][th]
        c = acc["fn_tot"][th]
        d = acc["tn_tot"][th]

        csi[th] = a / (a + b + c + EPS)
        hss[th] = 2 * (a * d - b * c) / ((a + c) * (c + d) + (a + b) * (b + d) + EPS)
        pod[th] = a / (a + c + EPS)
        far[th] = b / (a + b + EPS)

    return {
        "mse": mse,
        "mae": mae,
        "csi": csi,
        "hss": hss,
        "pod": pod,
        "far": far,
    }


def print_metrics(name, result):
    print(f"\n=== {name} ===")
    print(f"MSE: {result['mse']:.6f}")
    print(f"MAE: {result['mae']:.6f}")
    for th in THRESHOLDS:
        print(f"CSI @ {th} mm: {result['csi'][th]:.6f}")
    for th in THRESHOLDS:
        print(f"HSS @ {th} mm: {result['hss'][th]:.6f}")
    for th in THRESHOLDS:
        print(f"POD @ {th} mm: {result['pod'][th]:.6f}")
    for th in THRESHOLDS:
        print(f"FAR @ {th} mm: {result['far'][th]:.6f}")


# ============================================================
# PASS 1: TRAIN -> BUILD SPATIAL ALPHA MAP
# ============================================================
@torch.inference_mode()
def build_alpha_map_from_train(convlstm_model, unet_model, dataloader, device, max_lead):
    err_map_convlstm = None
    err_map_unet = None
    n_samples = 0

    for batch_idx, (X_batch, y_batch, *_) in enumerate(dataloader):
        if batch_idx % 20 == 0:
            print(f"TRAIN-SUBSET batch {batch_idx}/{len(dataloader)}")

        X_batch = X_batch.to(device, non_blocking=True).float()
        y_true = prepare_target(y_batch, max_lead, device)  # (B,H,W)

        y_hat_convlstm = predict_model(convlstm_model, X_batch, max_lead)
        y_hat_unet = predict_model(unet_model, X_batch, max_lead)

        sq_err_convlstm = (y_hat_convlstm - y_true) ** 2   # (B,H,W)
        sq_err_unet = (y_hat_unet - y_true) ** 2           # (B,H,W)

        batch_err_convlstm = sq_err_convlstm.sum(dim=0)    # (H,W)
        batch_err_unet = sq_err_unet.sum(dim=0)            # (H,W)

        if err_map_convlstm is None:
            err_map_convlstm = batch_err_convlstm
            err_map_unet = batch_err_unet
        else:
            err_map_convlstm += batch_err_convlstm
            err_map_unet += batch_err_unet

        n_samples += y_true.shape[0]

    err_map_convlstm = err_map_convlstm / n_samples
    err_map_unet = err_map_unet / n_samples

    alpha_map = err_map_unet / (err_map_convlstm + err_map_unet + EPS)
    alpha_map = torch.clamp(alpha_map, 0.0, 1.0)  # sécurité

    return alpha_map.cpu(), err_map_convlstm.cpu(), err_map_unet.cpu()


# ============================================================
# PASS 2: VAL -> EVALUATE CONVLSTM / UNET / SPATIAL ENSEMBLE
# ============================================================
@torch.inference_mode()
def evaluate_on_val_with_spatial_alpha(convlstm_model, unet_model, dataloader, alpha_map, device, max_lead, thresholds):
    conv_acc = init_metric_accumulators(thresholds)
    unet_acc = init_metric_accumulators(thresholds)
    ens_acc = init_metric_accumulators(thresholds)

    alpha_map = alpha_map.to(device)  # (H,W)

    for batch_idx, (X_batch, y_batch, *_) in enumerate(dataloader):
        if batch_idx % 20 == 0:
            print(f"VAL batch {batch_idx}/{len(dataloader)}")

        X_batch = X_batch.to(device, non_blocking=True).float()
        y_true = prepare_target(y_batch, max_lead, device)  # (B,H,W)

        y_hat_convlstm = predict_model(convlstm_model, X_batch, max_lead)
        y_hat_unet = predict_model(unet_model, X_batch, max_lead)

        # broadcasting alpha_map: (H,W) -> (B,H,W)
        y_hat_ens = alpha_map * y_hat_convlstm + (1.0 - alpha_map) * y_hat_unet
        y_hat_ens = torch.clamp(y_hat_ens, min=0.0)

        update_metrics(conv_acc, y_hat_convlstm, y_true, thresholds)
        update_metrics(unet_acc, y_hat_unet, y_true, thresholds)
        update_metrics(ens_acc, y_hat_ens, y_true, thresholds)

    conv_result = finalize_metrics(conv_acc, thresholds)
    unet_result = finalize_metrics(unet_acc, thresholds)
    ens_result = finalize_metrics(ens_acc, thresholds)

    return conv_result, unet_result, ens_result


# ============================================================
# MAIN
# ============================================================
def main():
    convlstm_model = load_model("convlstm", CONVLSTM_CKPT, C_in, DEVICE, MAX_LEAD)
    unet_model = load_model("unet", UNET_CKPT, C_in, DEVICE, MAX_LEAD)

    print("\nBuilding spatial alpha map on TRAIN-SUBSET (single pass)...")
    alpha_map, err_map_convlstm, err_map_unet = build_alpha_map_from_train(
        convlstm_model, unet_model, train_loader, DEVICE, MAX_LEAD
    )

    print("\nAlpha map stats:")
    print(f"alpha min  = {alpha_map.min().item():.6f}")
    print(f"alpha max  = {alpha_map.max().item():.6f}")
    print(f"alpha mean = {alpha_map.mean().item():.6f}")
    print(f"alpha std  = {alpha_map.std().item():.6f}")

    if SAVE_ALPHA_MAP:
        os.makedirs(os.path.dirname(ALPHA_OUT_PATH), exist_ok=True)
        np.save(ALPHA_OUT_PATH, alpha_map.numpy())
        print(f"Saved alpha map to {ALPHA_OUT_PATH}")

    print("\nEvaluating on VAL with spatial alpha map (single pass)...")
    val_convlstm_result, val_unet_result, val_ensemble_result = evaluate_on_val_with_spatial_alpha(
        convlstm_model,
        unet_model,
        val_loader,
        alpha_map,
        DEVICE,
        MAX_LEAD,
        THRESHOLDS,
    )

    print_metrics("ConvLSTM VAL", val_convlstm_result)
    print_metrics("UNet VAL", val_unet_result)
    print_metrics("Spatial Ensemble VAL", val_ensemble_result)


if __name__ == "__main__":
    main()

# Device: cuda
# Loaded convlstm checkpoint from checkpoints/convlstm/mse/epoch3_full.pt | epoch=3
# Loaded unet checkpoint from checkpoints/unet/best_mse_true.pt | epoch=5

# Building spatial alpha map on VAL (single pass)...
# VAL batch 0/363
# VAL batch 20/363
# VAL batch 40/363
# VAL batch 60/363
# VAL batch 80/363
# VAL batch 100/363
# VAL batch 120/363
# VAL batch 140/363
# VAL batch 160/363
# VAL batch 180/363
# VAL batch 200/363
# VAL batch 220/363
# VAL batch 240/363
# VAL batch 260/363
# VAL batch 280/363
# VAL batch 300/363
# VAL batch 320/363
# VAL batch 340/363
# VAL batch 360/363

# Alpha map stats:
# alpha min  = 0.361620
# alpha max  = 0.636747
# alpha mean = 0.477814
# alpha std  = 0.032784
# Saved alpha map to models/mixture/alpha_map_val.npy

# Evaluating on TEST with spatial alpha map (single pass)...
# TEST batch 0/363
# TEST batch 20/363
# TEST batch 40/363
# TEST batch 60/363
# TEST batch 80/363
# TEST batch 100/363
# TEST batch 120/363
# TEST batch 140/363
# TEST batch 160/363
# TEST batch 180/363
# TEST batch 200/363
# TEST batch 220/363
# TEST batch 240/363
# TEST batch 260/363
# TEST batch 280/363
# TEST batch 300/363
# TEST batch 320/363
# TEST batch 340/363
# TEST batch 360/363

# === ConvLSTM TEST ===
# MSE: 0.756259
# MAE: 0.334600
# CSI @ 0.1 mm: 0.681158
# CSI @ 1.0 mm: 0.587251
# CSI @ 5.0 mm: 0.398580
# CSI @ 10.0 mm: 0.264531
# HSS @ 0.1 mm: 0.651674
# HSS @ 1.0 mm: 0.693391
# HSS @ 5.0 mm: 0.561385
# HSS @ 10.0 mm: 0.416669
# POD @ 0.1 mm: 0.909907
# POD @ 1.0 mm: 0.758818
# POD @ 5.0 mm: 0.485650
# POD @ 10.0 mm: 0.310681
# FAR @ 0.1 mm: 0.269580
# FAR @ 1.0 mm: 0.277983
# FAR @ 5.0 mm: 0.310257
# FAR @ 10.0 mm: 0.359609

# === UNet TEST ===
# MSE: 0.702082
# MAE: 0.335946
# CSI @ 0.1 mm: 0.626050
# CSI @ 1.0 mm: 0.592064
# CSI @ 5.0 mm: 0.435559
# CSI @ 10.0 mm: 0.296255
# HSS @ 0.1 mm: 0.564953
# HSS @ 1.0 mm: 0.697857
# HSS @ 5.0 mm: 0.598771
# HSS @ 10.0 mm: 0.455452
# POD @ 0.1 mm: 0.903834
# POD @ 1.0 mm: 0.763099
# POD @ 5.0 mm: 0.525646
# POD @ 10.0 mm: 0.343505
# FAR @ 0.1 mm: 0.329272
# FAR @ 1.0 mm: 0.274607
# FAR @ 5.0 mm: 0.282373
# FAR @ 10.0 mm: 0.317078

# === Spatial Ensemble TEST ===
# MSE: 0.659468
# MAE: 0.318066
# CSI @ 0.1 mm: 0.665020
# CSI @ 1.0 mm: 0.611666
# CSI @ 5.0 mm: 0.435853
# CSI @ 10.0 mm: 0.284905
# HSS @ 0.1 mm: 0.622936
# HSS @ 1.0 mm: 0.715770
# HSS @ 5.0 mm: 0.599339
# HSS @ 10.0 mm: 0.441910
# POD @ 0.1 mm: 0.924563
# POD @ 1.0 mm: 0.780442
# POD @ 5.0 mm: 0.513223
# POD @ 10.0 mm: 0.320970
# FAR @ 0.1 mm: 0.296826
# FAR @ 1.0 mm: 0.261205
# FAR @ 5.0 mm: 0.256991
# FAR @ 10.0 mm: 0.282838

# Device: cuda
# Loaded convlstm checkpoint from checkpoints/convlstm/mse/epoch3_full.pt | epoch=3
# Loaded unet checkpoint from checkpoints/unet/best_mse_true.pt | epoch=5

# Building spatial alpha map on TRAIN-SUBSET (single pass)...
# TRAIN-SUBSET batch 0/365
# TRAIN-SUBSET batch 20/365
# TRAIN-SUBSET batch 40/365
# TRAIN-SUBSET batch 60/365
# TRAIN-SUBSET batch 80/365
# TRAIN-SUBSET batch 100/365
# TRAIN-SUBSET batch 120/365
# TRAIN-SUBSET batch 140/365
# TRAIN-SUBSET batch 160/365
# TRAIN-SUBSET batch 180/365
# TRAIN-SUBSET batch 200/365
# TRAIN-SUBSET batch 220/365
# TRAIN-SUBSET batch 240/365
# TRAIN-SUBSET batch 260/365
# TRAIN-SUBSET batch 280/365
# TRAIN-SUBSET batch 300/365
# TRAIN-SUBSET batch 320/365
# TRAIN-SUBSET batch 340/365
# TRAIN-SUBSET batch 360/365

# Alpha map stats:
# alpha min  = 0.308489
# alpha max  = 0.632317
# alpha mean = 0.474235
# alpha std  = 0.035153
# Saved alpha map to models/mixture/alpha_map_val.npy

# Evaluating on VAL with spatial alpha map (single pass)...
# VAL batch 0/363
# VAL batch 20/363
# VAL batch 40/363
# VAL batch 60/363
# VAL batch 80/363
# VAL batch 100/363
# VAL batch 120/363
# VAL batch 140/363
# VAL batch 160/363
# VAL batch 180/363
# VAL batch 200/363
# VAL batch 220/363
# VAL batch 240/363
# VAL batch 260/363
# VAL batch 280/363
# VAL batch 300/363
# VAL batch 320/363
# VAL batch 340/363
# VAL batch 360/363

# === ConvLSTM VAL ===
# MSE: 0.722578
# MAE: 0.328448
# CSI @ 0.1 mm: 0.676467
# CSI @ 1.0 mm: 0.586082
# CSI @ 5.0 mm: 0.399532
# CSI @ 10.0 mm: 0.270102
# HSS @ 0.1 mm: 0.649081
# HSS @ 1.0 mm: 0.693339
# HSS @ 5.0 mm: 0.562628
# HSS @ 10.0 mm: 0.423695
# POD @ 0.1 mm: 0.911816
# POD @ 1.0 mm: 0.758932
# POD @ 5.0 mm: 0.486973
# POD @ 10.0 mm: 0.316669
# FAR @ 0.1 mm: 0.276178
# FAR @ 1.0 mm: 0.279853
# FAR @ 5.0 mm: 0.310071
# FAR @ 10.0 mm: 0.352513

# === UNet VAL ===
# MSE: 0.679499
# MAE: 0.331537
# CSI @ 0.1 mm: 0.620745
# CSI @ 1.0 mm: 0.589895
# CSI @ 5.0 mm: 0.428980
# CSI @ 10.0 mm: 0.297320
# HSS @ 0.1 mm: 0.562382
# HSS @ 1.0 mm: 0.696929
# HSS @ 5.0 mm: 0.592519
# HSS @ 10.0 mm: 0.456787
# POD @ 0.1 mm: 0.904002
# POD @ 1.0 mm: 0.761489
# POD @ 5.0 mm: 0.518050
# POD @ 10.0 mm: 0.345372
# FAR @ 0.1 mm: 0.335449
# FAR @ 1.0 mm: 0.276411
# FAR @ 5.0 mm: 0.286120
# FAR @ 10.0 mm: 0.318780

# === Spatial Ensemble VAL ===
# MSE: 0.633734
# MAE: 0.313300
# CSI @ 0.1 mm: 0.659578
# CSI @ 1.0 mm: 0.609900
# CSI @ 5.0 mm: 0.432358
# CSI @ 10.0 mm: 0.288066
# HSS @ 0.1 mm: 0.619657
# HSS @ 1.0 mm: 0.715187
# HSS @ 5.0 mm: 0.596133
# HSS @ 10.0 mm: 0.445799
# POD @ 0.1 mm: 0.925619
# POD @ 1.0 mm: 0.779306
# POD @ 5.0 mm: 0.509446
# POD @ 10.0 mm: 0.324780
# FAR @ 0.1 mm: 0.303507
# FAR @ 1.0 mm: 0.262766
# FAR @ 5.0 mm: 0.259248
# FAR @ 10.0 mm: 0.281828