import numpy as np
import torch
from torch.utils.data import DataLoader

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

VAL_DATASET_PATH = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_validation.zarr"
TEST_DATASET_PATH = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"

CONVLSTM_CKPT = "checkpoints/convlstm/mse/epoch3_full.pt"
UNET_CKPT = "checkpoints/unet/best_mse_true.pt"

ALPHAS = np.linspace(0.0, 1.0, 51)

CSI_THRESHOLDS = [0.1, 5.0]


# ============================================================
# DATA
# ============================================================
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

test_dataset = ERA5Dataset(
    TEST_DATASET_PATH,
    T=T,
    lead=LEAD,
    without_precip=WITHOUT_PRECIP,
    max_lead=MAX_LEAD
)
test_loader = DataLoader(
    test_dataset,
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
        y_hat = y_hat.squeeze(1)
    else:
        y_hat = y_hat.squeeze(1)
        y_hat = y_hat[:, -1, :, :]

    return torch.clamp(y_hat, min=0.0)


def init_metric_accumulators(csi_thresholds):
    return {
        "mse_sum": 0.0,
        "mae_sum": 0.0,
        "num_pixels": 0,
        "tp": {th: 0 for th in csi_thresholds},
        "fp": {th: 0 for th in csi_thresholds},
        "fn": {th: 0 for th in csi_thresholds},
    }


def update_metrics(acc, pred, target, csi_thresholds):
    diff = pred - target
    acc["mse_sum"] += (diff * diff).sum().item()
    acc["mae_sum"] += diff.abs().sum().item()
    acc["num_pixels"] += target.numel()

    for th in csi_thresholds:
        pred_bin = pred >= th
        true_bin = target >= th

        acc["tp"][th] += torch.logical_and(pred_bin, true_bin).sum().item()
        acc["fp"][th] += torch.logical_and(pred_bin, ~true_bin).sum().item()
        acc["fn"][th] += torch.logical_and(~pred_bin, true_bin).sum().item()


def finalize_metrics(acc, csi_thresholds):
    eps = 1e-8

    result = {
        "mse": acc["mse_sum"] / acc["num_pixels"],
        "mae": acc["mae_sum"] / acc["num_pixels"],
        "csi": {}
    }

    for th in csi_thresholds:
        tp = acc["tp"][th]
        fp = acc["fp"][th]
        fn = acc["fn"][th]
        result["csi"][th] = tp / (tp + fp + fn + eps)

    return result


# ============================================================
# FAST EVAL: BOTH MODELS IN ONE PASS
# ============================================================
@torch.inference_mode()
def collect_predictions_and_metrics(convlstm_model, unet_model, dataloader, device, max_lead, csi_thresholds):
    conv_preds = []
    unet_preds = []
    targets_all = []

    conv_acc = init_metric_accumulators(csi_thresholds)
    unet_acc = init_metric_accumulators(csi_thresholds)

    for batch_idx, (X_batch, y_batch, *_) in enumerate(dataloader):
        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}")

        X_batch = X_batch.to(device, non_blocking=True).float()
        y_true = prepare_target(y_batch, max_lead, device)

        y_hat_convlstm = predict_model(convlstm_model, X_batch, max_lead)
        y_hat_unet = predict_model(unet_model, X_batch, max_lead)

        update_metrics(conv_acc, y_hat_convlstm, y_true, csi_thresholds)
        update_metrics(unet_acc, y_hat_unet, y_true, csi_thresholds)

        conv_preds.append(y_hat_convlstm.cpu())
        unet_preds.append(y_hat_unet.cpu())
        targets_all.append(y_true.cpu())

    conv_preds = torch.cat(conv_preds, dim=0)
    unet_preds = torch.cat(unet_preds, dim=0)
    targets_all = torch.cat(targets_all, dim=0)

    conv_result = finalize_metrics(conv_acc, csi_thresholds)
    unet_result = finalize_metrics(unet_acc, csi_thresholds)

    return conv_result, unet_result, conv_preds, unet_preds, targets_all


def evaluate_alpha_from_cached_predictions(conv_preds, unet_preds, targets, alpha, csi_thresholds):
    ens_preds = alpha * conv_preds + (1.0 - alpha) * unet_preds
    ens_preds = torch.clamp(ens_preds, min=0.0)

    acc = init_metric_accumulators(csi_thresholds)
    update_metrics(acc, ens_preds, targets, csi_thresholds)

    result = finalize_metrics(acc, csi_thresholds)
    result["alpha"] = float(alpha)
    return result


def print_metrics(name, result):
    print(f"\n=== {name} ===")
    print(f"MSE: {result['mse']:.6f}")
    print(f"MAE: {result['mae']:.6f}")
    for th in CSI_THRESHOLDS:
        print(f"CSI @ {th} mm: {result['csi'][th]:.6f}")


# ============================================================
# RANK-BASED SELECTION
# ============================================================
def rank_values(values, higher_is_better):
    """
    Returns ranks starting at 1.
    Best value gets rank 1.
    """
    values = np.asarray(values)

    if higher_is_better:
        order = np.argsort(-values)
    else:
        order = np.argsort(values)

    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def select_best_alpha_by_ranks(alpha_results):
    mse_values = [r["mse"] for r in alpha_results]
    csi_low_values = [r["csi"][0.1] for r in alpha_results]
    csi_high_values = [r["csi"][5.0] for r in alpha_results]

    mse_ranks = rank_values(mse_values, higher_is_better=False)
    csi_low_ranks = rank_values(csi_low_values, higher_is_better=True)
    csi_high_ranks = rank_values(csi_high_values, higher_is_better=True)

    for i, r in enumerate(alpha_results):
        r["rank_mse"] = int(mse_ranks[i])
        r["rank_csi_0.1"] = int(csi_low_ranks[i])
        r["rank_csi_5.0"] = int(csi_high_ranks[i])
        r["rank_total"] = int(mse_ranks[i] + csi_low_ranks[i] + csi_high_ranks[i])

    best_idx = min(range(len(alpha_results)), key=lambda i: (alpha_results[i]["rank_total"], alpha_results[i]["mse"]))
    return alpha_results[best_idx], alpha_results


# ============================================================
# MAIN
# ============================================================
def main():
    convlstm_model = load_model("convlstm", CONVLSTM_CKPT, C_in, DEVICE, MAX_LEAD)
    unet_model = load_model("unet", UNET_CKPT, C_in, DEVICE, MAX_LEAD)

    print("\nCollecting VAL predictions (single pass)...")
    val_convlstm_result, val_unet_result, val_conv_preds, val_unet_preds, val_targets = \
        collect_predictions_and_metrics(
            convlstm_model, unet_model, val_loader, DEVICE, MAX_LEAD, CSI_THRESHOLDS
        )

    print_metrics("ConvLSTM VAL", val_convlstm_result)
    print_metrics("UNet VAL", val_unet_result)

    print("\n=== Ensemble search on validation (cached predictions) ===")
    alpha_results = []
    for alpha in ALPHAS:
        result = evaluate_alpha_from_cached_predictions(
            val_conv_preds, val_unet_preds, val_targets, alpha, CSI_THRESHOLDS
        )
        alpha_results.append(result)

    best_result, alpha_results = select_best_alpha_by_ranks(alpha_results)

    for r in alpha_results:
        print(
            f"alpha={r['alpha']:.2f} | "
            f"MSE={r['mse']:.6f} | MAE={r['mae']:.6f} | "
            f"CSI@0.1={r['csi'][0.1]:.6f} | CSI@5.0={r['csi'][5.0]:.6f} | "
            f"ranks=({r['rank_mse']},{r['rank_csi_0.1']},{r['rank_csi_5.0']}) | "
            f"total={r['rank_total']}"
        )

    print_metrics(f"Best Ensemble VAL (alpha={best_result['alpha']:.2f})", best_result)
    print(
        f"Selected by rank sum: "
        f"MSE rank={best_result['rank_mse']}, "
        f"CSI@0.1 rank={best_result['rank_csi_0.1']}, "
        f"CSI@5.0 rank={best_result['rank_csi_5.0']}, "
        f"total={best_result['rank_total']}"
    )

    print("\nCollecting TEST predictions (single pass)...")
    test_convlstm_result, test_unet_result, test_conv_preds, test_unet_preds, test_targets = \
        collect_predictions_and_metrics(
            convlstm_model, unet_model, test_loader, DEVICE, MAX_LEAD, CSI_THRESHOLDS
        )

    print_metrics("ConvLSTM TEST", test_convlstm_result)
    print_metrics("UNet TEST", test_unet_result)

    test_ensemble_result = evaluate_alpha_from_cached_predictions(
        test_conv_preds, test_unet_preds, test_targets, best_result["alpha"], CSI_THRESHOLDS
    )
    print_metrics(f"Best Ensemble TEST (alpha={best_result['alpha']:.2f})", test_ensemble_result)


if __name__ == "__main__":
    main()

# Device: cuda
# Loaded convlstm checkpoint from checkpoints/convlstm/mse/epoch3_full.pt | epoch=3
# Loaded unet checkpoint from checkpoints/unet/best_mse_true.pt | epoch=5

# Collecting VAL predictions (single pass)...
# Batch 0/363
# Batch 20/363
# Batch 40/363
# Batch 60/363
# Batch 80/363
# Batch 100/363
# Batch 120/363
# Batch 140/363
# Batch 160/363
# Batch 180/363
# Batch 200/363
# Batch 220/363
# Batch 240/363
# Batch 260/363
# Batch 280/363
# Batch 300/363
# Batch 320/363
# Batch 340/363
# Batch 360/363

# === ConvLSTM VAL ===
# MSE: 0.722578
# MAE: 0.328448
# CSI @ 0.1 mm: 0.676467
# CSI @ 5.0 mm: 0.399532

# === UNet VAL ===
# MSE: 0.679498
# MAE: 0.331537
# CSI @ 0.1 mm: 0.620743
# CSI @ 5.0 mm: 0.428984

# === Ensemble search on validation (cached predictions) ===
# alpha=0.00 | MSE=0.679498 | MAE=0.331537 | CSI@0.1=0.620743 | CSI@5.0=0.428984 | ranks=(42,51,27) | total=120
# alpha=0.02 | MSE=0.675292 | MAE=0.330145 | CSI@0.1=0.622712 | CSI@5.0=0.429624 | ranks=(40,50,25) | total=115
# alpha=0.04 | MSE=0.671293 | MAE=0.328817 | CSI@0.1=0.624646 | CSI@5.0=0.430281 | ranks=(38,49,24) | total=111
# alpha=0.06 | MSE=0.667501 | MAE=0.327551 | CSI@0.1=0.626527 | CSI@5.0=0.430997 | ranks=(36,48,21) | total=105
# alpha=0.08 | MSE=0.663916 | MAE=0.326344 | CSI@0.1=0.628375 | CSI@5.0=0.431579 | ranks=(34,47,19) | total=100
# alpha=0.10 | MSE=0.660538 | MAE=0.325195 | CSI@0.1=0.630190 | CSI@5.0=0.432109 | ranks=(32,46,17) | total=95
# alpha=0.12 | MSE=0.657366 | MAE=0.324103 | CSI@0.1=0.631962 | CSI@5.0=0.432574 | ranks=(30,45,15) | total=90
# alpha=0.14 | MSE=0.654401 | MAE=0.323067 | CSI@0.1=0.633709 | CSI@5.0=0.432997 | ranks=(28,44,14) | total=86
# alpha=0.16 | MSE=0.651643 | MAE=0.322087 | CSI@0.1=0.635427 | CSI@5.0=0.433354 | ranks=(26,43,12) | total=81
# alpha=0.18 | MSE=0.649092 | MAE=0.321163 | CSI@0.1=0.637121 | CSI@5.0=0.433693 | ranks=(24,42,10) | total=76
# alpha=0.20 | MSE=0.646748 | MAE=0.320294 | CSI@0.1=0.638795 | CSI@5.0=0.433921 | ranks=(22,41,7) | total=70
# alpha=0.22 | MSE=0.644611 | MAE=0.319480 | CSI@0.1=0.640445 | CSI@5.0=0.434087 | ranks=(20,40,6) | total=66
# alpha=0.24 | MSE=0.642680 | MAE=0.318720 | CSI@0.1=0.642078 | CSI@5.0=0.434206 | ranks=(18,39,3) | total=60
# alpha=0.26 | MSE=0.640956 | MAE=0.318015 | CSI@0.1=0.643672 | CSI@5.0=0.434225 | ranks=(16,38,2) | total=56
# alpha=0.28 | MSE=0.639439 | MAE=0.317364 | CSI@0.1=0.645264 | CSI@5.0=0.434197 | ranks=(14,37,4) | total=55
# alpha=0.30 | MSE=0.638129 | MAE=0.316767 | CSI@0.1=0.646816 | CSI@5.0=0.434231 | ranks=(12,36,1) | total=49
# alpha=0.32 | MSE=0.637026 | MAE=0.316223 | CSI@0.1=0.648338 | CSI@5.0=0.434093 | ranks=(10,35,5) | total=50
# alpha=0.34 | MSE=0.636129 | MAE=0.315732 | CSI@0.1=0.649846 | CSI@5.0=0.433888 | ranks=(8,34,8) | total=50
# alpha=0.36 | MSE=0.635440 | MAE=0.315294 | CSI@0.1=0.651336 | CSI@5.0=0.433698 | ranks=(6,33,9) | total=48
# alpha=0.38 | MSE=0.634957 | MAE=0.314910 | CSI@0.1=0.652794 | CSI@5.0=0.433391 | ranks=(4,32,11) | total=47
# alpha=0.40 | MSE=0.634681 | MAE=0.314578 | CSI@0.1=0.654223 | CSI@5.0=0.433015 | ranks=(2,31,13) | total=46
# alpha=0.42 | MSE=0.634612 | MAE=0.314299 | CSI@0.1=0.655625 | CSI@5.0=0.432560 | ranks=(1,30,16) | total=47
# alpha=0.44 | MSE=0.634750 | MAE=0.314072 | CSI@0.1=0.657007 | CSI@5.0=0.432102 | ranks=(3,29,18) | total=50
# alpha=0.46 | MSE=0.635094 | MAE=0.313897 | CSI@0.1=0.658363 | CSI@5.0=0.431547 | ranks=(5,28,20) | total=53
# alpha=0.48 | MSE=0.635646 | MAE=0.313773 | CSI@0.1=0.659679 | CSI@5.0=0.430887 | ranks=(7,27,22) | total=56
# alpha=0.50 | MSE=0.636404 | MAE=0.313701 | CSI@0.1=0.660976 | CSI@5.0=0.430288 | ranks=(9,26,23) | total=58
# alpha=0.52 | MSE=0.637369 | MAE=0.313680 | CSI@0.1=0.662230 | CSI@5.0=0.429616 | ranks=(11,25,26) | total=62
# alpha=0.54 | MSE=0.638541 | MAE=0.313711 | CSI@0.1=0.663455 | CSI@5.0=0.428864 | ranks=(13,24,28) | total=65
# alpha=0.56 | MSE=0.639919 | MAE=0.313792 | CSI@0.1=0.664641 | CSI@5.0=0.428028 | ranks=(15,23,29) | total=67
# alpha=0.58 | MSE=0.641505 | MAE=0.313925 | CSI@0.1=0.665788 | CSI@5.0=0.427068 | ranks=(17,22,30) | total=69
# alpha=0.60 | MSE=0.643297 | MAE=0.314108 | CSI@0.1=0.666889 | CSI@5.0=0.426195 | ranks=(19,21,31) | total=71
# alpha=0.62 | MSE=0.645296 | MAE=0.314342 | CSI@0.1=0.667965 | CSI@5.0=0.425183 | ranks=(21,20,32) | total=73
# alpha=0.64 | MSE=0.647502 | MAE=0.314627 | CSI@0.1=0.668996 | CSI@5.0=0.424188 | ranks=(23,19,33) | total=75
# alpha=0.66 | MSE=0.649915 | MAE=0.314961 | CSI@0.1=0.669986 | CSI@5.0=0.423162 | ranks=(25,18,34) | total=77
# alpha=0.68 | MSE=0.652535 | MAE=0.315347 | CSI@0.1=0.670915 | CSI@5.0=0.422022 | ranks=(27,17,35) | total=79
# alpha=0.70 | MSE=0.655361 | MAE=0.315782 | CSI@0.1=0.671798 | CSI@5.0=0.420802 | ranks=(29,16,36) | total=81
# alpha=0.72 | MSE=0.658395 | MAE=0.316268 | CSI@0.1=0.672628 | CSI@5.0=0.419595 | ranks=(31,15,37) | total=83
# alpha=0.74 | MSE=0.661635 | MAE=0.316804 | CSI@0.1=0.673394 | CSI@5.0=0.418337 | ranks=(33,14,38) | total=85
# alpha=0.76 | MSE=0.665082 | MAE=0.317390 | CSI@0.1=0.674114 | CSI@5.0=0.417108 | ranks=(35,13,39) | total=87
# alpha=0.78 | MSE=0.668736 | MAE=0.318026 | CSI@0.1=0.674773 | CSI@5.0=0.415773 | ranks=(37,12,40) | total=89
# alpha=0.80 | MSE=0.672596 | MAE=0.318712 | CSI@0.1=0.675363 | CSI@5.0=0.414439 | ranks=(39,11,41) | total=91
# alpha=0.82 | MSE=0.676664 | MAE=0.319450 | CSI@0.1=0.675878 | CSI@5.0=0.413043 | ranks=(41,10,42) | total=93
# alpha=0.84 | MSE=0.680938 | MAE=0.320238 | CSI@0.1=0.676306 | CSI@5.0=0.411696 | ranks=(43,9,43) | total=95
# alpha=0.86 | MSE=0.685419 | MAE=0.321076 | CSI@0.1=0.676670 | CSI@5.0=0.410290 | ranks=(44,7,44) | total=95
# alpha=0.88 | MSE=0.690107 | MAE=0.321966 | CSI@0.1=0.676955 | CSI@5.0=0.408786 | ranks=(45,5,45) | total=95
# alpha=0.90 | MSE=0.695002 | MAE=0.322908 | CSI@0.1=0.677152 | CSI@5.0=0.407305 | ranks=(46,3,46) | total=95
# alpha=0.92 | MSE=0.700103 | MAE=0.323903 | CSI@0.1=0.677231 | CSI@5.0=0.405820 | ranks=(47,1,47) | total=95
# alpha=0.94 | MSE=0.705412 | MAE=0.324951 | CSI@0.1=0.677201 | CSI@5.0=0.404284 | ranks=(48,2,48) | total=98
# alpha=0.96 | MSE=0.710927 | MAE=0.326055 | CSI@0.1=0.677078 | CSI@5.0=0.402740 | ranks=(49,4,49) | total=102
# alpha=0.98 | MSE=0.716649 | MAE=0.327218 | CSI@0.1=0.676823 | CSI@5.0=0.401150 | ranks=(50,6,50) | total=106
# alpha=1.00 | MSE=0.722578 | MAE=0.328448 | CSI@0.1=0.676467 | CSI@5.0=0.399532 | ranks=(51,8,51) | total=110

# === Best Ensemble VAL (alpha=0.40) ===
# MSE: 0.634681
# MAE: 0.314578
# CSI @ 0.1 mm: 0.654223
# CSI @ 5.0 mm: 0.433015
# Selected by rank sum: MSE rank=2, CSI@0.1 rank=31, CSI@5.0 rank=13, total=46

# Collecting TEST predictions (single pass)...
# Batch 0/363
# Batch 20/363
# Batch 40/363
# Batch 60/363
# Batch 80/363
# Batch 100/363
# Batch 120/363
# Batch 140/363
# Batch 160/363
# Batch 180/363
# Batch 200/363
# Batch 220/363
# ^[[B^[[ABatch 240/363
# Batch 260/363
# Batch 280/363
# Batch 300/363
# Batch 320/363
# Batch 340/363
# Batch 360/363

# === ConvLSTM TEST ===
# MSE: 0.756259
# MAE: 0.334600
# CSI @ 0.1 mm: 0.681158
# CSI @ 5.0 mm: 0.398580

# === UNet TEST ===
# MSE: 0.702081
# MAE: 0.335946
# CSI @ 0.1 mm: 0.626054
# CSI @ 5.0 mm: 0.435565

# === Best Ensemble TEST (alpha=0.40) ===
# MSE: 0.659273
# MAE: 0.319197
# CSI @ 0.1 mm: 0.659404
# CSI @ 5.0 mm: 0.437381