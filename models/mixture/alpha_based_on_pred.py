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

CONVLSTM_CKPT = "checkpoints/convlstm/mse/epoch3_full.pt"
UNET_CKPT = "checkpoints/unet/best_mse_true.pt"

USE_TRAIN_SUBSAMPLE = True
TRAIN_SUBSAMPLE_YEARS = 2
RANDOM_SEED = 42

CSI_THRESHOLDS = [0.1, 5.0]

# grille de recherche
TAUS = [0.1, 0.5, 1.0, 5.0, 10.0]
ALPHAS_LOW = np.linspace(0.0, 1.0, 6)
ALPHAS_HIGH = np.linspace(0.0, 1.0, 6)


# ============================================================
# DATA
# ============================================================
def build_random_train_sampler(dataset, n_years, seed=42):
    n_samples = min(len(dataset), n_years * 365 * 4)  # approx 2 years in 6h steps
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n_samples, replace=False)
    return SubsetRandomSampler(indices.tolist())


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


input_vars = list(train_dataset.X.coords["channel"].values)
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

    if model_type == "unet":
        return WFUNet_with_train(T, 149, 221, c_in, max_lead, 8, 32, 0).to(device)

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


def print_metrics(name, result):
    print(f"\n=== {name} ===")
    print(f"MSE: {result['mse']:.6f}")
    print(f"MAE: {result['mae']:.6f}")
    for th in CSI_THRESHOLDS:
        print(f"CSI @ {th} mm: {result['csi'][th]:.6f}")


# ============================================================
# COLLECT PREDICTIONS ONCE
# ============================================================
@torch.inference_mode()
def collect_predictions_and_metrics(convlstm_model, unet_model, dataloader, device, max_lead, csi_thresholds, split_name="split"):
    conv_preds = []
    unet_preds = []
    targets_all = []

    conv_acc = init_metric_accumulators(csi_thresholds)
    unet_acc = init_metric_accumulators(csi_thresholds)

    for batch_idx, (X_batch, y_batch, *_) in enumerate(dataloader):
        if batch_idx % 20 == 0:
            print(f"{split_name} batch {batch_idx}/{len(dataloader)}")

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


# ============================================================
# THRESHOLD-DEPENDENT GATING
# ============================================================
def evaluate_threshold_rule_from_cached_predictions(conv_preds, unet_preds, targets, tau, alpha_low, alpha_high, csi_thresholds):
    mean_pred = 0.5 * (conv_preds + unet_preds)
    alpha_map = torch.where(
        mean_pred < tau,
        torch.full_like(mean_pred, float(alpha_low)),
        torch.full_like(mean_pred, float(alpha_high)),
    )

    ens_preds = alpha_map * conv_preds + (1.0 - alpha_map) * unet_preds
    ens_preds = torch.clamp(ens_preds, min=0.0)

    acc = init_metric_accumulators(csi_thresholds)
    update_metrics(acc, ens_preds, targets, csi_thresholds)

    result = finalize_metrics(acc, csi_thresholds)
    result["tau"] = float(tau)
    result["alpha_low"] = float(alpha_low)
    result["alpha_high"] = float(alpha_high)
    return result


# ============================================================
# RANK-BASED SELECTION
# ============================================================
def rank_values(values, higher_is_better):
    values = np.asarray(values)
    order = np.argsort(-values) if higher_is_better else np.argsort(values)
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def select_best_rule_by_ranks(rule_results):
    mse_values = [r["mse"] for r in rule_results]
    csi_low_values = [r["csi"][0.1] for r in rule_results]
    csi_high_values = [r["csi"][5.0] for r in rule_results]

    mse_ranks = rank_values(mse_values, higher_is_better=False)
    csi_low_ranks = rank_values(csi_low_values, higher_is_better=True)
    csi_high_ranks = rank_values(csi_high_values, higher_is_better=True)

    for i, r in enumerate(rule_results):
        r["rank_mse"] = int(mse_ranks[i])
        r["rank_csi_0.1"] = int(csi_low_ranks[i])
        r["rank_csi_5.0"] = int(csi_high_ranks[i])
        r["rank_total"] = int(mse_ranks[i] + csi_low_ranks[i] + csi_high_ranks[i])

    best_idx = min(
        range(len(rule_results)),
        key=lambda i: (rule_results[i]["rank_total"], rule_results[i]["mse"])
    )
    return rule_results[best_idx], rule_results


# ============================================================
# MAIN
# ============================================================
def main():
    convlstm_model = load_model("convlstm", CONVLSTM_CKPT, C_in, DEVICE, MAX_LEAD)
    unet_model = load_model("unet", UNET_CKPT, C_in, DEVICE, MAX_LEAD)

    print("\nCollecting TRAIN predictions (single pass)...")
    train_convlstm_result, train_unet_result, train_conv_preds, train_unet_preds, train_targets = \
        collect_predictions_and_metrics(
            convlstm_model, unet_model, train_loader, DEVICE, MAX_LEAD, CSI_THRESHOLDS, split_name="TRAIN"
        )

    print_metrics("ConvLSTM TRAIN", train_convlstm_result)
    print_metrics("UNet TRAIN", train_unet_result)

    print("\n=== Threshold-rule search on TRAIN (cached predictions) ===")
    rule_results = []
    for tau in TAUS:
        for alpha_low in ALPHAS_LOW:
            for alpha_high in ALPHAS_HIGH:
                result = evaluate_threshold_rule_from_cached_predictions(
                    train_conv_preds,
                    train_unet_preds,
                    train_targets,
                    tau,
                    alpha_low,
                    alpha_high,
                    CSI_THRESHOLDS
                )
                rule_results.append(result)

    best_train_rule, rule_results = select_best_rule_by_ranks(rule_results)

    print(
        f"Best TRAIN rule: tau={best_train_rule['tau']:.3f} | "
        f"alpha_low={best_train_rule['alpha_low']:.2f} | "
        f"alpha_high={best_train_rule['alpha_high']:.2f}"
    )
    print_metrics("Best Rule TRAIN", best_train_rule)
    print(
        f"Ranks: MSE={best_train_rule['rank_mse']} | "
        f"CSI@0.1={best_train_rule['rank_csi_0.1']} | "
        f"CSI@5.0={best_train_rule['rank_csi_5.0']} | "
        f"TOTAL={best_train_rule['rank_total']}"
    )

    print("\nCollecting VAL predictions (single pass)...")
    val_convlstm_result, val_unet_result, val_conv_preds, val_unet_preds, val_targets = \
        collect_predictions_and_metrics(
            convlstm_model, unet_model, val_loader, DEVICE, MAX_LEAD, CSI_THRESHOLDS, split_name="VAL"
        )

    print_metrics("ConvLSTM VAL", val_convlstm_result)
    print_metrics("UNet VAL", val_unet_result)

    val_rule_result = evaluate_threshold_rule_from_cached_predictions(
        val_conv_preds,
        val_unet_preds,
        val_targets,
        best_train_rule["tau"],
        best_train_rule["alpha_low"],
        best_train_rule["alpha_high"],
        CSI_THRESHOLDS
    )
    print_metrics(
        f"Threshold Ensemble VAL "
        f"(tau={best_train_rule['tau']:.3f}, "
        f"alpha_low={best_train_rule['alpha_low']:.2f}, "
        f"alpha_high={best_train_rule['alpha_high']:.2f})",
        val_rule_result
    )


if __name__ == "__main__":
    main()

# Collecting TRAIN predictions (single pass)...
# TRAIN batch 0/365
# TRAIN batch 20/365
# TRAIN batch 40/365
# TRAIN batch 60/365
# TRAIN batch 80/365
# TRAIN batch 100/365
# TRAIN batch 120/365
# TRAIN batch 140/365
# TRAIN batch 160/365
# TRAIN batch 180/365
# TRAIN batch 200/365
# TRAIN batch 220/365
# TRAIN batch 240/365
# TRAIN batch 260/365
# TRAIN batch 280/365
# TRAIN batch 300/365
# TRAIN batch 320/365
# TRAIN batch 340/365
# TRAIN batch 360/365

# === ConvLSTM TRAIN ===
# MSE: 0.646735
# MAE: 0.314666
# CSI @ 0.1 mm: 0.681571
# CSI @ 5.0 mm: 0.405957

# === UNet TRAIN ===
# MSE: 0.599645
# MAE: 0.316443
# CSI @ 0.1 mm: 0.626334
# CSI @ 5.0 mm: 0.443566

# === Threshold-rule search on TRAIN (cached predictions) ===
# Best TRAIN rule: tau=0.500 | alpha_low=1.00 | alpha_high=0.40

# === Best Rule TRAIN ===
# MSE: 0.561668
# MAE: 0.296029
# CSI @ 0.1 mm: 0.681766
# CSI @ 5.0 mm: 0.444860
# Ranks: MSE=18 | CSI@0.1=5 | CSI@5.0=33 | TOTAL=56

# Collecting VAL predictions (single pass)...
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
# CSI @ 5.0 mm: 0.399532

# === UNet VAL ===
# MSE: 0.679498
# MAE: 0.331537
# CSI @ 0.1 mm: 0.620745
# CSI @ 5.0 mm: 0.428984

# === Threshold Ensemble VAL (tau=0.500, alpha_low=1.00, alpha_high=0.40) ===
# MSE: 0.635548
# MAE: 0.310249
# CSI @ 0.1 mm: 0.676685
# CSI @ 5.0 mm: 0.433017