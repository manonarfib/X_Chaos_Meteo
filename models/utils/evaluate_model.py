# Two metrics are implemented :
# - MSE
# - Critical Success Index

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from models.utils.ERA5_dataset_from_local import  ERA5Dataset
from models.ConvLSTM.convlstm import PrecipConvLSTM
from models.unet.model_without_collapse import WFUNet_with_train

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

dataset_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"
T, lead = 8, 1
batch_size = 8

dataset = ERA5Dataset(dataset_path, T=T, lead=lead)    
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
input_vars = list(dataset.X.coords["channel"].values)
C_in = len(input_vars)

model = PrecipConvLSTM(
    input_channels=C_in,
    hidden_channels=[32, 64],
    kernel_size=3,
).to(device)

ckpt_path = "epoch3_full.pt"
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded checkpoint epoch={ckpt.get('epoch', 'unknown')} from {ckpt_path}")

# Metrics accumulators
mse_sum = 0.0
mae_sum = 0.0
num_pixels = 0

# CSI accumulators per threshold
thresholds = [0.1, 1.0, 5.0, 10.0]
tp_tot = {th: 0 for th in thresholds}
fp_tot = {th: 0 for th in thresholds}
fn_tot = {th: 0 for th in thresholds}
tn_tot = {th: 0 for th in thresholds}
print(len(test_loader))
with torch.no_grad():
    for X_batch, y_batch, i in test_loader:
        print(f"days {i[0]} to {(i[-1]+1)/4} computed")
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device).float()
        
        y_hat = model(X_batch).squeeze(1)  # (B,H,W)
        y_hat = torch.clamp(y_hat, min=0.0)
        
        # MSE & MAE
        mse_sum += nn.MSELoss(reduction='sum')(y_hat, y_batch).item()
        mae_sum += torch.sum(torch.abs(y_hat - y_batch)).item()
        num_pixels += y_batch.numel()
        
        # accum
        for th in thresholds:
            pred_bin = y_hat >= th
            true_bin = y_batch >= th
            tp_tot[th] += torch.logical_and(pred_bin, true_bin).sum().item()
            fp_tot[th] += torch.logical_and(pred_bin, ~true_bin).sum().item()
            fn_tot[th] += torch.logical_and(~pred_bin, true_bin).sum().item()
            tn_tot[th] += torch.logical_and(~pred_bin, ~true_bin).sum().item()

# Average metrics
mse = mse_sum / num_pixels
mae = mae_sum / num_pixels

# Global CSI
csi_global = {}
eps = 1e-8
for th in thresholds:
    csi_global[th] = tp_tot[th] / (tp_tot[th] + fp_tot[th] + fn_tot[th] + eps)

# Global Heidke Skill Score (HSS) (better when close to 1)
hss_global = {}
eps = 1e-8
for th in thresholds:
    a = tp_tot[th]
    b = fn_tot[th]
    c = fp_tot[th]
    d = tn_tot[th]
    hss_global[th] = 2*(a*d-b*c) / ((a+c)*(c+d)+(a+b)*(b+d) + eps)

# Probability of Detection (POD) (better when close to 1)
pod_global = {}
eps = 1e-8
for th in thresholds:
    a = tp_tot[th]
    b = fn_tot[th]
    c = fp_tot[th]
    d = tn_tot[th]
    pod_global[th] = a/ (a+b + eps)


# False Alarm Ratio (better when close to 0)
far_global = {}
eps = 1e-8
for th in thresholds:
    a = tp_tot[th]
    b = fn_tot[th]
    c = fp_tot[th]
    d = tn_tot[th]
    far_global[th] = b / (a+b + eps)


print(f"Test set metrics - MSE: {mse:.6f} | MAE: {mae:.6f}")
for th, csi in csi_global.items():
    print(f"CSI @ {th} mm: {csi:.6f}")
for th, hss in hss_global.items():
    print(f"HSS @ {th} mm: {hss:.6f}")
for th, pod in pod_global.items():
    print(f"POD @ {th} mm: {pod:.6f}")
for th, far in far_global.items():
    print(f"FAR @ {th} mm: {far:.6f}")

# Résultats du ConvLSTM de base (MSE, tp_6h in, prévisions à 6h) :
# Test set metrics - MSE: 0.756467 | MAE: 0.336837
# CSI @ 0.1 mm: 0.681179
# CSI @ 1.0 mm: 0.587248
# CSI @ 5.0 mm: 0.398580
# CSI @ 10.0 mm: 0.264549


