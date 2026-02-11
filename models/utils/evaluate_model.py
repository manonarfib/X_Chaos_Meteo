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

dataset_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_validation.zarr"
T, lead = 8, 1
batch_size = 8
without_precip=False
max_lead = 1

dataset = ERA5Dataset(dataset_path, T=T, lead=lead, without_precip=without_precip, max_lead=max_lead)    
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
input_vars = list(dataset.X.coords["channel"].values)
C_in = len(input_vars)

model = PrecipConvLSTM(
    input_channels=C_in,
    hidden_channels=[32, 64],
    kernel_size=3,
    output_size=max_lead
).to(device)
# ckpt_path = "checkpoints_input_all_lead/epoch3_full.pt"
# ckpt_path = "epoch3_full.pt"
ckpt_path = "epoch1_full_mse_before_norm.pt"
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
        print(f"days {i[0]/4} to {(i[-1]+1)/4} computed")
        if (i[-1]+1)/4==10:
            break
        if max_lead==1:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            
            y_hat = model(X_batch).squeeze(1)  # (B,H,W)
            y_hat = torch.clamp(y_hat, min=0.0)
        else:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch[:, -1, :, :].to(device).float()
            y_hat = model(X_batch).squeeze(1)  # (B,H,W)
            y_hat = y_hat[:, -1, :, :]
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
hss_global = {}
pod_global = {}
far_global = {}
eps = 1e-8
for th in thresholds:
    a = tp_tot[th]
    b = fp_tot[th]
    c = fn_tot[th]
    d = tn_tot[th]
    csi_global[th] = a / (a + b + c + eps)
    hss_global[th] = 2*(a*d-b*c) / ((a+c)*(c+d)+(a+b)*(b+d) + eps)
    pod_global[th] = a/ (a+c + eps)
    far_global[th] = b / (a+b + eps)


print(f"Validation set metrics - MSE: {mse:.6f} | MAE: {mae:.6f}")
for th, csi in csi_global.items():
    print(f"CSI @ {th} mm: {csi:.6f}")
for th, hss in hss_global.items():
    print(f"HSS @ {th} mm: {hss:.6f}")
for th, pod in pod_global.items():
    print(f"POD @ {th} mm: {pod:.6f}")
for th, far in far_global.items():
    print(f"FAR @ {th} mm: {far:.6f}")

