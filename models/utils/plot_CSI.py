# Reproduction de la figure 1 de l'article Advanced Torrential Loss Function for Precipitation Forecasting

import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.utils.ERA5_dataset_from_local import  ERA5Dataset
from models.ConvLSTM.convlstm import PrecipConvLSTM


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

sample_idx = 50
X, y, *_ = dataset[sample_idx]

X = X.unsqueeze(0).to(device).float()  # ajouter dimension batch
y = y.unsqueeze(0).to(device).float()

with torch.no_grad():
    y_hat = model(X).squeeze(1)  # (B,H,W)
print("y predicted")

threshold = 1.0  # mm/h par exemple, à adapter

# y et y_hat : (1, H, W)
y_np = y.squeeze(0).cpu().numpy()
y_hat_np = y_hat.squeeze(0).cpu().numpy()

# Binarisation
gt_bin = (y_np >= threshold).astype(np.uint8)
pred_bin = (y_hat_np >= threshold).astype(np.uint8)

# Comptes CSI
hits = np.logical_and(pred_bin == 1, gt_bin == 1).sum()
false_alarms = np.logical_and(pred_bin == 1, gt_bin == 0).sum()
misses = np.logical_and(pred_bin == 0, gt_bin == 1).sum()

csi = hits / (hits + false_alarms + misses + 1e-8)

print(f"CSI (threshold={threshold}) = {csi:.4f}")
print(f"Hits={hits}, False Alarms={false_alarms}, Misses={misses}")

hits_mask = np.logical_and(pred_bin == 1, gt_bin == 1)
fa_mask    = np.logical_and(pred_bin == 1, gt_bin == 0)
miss_mask  = np.logical_and(pred_bin == 0, gt_bin == 1)

H, W = gt_bin.shape
pred_viz = np.zeros((H, W, 3), dtype=np.float32)

# Couleurs
pred_viz[hits_mask] = [1.0, 1.0, 0.0]   # jaune
pred_viz[fa_mask]   = [1.0, 0.4, 0.7]   # rose
pred_viz[miss_mask] = [0.2, 0.4, 1.0]   # bleu

rain_channel_idx = input_vars.index("tp_6h")
print("Rain channel index:", rain_channel_idx)
gt_T = X[0, -1, rain_channel_idx].cpu().numpy() 

gt_T_mask   = (gt_T >= threshold)
gt_T1_mask  = (y_np >= threshold)

def apply_mask(data, mask):
    out = data.copy().astype(float)
    out[~mask.astype(bool)] = np.nan
    return out

def cmap_with_white_bad(name):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("white")
    return cmap

lats = dataset.X.coords["latitude"].values
lons = dataset.X.coords["longitude"].values
lons = np.where(lons > 180, lons - 360, lons)

Lon, Lat = np.meshgrid(lons, lats)

region = (-12.5, 42.5, 35, 72)
lon_min, lon_max, lat_min, lat_max = region

europe_mask = (
    (Lon >= lon_min) & (Lon <= lon_max) &
    (Lat >= lat_min) & (Lat <= lat_max)
)

valid_mask = europe_mask & (
    hits_mask | fa_mask | miss_mask
)
pred_viz_rgba = np.zeros((H, W, 4), dtype=np.float32)
pred_viz_rgba[..., :3] = pred_viz
pred_viz_rgba[..., 3] = valid_mask.astype(np.float32)
# pred_viz_m = pred_viz.copy()
# pred_viz_m[~europe_mask] = np.nan

gt_T_m  = apply_mask(gt_T_mask, europe_mask)
gt_T1_m = apply_mask(gt_T1_mask, europe_mask)
gt_T_m_float = gt_T_m.astype(np.float32) * 0.7  # 0.7 = gris clair
gt_T1_m_float = gt_T1_m.astype(np.float32) * 0.7

def cmap_with_light_gray(name="Greys"):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("white")  # valeurs masquées en blanc
    return cmap

import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

proj = ccrs.PlateCarree()
extent = [lon_min, lon_max, lat_min, lat_max]

fig = plt.figure(figsize=(15, 4))
ax0 = fig.add_subplot(1, 3, 1, projection=proj)
ax1 = fig.add_subplot(1, 3, 2, projection=proj)
ax2 = fig.add_subplot(1, 3, 3, projection=proj)
axes = [ax0, ax1, ax2]

# --- Fond carte Europe (copié de ton code) ---
for ax in axes:
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, edgecolor='gray')
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, edgecolor='gray')
    gl = ax.gridlines(linestyle="--", linewidth=0.4)
    gl.right_labels = False
    gl.top_labels = False

im0 = ax0.imshow(
    # gt_T_m_float,
    gt_T_m.astype(np.float32)*0.2,
    # cmap=cmap_with_light_gray(),
    cmap=plt.get_cmap("Greys"),
    origin="upper",
    extent=extent,
    transform=proj,
    vmin=0, vmax=1  
)
# Contours gris moyen
ax0.contour(gt_T_m_float, levels=[0.5], colors="gray", linewidths=0.6, 
            origin="upper", extent=extent, transform=proj)
ax0.set_title("GT ≥ 1 mm / 6h à T")

im1 = ax1.imshow(
    # gt_T1_m_float,
    # cmap=cmap_with_light_gray(),
    gt_T1_m.astype(np.float32)*0.2,
    cmap=plt.get_cmap("Greys"),
    origin="upper",
    extent=extent,
    transform=proj,
    vmin=0, vmax=1
)
ax1.contour(gt_T1_m_float, levels=[0.5], colors="gray", linewidths=0.6, 
            origin="upper", extent=extent, transform=proj)
ax1.set_title("GT ≥ 1 mm / 6h à T+6h")

ax2.imshow(
    pred_viz_rgba,
    origin="upper",
    extent=extent,
    transform=proj,
)
ax2.set_title(f"Prévision T+6h\nCSI 1mm = {csi:.3f}")

from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=[1.0, 1.0, 0.0], edgecolor='k', label='Hits'),
    Patch(facecolor=[1.0, 0.4, 0.7], edgecolor='k', label='False Alarm'),
    Patch(facecolor=[0.2, 0.4, 1.0], edgecolor='k', label='Miss'),
]

ax2.legend(handles=legend_elements, loc='lower right', framealpha=1.0)

output_path = "CSI_plot_idx50.png"
plt.savefig(
    output_path,
    dpi=300,              # qualité publication
    bbox_inches="tight"   # pas de marges inutiles
)

plt.close()