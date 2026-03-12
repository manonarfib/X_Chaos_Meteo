import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend non interactif (clusters)

import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

ALPHA_PATH = "models/mixture/alpha_map_val.npy"
OUTPUT_PATH = "models/mixture/alpha_map.png"

STD_FACTOR = 2.0
CMAP = "coolwarm"


# ============================================================
# LOAD
# ============================================================

alpha = np.load(ALPHA_PATH)

alpha_min = float(alpha.min())
alpha_max = float(alpha.max())
alpha_mean = float(alpha.mean())
alpha_std = float(alpha.std())

print("Alpha statistics")
print("----------------")
print("min  :", alpha_min)
print("max  :", alpha_max)
print("mean :", alpha_mean)
print("std  :", alpha_std)


# ============================================================
# COLOR SCALE
# ============================================================

vmin = alpha_mean - STD_FACTOR * alpha_std
vmax = alpha_mean + STD_FACTOR * alpha_std

vmin = max(0.0, vmin)
vmax = min(1.0, vmax)


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(10,6))

im = plt.imshow(
    alpha,
    cmap=CMAP,
    vmin=vmin,
    vmax=vmax,
    origin="lower"
)

plt.colorbar(im, label="Alpha weight (ConvLSTM)")

plt.title(
    f"Spatial Alpha Map\n"
    f"min={alpha_min:.3f}  max={alpha_max:.3f}  mean={alpha_mean:.3f}  std={alpha_std:.3f}"
)

plt.xlabel("Longitude index")
plt.ylabel("Latitude index")

plt.tight_layout()

plt.savefig(OUTPUT_PATH, dpi=300)
plt.close()

print("Saved figure to:", OUTPUT_PATH)