import matplotlib.pyplot as plt
from models.utils.ERA5_dataset_from_local import ERA5Dataset
import numpy as np

train_dataset_path: str = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_train.zarr"
dataset = ERA5Dataset(train_dataset_path, T=8, lead=1)

# Récupération d'un échantillon
i = 1000
x, y, idx = dataset[i]

# Sélection du pas de temps 0
t = 0
_, C, H, W = x.shape

# Grille de subplots
ncols = 6
nrows = int(np.ceil(C / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
axes = axes.flatten()

for c in range(C):
    im = axes[c].imshow(x[t, c], origin="lower")
    axes[c].set_title(f"Channel {c}")
    axes[c].axis("off")
    plt.colorbar(im, ax=axes[c], fraction=0.046)

# Masquer les axes inutilisés
for ax in axes[C:]:
    ax.axis("off")

plt.suptitle(f"All X channels at t=0 (sample i={idx})", fontsize=14)
plt.tight_layout()

output_path = f"X_channels_t0_sample_{idx}.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close(fig)

import os
print("Current working directory:", os.getcwd())
print("Saved file:", os.path.abspath(output_path))