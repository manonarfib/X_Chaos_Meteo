import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from models.utils.ERA5_dataset_from_local import ERA5Dataset
from explainability.explainable_by_design.WeatherCBM import WeatherCBM


def load_model(CKPT_PATH, C_in, device):
    model = WeatherCBM(
        input_channels=C_in,
        hidden_channels=[32, 64],
        kernel_size=3,
        output_size=1,
        n_concepts=10
    ).to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model


def load_sample(dataset, sample_idx, device):
    X, y, *_ = dataset[sample_idx]

    X = X.unsqueeze(0).to(device).float()
    y = y.unsqueeze(0).to(device).float()

    return X, y


@torch.no_grad()
def forward_sample(model, X):
    y_hat, alpha = model(X)
    return y_hat, alpha


def normalize_map(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def save_alpha_maps_europe(alpha, sample_idx, save_root, B=None, region=(-12.5, 42.5, 35, 72)):
    """
    Save alpha maps with Europe background using Cartopy.
    alpha: torch.Tensor (1, K, H, W)
    B: optional, array of shape (K,) to show B_k in title
    """
    alpha = alpha.squeeze(0).cpu()  # (K, H, W)
    K, H, W = alpha.shape

    sample_dir = os.path.join(save_root, f"sample_{sample_idx}")
    os.makedirs(sample_dir, exist_ok=True)

    lon_min, lon_max, lat_min, lat_max = region
    extent = [lon_min, lon_max, lat_min, lat_max]
    proj = ccrs.PlateCarree()

    def _setup_ax(ax):
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.4)
        gl.right_labels = False
        gl.top_labels = False

    for k in range(K):
        alpha_map = alpha[k].numpy()
        vmin = 0
        vmax = np.max(alpha_map)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        _setup_ax(ax)

        im = ax.imshow(alpha_map, cmap="Reds", vmin=vmin, vmax=vmax,
                       origin="upper", extent=extent, transform=proj)

        title = f"Concept {k}"
        if B is not None:
            title += f" | B_k = {B[k]:.3f}"
        ax.set_title(title)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        save_path = os.path.join(sample_dir, f"concept_{k}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


if __name__ == "__main__":

    DATASET_PATH = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_validation.zarr"
    CKPT_PATH = "checkpoints/weather_cbm/exp_0/epoch6_full.pt"
    SAVE_ROOT = "explainability/explainable_by_design/explain_results/alpha_maps"

    SAMPLE_IDX = 0
    T = 8
    MAX_LEAD = 1
    LEAD = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ERA5Dataset(DATASET_PATH, T=T, lead=LEAD, without_precip=False, max_lead=MAX_LEAD)
    C_in = len(dataset.X.coords["channel"].values)

    model = load_model(CKPT_PATH, C_in, device)

    X, y = load_sample(dataset, SAMPLE_IDX, device)

    y_hat, alpha = forward_sample(model, X)

    B = model.linear_combination.weight.data.view(-1).cpu().numpy()  # (K,)
    save_alpha_maps_europe(alpha, SAMPLE_IDX, SAVE_ROOT, B)