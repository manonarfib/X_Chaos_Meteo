import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import ndimage
from skimage.measure import find_contours


from models.utils.ERA5_dataset_from_local import ERA5Dataset
from models.ConvLSTM.convlstm import PrecipConvLSTM

DATASET_PATH = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"
CKPT_PATH = "explainability/epoch3_full.pt"

T = 8
LEAD = 1
BATCH_SIZE = 8
SAMPLE_IDX = 982
THRESHOLD = 2.0

CLUSTER_ID_TO_PLOT = 0

EUROPE_REGION = (-12.5, 42.5, 35, 72)  # lon_min, lon_max, lat_min, lat_max

SAVE_IMG_PATH=os.path.join('explainability', 'clusters', f'rain_above_{int(THRESHOLD)}_sample_{SAMPLE_IDX}_clusters.png')
SAVE_IMG_PATH_ONE_CLUSTER=os.path.join('explainability', 'clusters', f'rain_above_{int(THRESHOLD)}_sample_{SAMPLE_IDX}_cluster_{CLUSTER_ID_TO_PLOT}.png')



def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    return device


def apply_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = data.astype(float).copy()
    out[~mask.astype(bool)] = np.nan
    return out


def build_europe_mask(lons: np.ndarray, lats: np.ndarray, region):
    lon_min, lon_max, lat_min, lat_max = region
    Lon, Lat = np.meshgrid(lons, lats)

    mask = (
        (Lon >= lon_min) & (Lon <= lon_max) &
        (Lat >= lat_min) & (Lat <= lat_max)
    )
   
    return mask


def load_dataset():
    dataset = ERA5Dataset(DATASET_PATH, T=T, lead=LEAD)
    return dataset


def load_model(input_channels: int, device: torch.device) -> PrecipConvLSTM:
    model = PrecipConvLSTM(
        input_channels=input_channels,
        hidden_channels=[32, 64],
        kernel_size=3,
    ).to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint epoch={ckpt.get('epoch', 'unknown')} from {CKPT_PATH}")
    return model

def extract_large_rain_clusters(
    binary_map: np.ndarray,
    min_size: int = 1000,
):
    """
    Identifie les clusters connexes dans une carte binaire
    et retourne ceux dont la surface dépasse min_size pixels.
    """
    structure = np.ones((3, 3))  # connectivité 8
    labeled, num_labels = ndimage.label(binary_map, structure=structure)

    sizes = ndimage.sum(binary_map, labeled, range(1, num_labels + 1))

    large_labels = [
        i + 1 for i, size in enumerate(sizes) if size >= min_size
    ]

    return labeled, large_labels



@torch.no_grad()
def predict_sample(model, dataset, idx: int, device: torch.device):
    X, y, *_ = dataset[idx]

    X = X.unsqueeze(0).to(device).float()
    y = y.unsqueeze(0).to(device).float()

    y_hat = model(X).squeeze(1)  # (B, H, W)
    return y.squeeze(0).cpu().numpy(), y_hat.squeeze(0).cpu().numpy()


def plot_binary_map(
    data: np.ndarray,
    region,
    title: str,
    save_path=None,
    labeled_map=None,
    cluster_labels=None,
    lons=None,
    lats=None,
):
    
    display_labels = {lbl: i for i, lbl in enumerate(large_clusters)}

    lon_min, lon_max, lat_min, lat_max = region
    extent = [lon_min, lon_max, lat_min, lat_max]

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        figsize=(6, 4),
        subplot_kw={"projection": proj}
    )

    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)

    gl = ax.gridlines(linestyle="--", linewidth=0.4)
    gl.right_labels = False
    gl.top_labels = False

    ax.imshow(
        data.astype(np.float32) * 0.4,
        cmap="Greys",
        origin="upper",
        extent=extent,
        transform=proj,
        vmin=0,
        vmax=1,
    )

    # # --- Contours rouges ---
    # if labeled_map is not None and cluster_labels is not None:
    #     for lbl in cluster_labels:
    #         mask = (labeled_map == lbl).astype(np.uint8)

    #         contours = find_contours(mask, level=0.5)
    #         for contour in contours:
    #             y_idx, x_idx = contour[:, 0], contour[:, 1]

    #             lon = np.interp(x_idx, np.arange(len(lons)), lons)
    #             lat = np.interp(y_idx, np.arange(len(lats)), lats)

    #             ax.plot(
    #                 lon,
    #                 lat,
    #                 color="red",
    #                 linewidth=1.5,
    #                 transform=proj,
    #             )

    # --- Contours + labels ---
    if labeled_map is not None and cluster_labels is not None:
        for lbl in cluster_labels:
            mask = (labeled_map == lbl).astype(np.uint8)

            # Contours
            contours = find_contours(mask, level=0.5)
            for contour in contours:
                y_idx, x_idx = contour[:, 0], contour[:, 1]

                lon = np.interp(x_idx, np.arange(len(lons)), lons)
                lat = np.interp(y_idx, np.arange(len(lats)), lats)

                ax.plot(
                    lon,
                    lat,
                    color="red",
                    linewidth=1.5,
                    transform=proj,
                )

            # --- Centroïde pour l'étiquette ---
            y_centroid, x_centroid = np.mean(np.where(mask), axis=1)

            lon_c = np.interp(x_centroid, np.arange(len(lons)), lons)
            lat_c = np.interp(y_centroid, np.arange(len(lats)), lats)

            ax.text(
                lon_c,
                lat_c,
                str(display_labels[lbl]),
                color="red",
                fontsize=9,
                fontweight="normal",
                ha="center",
                va="center",
                transform=proj,
                path_effects=[
                    pe.withStroke(linewidth=2.5, foreground="white")
                ],
            )


    ax.set_title(title)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_single_cluster_contour(
    data: np.ndarray,
    region,
    title: str,
    cluster_id,
    save_path=None,
    labeled_map=None,
    cluster_labels=None,
    lons=None,
    lats=None,
):
    lon_min, lon_max, lat_min, lat_max = region
    extent = [lon_min, lon_max, lat_min, lat_max]

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        figsize=(6, 4),
        subplot_kw={"projection": proj}
    )

    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)

    gl = ax.gridlines(linestyle="--", linewidth=0.4)
    gl.right_labels = False
    gl.top_labels = False

    ax.imshow(
        data.astype(np.float32) * 0.4,
        cmap="Greys",
        origin="upper",
        extent=extent,
        transform=proj,
        vmin=0,
        vmax=1,
    )

    lbl=cluster_labels[cluster_id]
    mask = (labeled_map == lbl).astype(np.uint8)
    contours = find_contours(mask, level=0.5)
    for contour in contours:
        y_idx, x_idx = contour[:, 0], contour[:, 1]

        lon = np.interp(x_idx, np.arange(len(lons)), lons)
        lat = np.interp(y_idx, np.arange(len(lats)), lats)

        ax.plot(
            lon,
            lat,
            color="red",
            linewidth=1.5,
            transform=proj,
        )

    ax.set_title(title)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    device = get_device()

    dataset = load_dataset()
    input_vars = list(dataset.X.coords["channel"].values)
    model = load_model(len(input_vars), device)

    y, y_hat = predict_sample(model, dataset, SAMPLE_IDX, device)

    # Binarisation
    gt_bin = (y >= THRESHOLD).astype(np.uint8)
    pred_bin = (y_hat >= THRESHOLD).astype(np.uint8)

    # Coordonnées
    lats = dataset.X.coords["latitude"].values
    lons = dataset.X.coords["longitude"].values
    lons = np.where(lons > 180, lons - 360, lons)

    europe_mask = build_europe_mask(lons, lats, EUROPE_REGION)
    pred_masked = apply_mask(pred_bin, europe_mask)

    labeled_map, large_clusters = extract_large_rain_clusters(pred_masked, min_size=100)

    print(f"Detected {len(large_clusters)} large rain clusters")

    plot_binary_map(
    pred_masked,
    EUROPE_REGION,
    "Rain clusters > 2mm & > 1000 px",
    save_path=SAVE_IMG_PATH,
    labeled_map=labeled_map,
    cluster_labels=large_clusters,
    lons=lons,
    lats=lats,
)
    
    plot_single_cluster_contour(
    pred_masked,
    EUROPE_REGION,
    "Rain clusters > 2mm & > 1000 px",
    cluster_id=CLUSTER_ID_TO_PLOT,
    save_path=SAVE_IMG_PATH_ONE_CLUSTER,
    labeled_map=labeled_map,
    cluster_labels=large_clusters,
    lons=lons,
    lats=lats,
)
    # plot_binary_map(pred_masked, EUROPE_REGION, "Predicted precipitation (binary)", SAVE_IMG_PATH)
