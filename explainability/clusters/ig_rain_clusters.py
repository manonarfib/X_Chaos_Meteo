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
from models.unet.model_without_collapse import WFUNet_with_train


DATASET_PATH = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"
CKPT_PATH = "checkpoints/run_144949/best_checkpoint_epoch5_batch_idx6479.pt"

MODEL_TYPE="UNet"

T = 8
LEAD = 1
BATCH_SIZE = 8
SAMPLE_IDX = 982
THRESHOLD = 2.0

CLUSTER_ID_TO_PLOT = 0

EUROPE_REGION = (-12.5, 42.5, 35, 72)  # lon_min, lon_max, lat_min, lat_max

SAVE_IMGS_DIR_PATH=os.path.join('explainability', 'clusters', MODEL_TYPE, f"sample{SAMPLE_IDX}")
SAVE_IMG_PATH=os.path.join(SAVE_IMGS_DIR_PATH, f'rain_above_{int(THRESHOLD)}_sample_{SAMPLE_IDX}_clusters.png')
SAVE_IMG_PATH_ONE_CLUSTER=os.path.join(SAVE_IMGS_DIR_PATH, f'rain_above_{int(THRESHOLD)}_sample_{SAMPLE_IDX}_cluster_{CLUSTER_ID_TO_PLOT}.png')

T_VIEW=7
TOP_K_VARS=5
TIME_STEP_HOURS=6


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    return device


def build_model(model_type: str, C_in: int, T: int, device: torch.device) -> torch.nn.Module:
    model_type = model_type.lower().strip()
    if model_type == "convlstm":
        return PrecipConvLSTM(
            input_channels=C_in,
            hidden_channels=[32, 64],
            kernel_size=3,
        ).to(device)
    elif model_type == "unet":
        # garde exactement tes params UNet utilisés ailleurs
        return WFUNet_with_train(T, 149, 221, C_in, 1, 8, 32, 0).to(device)
    else:
        raise ValueError(f"Unknown model_type={model_type}")
    


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


def integrated_gradients(model, x, baseline, steps=30, target="region_mean", region_mask=None):
    """
    x: (B,T,C,H,W)
    baseline: (B,T,C,H,W)
    region_mask: (B,1,H,W) if target="region_sum"
    returns attr: (B,T,C,H,W)
    """
    assert x.shape == baseline.shape
    B, T, C, H, W = x.shape

    if target == "region_mean" and region_mask is None:
        region_mask = torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype)

    x = x.float()
    baseline = baseline.float()
    total_grad = torch.zeros_like(x, dtype=torch.float32)

    for s in range(1, steps + 1):
        alpha = s / steps
        x_alpha = baseline + alpha * (x - baseline)
        x_alpha.requires_grad_(True)

        y_hat = model(x_alpha)
        if y_hat.dim() == 3:
            y_hat = y_hat.unsqueeze(1)

        if target == "mean":
            S = y_hat.mean()
        elif target == "region_mean":
            S = (y_hat * region_mask).mean()
        else:
            raise ValueError(f"Unknown target: {target}")

        model.zero_grad(set_to_none=True)
        if x_alpha.grad is not None:
            x_alpha.grad.zero_()
        S.backward()

        total_grad += x_alpha.grad.detach()

    avg_grad = total_grad / float(steps)
    attr = (x - baseline) * avg_grad
    return attr


def find_precip_channel(input_vars):
    """
    Try to locate precip channel in input variables.
    Adjust candidates if needed.
    """
    candidates = ["tp_6h", "tp", "total_precipitation", "precip", "precipitation"]
    for cand in candidates:
        for i, name in enumerate(input_vars):
            if name == cand or cand in name:
                return i, name
    return None, None


def _apply_mask(arr2d: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    out = arr2d.astype(np.float32).copy()
    out[~mask2d.astype(bool)] = np.nan
    return out


def _cmap_with_white_bad(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color="white")  # NaNs will render white
    return cmap


def plot_tp_attr_contour(
    tp_map: np.ndarray,
    attr_map: np.ndarray,
    cluster_mask: np.ndarray,   # <-- nouveau
    europe_mask: np.ndarray,
    out_path: str,
    title: str,
    tp_title: str,
    attr_title: str,
    region=(-12.5, 42.5, 35, 72),
):
    """
    3 panels:
      1) tp (Blues)
      2) attribution (Reds)
      3) tp + contour(cluster)

    White outside Europe (mask).
    """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tp_m = _apply_mask(tp_map, europe_mask)
    at_m = _apply_mask(attr_map, europe_mask)

    blues = _cmap_with_white_bad("Blues")
    reds = _cmap_with_white_bad("Reds")

    lon_min, lon_max, lat_min, lat_max = region
    extent = [lon_min, lon_max, lat_min, lat_max]
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(20, 9))
    ax0 = fig.add_subplot(1, 3, 1, projection=proj)
    ax1 = fig.add_subplot(1, 3, 2, projection=proj)
    ax2 = fig.add_subplot(1, 3, 3, projection=proj)
    axes = [ax0, ax1, ax2]

    # --- Map features ---
    for ax in axes:
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)

        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.4)
        gl.right_labels = False
        gl.top_labels = False

    # --- Panel 1: TP ---
    im0 = ax0.imshow(
        tp_m,
        cmap=blues,
        origin="upper",
        extent=extent,
        transform=proj,
    )
    ax0.set_title(tp_title)
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)

    # --- Panel 2: Attribution ---
    im1 = ax1.imshow(
        at_m,
        cmap=reds,
        origin="upper",
        extent=extent,
        transform=proj,
    )
    ax1.set_title(attr_title)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

    # --- Panel 3: TP + cluster contour ---
    ax2.imshow(
        tp_m,
        cmap=blues,
        origin="upper",
        extent=extent,
        transform=proj,
    )

    if cluster_mask is not None:

        cluster_mask = cluster_mask.cpu().numpy().squeeze()

        lon_min, lon_max, lat_min, lat_max = region

        lons = np.linspace(lon_min, lon_max, cluster_mask.shape[1])
        lats = np.linspace(lat_max, lat_min, cluster_mask.shape[0])  # inversion latitude

        Lon, Lat = np.meshgrid(lons, lats)

        ax2.contour(
            Lon,
            Lat,
            cluster_mask,
            levels=[0.5],
            colors="red",
            linewidths=1.5,
            transform=proj,
        )

    ax2.set_title(f"{tp_title} + cluster contour")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[FIG] Saved: {out_path}")


def plot_tp_var_times_attr_contour(
    tp_map: np.ndarray,
    var_maps: list,           # list of 2D arrays [t=7, t=6, t=5]
    var_name: str,
    attr_map: np.ndarray,
    cluster_mask: np.ndarray,
    europe_mask: np.ndarray,
    out_path: str,
    title: str,
    tp_title: str,
    attr_title: str,
    var_titles: list,         # ["var t=7", "var t=6", "var t=5"]
    region=(-12.5, 42.5, 35, 72),
    var_cmap: str = "cividis",
):
    """
    Layout:
      Col 1: TP(t=7)
      Col 2: Variable at t=7,6,5 stacked vertically
      Col 3: Attribution (for variable)
      Col 4: TP(t=7) + contour(top-5% attribution)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)



    # Masks (NaN outside Europe)
    tp_m = _apply_mask(tp_map, europe_mask)
    var_ms = [_apply_mask(v, europe_mask) for v in var_maps]
    at_m = _apply_mask(attr_map, europe_mask)

    blues = _cmap_with_white_bad("Blues")
    reds = _cmap_with_white_bad("Reds")
    varcmap = _cmap_with_white_bad(var_cmap)

    lon_min, lon_max, lat_min, lat_max = region
    extent = [lon_min, lon_max, lat_min, lat_max]
    proj = ccrs.PlateCarree()

    # GridSpec: 3 rows, 4 columns
    # Col 0: tp spans 3 rows
    # Col 1: var t=7 / t=6 / t=5 (3 rows)
    # Col 2: attribution spans 3 rows
    # Col 3: tp+contour spans 3 rows
    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(nrows=3, ncols=4, width_ratios=[1.0, 1.0, 1.0, 1.0], wspace=0.25, hspace=0.15)

    ax_tp   = fig.add_subplot(gs[:, 0], projection=proj)
    ax_v7   = fig.add_subplot(gs[0, 1], projection=proj)
    ax_v6   = fig.add_subplot(gs[1, 1], projection=proj)
    ax_v5   = fig.add_subplot(gs[2, 1], projection=proj)
    ax_attr = fig.add_subplot(gs[:, 2], projection=proj)
    ax_tp_c = fig.add_subplot(gs[:, 3], projection=proj)

    axes = [ax_tp, ax_v7, ax_v6, ax_v5, ax_attr, ax_tp_c]

    # Map features
    for ax in axes:
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
        gl = ax.gridlines(draw_labels=False, linestyle="--", linewidth=0.35)
        gl.right_labels = False
        gl.top_labels = False

    # --- Col 1: TP ---
    im_tp = ax_tp.imshow(tp_m, cmap=blues, origin="upper", extent=extent, transform=proj)
    ax_tp.set_title(tp_title)
    plt.colorbar(im_tp, ax=ax_tp, fraction=0.046, pad=0.02)

    # --- Col 2: Variable times ---
    ims = []
    for ax, vm, vt in zip([ax_v7, ax_v6, ax_v5], var_ms, var_titles):
        imv = ax.imshow(vm, cmap=varcmap, origin="upper", extent=extent, transform=proj)
        ax.set_title(vt)
        ims.append(imv)

    # One shared colorbar for the 3 var maps (using the first one)
    plt.colorbar(ims[0], ax=[ax_v7, ax_v6, ax_v5], fraction=0.046, pad=0.02)

    # --- Col 3: Attribution ---
    im_at = ax_attr.imshow(at_m, cmap=reds, origin="upper", extent=extent, transform=proj)
    ax_attr.set_title(attr_title)
    plt.colorbar(im_at, ax=ax_attr, fraction=0.046, pad=0.02)

    # --- Col 4: TP + contour ---
    ax_tp_c.imshow(tp_m, cmap=blues, origin="upper", extent=extent, transform=proj)

    if cluster_mask is not None:

        cluster_mask = cluster_mask.cpu().numpy().squeeze()

        lons = np.linspace(lon_min, lon_max, cluster_mask.shape[1])
        lats = np.linspace(lat_max, lat_min, cluster_mask.shape[0])

        Lon, Lat = np.meshgrid(lons, lats)

        ax_tp_c.contour(
            Lon,
            Lat,
            cluster_mask,
            levels=[0.5],
            colors="red",
            linewidths=1.5,
            transform=proj,
        )


    ax_tp_c.set_title(f"{tp_title} + cluster contour")

    fig.suptitle(f"{title}\nVariable: {var_name}", y=0.98) 
    plt.tight_layout() 
    plt.savefig(out_path, dpi=200, bbox_inches="tight") 
    plt.close(fig) 
    print(f"[FIG] Saved: {out_path}")

def rel_time_label(t_idx: int, t_ref: int) -> str:
    """
    t_ref = index du temps de référence (ex: t_view=7) qui correspond à 't' (0h).
    t_idx=t_ref   -> 't'
    t_idx=t_ref-1 -> 't-6h'
    t_idx=t_ref-2 -> 't-12h'
    ...
    """
    dh = (t_ref - t_idx) * TIME_STEP_HOURS
    return "t" if dh == 0 else f"t-{dh}h"


if __name__ == "__main__":
    device = get_device()

    os.makedirs(SAVE_IMGS_DIR_PATH, exist_ok=True)

    dataset = load_dataset()
    input_vars = list(dataset.X.coords["channel"].values)
    model = build_model(MODEL_TYPE, C_in=len(input_vars), T=T, device=device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint epoch={ckpt.get('epoch', 'unknown')} from {CKPT_PATH}")

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

    for cluster_id in large_clusters:
        cluster_mask_np = (labeled_map == cluster_id).astype(np.float32)
        cluster_mask = torch.from_numpy(cluster_mask_np).unsqueeze(0).unsqueeze(0).to(device)
        x, y, *_ = dataset[SAMPLE_IDX]
        x=x.unsqueeze(0).to(device).float()
        baseline=torch.zeros_like(x)
        attr = integrated_gradients(model,x,baseline,steps=30,target="region_mean",region_mask=cluster_mask)

        tp_idx, tp_name = find_precip_channel(input_vars)
        tp_in = x[0, T_VIEW, tp_idx].detach().cpu().numpy()  # (H,W)
        europe_mask = np.isfinite(tp_in)

        attr_abs = attr.abs()
        var_importance = attr_abs.sum(dim=(1, 3, 4))[0].detach().cpu().numpy()
        top_idx = np.argsort(-var_importance)[:TOP_K_VARS]

        spatial_all = attr_abs.sum(dim=(1, 2))[0].detach().cpu().numpy()
        plot_tp_attr_contour(tp_in, spatial_all, cluster_mask, europe_mask, os.path.join(SAVE_IMGS_DIR_PATH, f"cluster{cluster_id}", f"ig_ALL_tp_t{T_VIEW}_contour_cluster{cluster_id}.png"),title=f"Sample {SAMPLE_IDX} | ALL vars (sum over T,C) | method=Integrated Gardients | model={MODEL_TYPE}",
            tp_title=f"{tp_name} input (t={T_VIEW})",attr_title=f"Integrated Gradients abs attribution (sum over T,C)")
        
        for rank, c_idx in enumerate(top_idx, start=1):
            var_name = input_vars[c_idx]
            attr_c = attr_abs[0, :, c_idx].sum(dim=0).detach().cpu().numpy()  # (H,W)

            t_idxs = [T_VIEW, T_VIEW - 1, T_VIEW - 2]
            var_t0 = x[0, t_idxs[0], c_idx].detach().cpu().numpy()
            var_t1 = x[0, t_idxs[1], c_idx].detach().cpu().numpy()
            var_t2 = x[0, t_idxs[2], c_idx].detach().cpu().numpy()

            t_ref = T_VIEW
            var_titles = [
                f"{var_name} input ({rel_time_label(t_idxs[0], t_ref)})",
                f"{var_name} input ({rel_time_label(t_idxs[1], t_ref)})",
                f"{var_name} input ({rel_time_label(t_idxs[2], t_ref)})",
            ]

            plot_tp_var_times_attr_contour(
                tp_map=tp_in,
                var_maps=[var_t0, var_t1, var_t2],
                var_name=var_name,
                attr_map=attr_c,
                cluster_mask=cluster_mask,
                europe_mask=europe_mask,
                out_path=os.path.join(SAVE_IMGS_DIR_PATH, f"cluster{cluster_id}", f"ig_top{rank}_{var_name}_tp_t{T_VIEW}.png"),
                title=f"Sample {SAMPLE_IDX} | Top-{rank} variable: {var_name} | method=ig | model={MODEL_TYPE}",
                tp_title=f"{tp_name} input (t)",
                attr_title=f"Inetgrated Gradients abs attribution for {var_name} (sum over T)",
                var_titles=var_titles,
                var_cmap="cividis",
            )