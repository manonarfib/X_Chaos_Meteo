import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

from models.utils.ERA5_dataset_from_local import  ERA5Dataset
from models.ConvLSTM.convlstm import PrecipConvLSTM
from models.unet.model_without_collapse import WFUNet_with_train

class PixelModel(nn.Module):
    def __init__(self, base_model, i, j):
        super().__init__()
        self.base_model = base_model
        self.i = i
        self.j = j

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        return: (B,)
        """
        y = self.base_model(x)          # (B, 1, H, W)
        return y[:, 0, self.i, self.j].unsqueeze(1)  # scalaire par batch

def build_model(model_type: str, C_in: int, device: torch.device) -> torch.nn.Module:
    model_type = model_type.lower().strip()
    if model_type == "convlstm":
        model = PrecipConvLSTM(
            input_channels=C_in,
            hidden_channels=[32, 64],
            kernel_size=3
        ).to(device)
        return model
    elif model_type == "unet":
        # signature in your commented line: WFUNet_with_train(8,149,221,33,1, 8,32,0)
        # If you change T/H/W/C_in etc., update these args accordingly.
        model = WFUNet_with_train(T, 149, 221, C_in, 1, 8, 32, 0).to(device)
        return model
    else:
        raise ValueError(f"Unknown MODEL_TYPE='{model_type}'. Use 'convlstm' or 'unet'.")   

def integrated_gradients(model, x, baseline, steps=20):
    """
    model: PixelModel
    x: (1, T, C, H, W)
    baseline: same shape
    """
    model.eval()
    grads = torch.zeros_like(x)

    for alpha in torch.linspace(0, 1, steps):
        x_interp = baseline + alpha * (x - baseline)
        x_interp.requires_grad_(True)

        out = model(x_interp)          # scalaire
        out.backward()

        grads += x_interp.grad.detach()
        model.zero_grad()

    avg_grads = grads / steps
    ig = (x - baseline) * avg_grads
    return ig


def create_ig_plot_1var_1t(ig_np, t,c,i,j, out_dir_all_plots, var):
    plt.figure(figsize=(6,5))
    plt.imshow(ig_np[t, c], cmap="RdBu", vmin=-np.max(np.abs(ig_np[t, c])),
               vmax=np.max(np.abs(ig_np[t, c])))
    plt.colorbar(label="Integrated Gradient Contribution")
    plt.title(f"Pixel ({i},{j}) - {var} - t={t}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_all_plots, f"ig_{var}_t{t}_pixel_{i}_{j}.png"), dpi=150)
    plt.close()


def create_ig_plots_all_vars_1plot(C, input_vars, t, ig_np, i, j, out_dir):
    # Définir grid automatiquement : 2 colonnes
    ncols = 2
    nrows = (C + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))
    axes = axes.flatten()

    for c, var in enumerate(input_vars):
        im = axes[c].imshow(
            ig_np[t, c],
            cmap="RdBu",
            vmin=-np.max(np.abs(ig_np[t, c])),
            vmax=np.max(np.abs(ig_np[t, c]))
        )
        axes[c].set_title(var)
        axes[c].axis("off")

    # Supprimer axes vides
    for ax in axes[C:]:
        ax.axis("off")

    # Colorbar globale
    fig.colorbar(im, ax=axes[:C], fraction=0.02, pad=0.04, label="IG contribution")

    plt.suptitle(f"Integrated Gradients - Input timestep {t}", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])

    # Sauvegarde
    fname = os.path.join(out_dir, f"ig_grid_t{t}_pixel_{i}_{j}.png")
    plt.savefig(fname, dpi=150)
    plt.close()

def create_ig_plot_aggregate_vars_1_t(importance_maps_per_time, t, out_dir, i, j):
    plt.figure(figsize=(6,5))
    plt.imshow(importance_maps_per_time[t], cmap="Reds", vmin=0,
               vmax=np.max(importance_maps_per_time[t]))
    plt.colorbar(label="Integrated Gradient Contribution")
    plt.title(f"Pixel ({i},{j}) - t={t}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"ig_aggregate_t{t}_pixel_{i}_{j}.png"), dpi=150)
    plt.close()

def create_ig_plot_fully_aggregated(importance_map_fully_aggregated, out_dir):
    plt.figure(figsize=(6,5))
    plt.imshow(importance_map_fully_aggregated, cmap="Reds", vmin=0,
               vmax=np.max(importance_map_fully_aggregated))
    plt.colorbar(label="Integrated Gradient Contribution")
    plt.title(f"Pixel ({i},{j})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"ig_aggregate_fully_pixel_{i}_{j}.png"), dpi=150)
    plt.close()

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device=torch.device("cpu")
    print("Device:", device)


    dataset_path_train = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_train.zarr"
    T, lead = 8, 1
    batch_size = 8

    dataset_train = ERA5Dataset(dataset_path_train, T=T, lead=lead)


    dataset_path_test = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_test.zarr"
    T, lead = 8, 1
    batch_size = 8

    dataset_test = ERA5Dataset(dataset_path_test, T=T, lead=lead)    
    input_vars = list(dataset_test.X.coords["channel"].values)
    C_in = len(input_vars)

    MODEL_TYPE = "convlstm"#"unet" 
    model = build_model(MODEL_TYPE, C_in, device)

    ckpt_path = "checkpoints/convlstm/mse/epoch3_full.pt"#"checkpoints/run_144949/best_checkpoint_epoch5_batch_idx6479.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint epoch={ckpt.get('epoch', 'unknown')} from {ckpt_path}")

    sample_idx = 50
    X, y, *_ = dataset_test[sample_idx]

    X = X.unsqueeze(0).to(device).float()  # ajouter dimension batch
    y = y.unsqueeze(0).to(device).float()

    with torch.no_grad():
        y_hat = model(X).squeeze(1)
        i, j = torch.unravel_index(
        torch.argmax(y_hat[0]),
        y_hat.shape[1:]
    )
        
    print("i, j: ", i, j)
    
    # i, j = 40, 60   # pixel géographique précis
    pixel_model = PixelModel(model, i, j).to(device)
    pixel_model.eval()

    X_cpu = X.cpu()

    baseline = torch.zeros_like(X_cpu)  # zéro ou climatologie moyenne

    ig = integrated_gradients(pixel_model, X_cpu, baseline, steps=50)
    print("Integrated Gradients computed, shape:", ig.shape)

    out_dir = "explainability/ig_maps/ConvLSTM"
    os.makedirs(out_dir, exist_ok=True)

    ig_np = ig[0].numpy()  # (T, C, H, W)

    T, C, H, W = ig_np.shape

    out_dir_all_plots=os.path.join(out_dir, 'all_variables')
    os.makedirs(out_dir_all_plots, exist_ok=True)
    for t in range(T):
        for c, var in enumerate(input_vars):
            create_ig_plot_1var_1t(ig_np, t,c,i,j, out_dir_all_plots, var)

    for t in range(T):
        create_ig_plots_all_vars_1plot(C, input_vars, t, ig_np, i, j, out_dir)

    #Aggregate on variables (one plot per time step)
    importance_maps_per_time = np.abs(ig_np).sum(axis=1)
    out_dir_aggregate=os.path.join(out_dir, 'aggregate')
    os.makedirs(out_dir_aggregate, exist_ok=True)
    for t in range(T):
        create_ig_plot_aggregate_vars_1_t(importance_maps_per_time, t, out_dir_aggregate, i, j)

    importance_map_fully_aggregated = importance_maps_per_time.sum(axis=0)
    create_ig_plot_fully_aggregated(importance_map_fully_aggregated, out_dir_aggregate)

    print(f"Saved IG grids in {out_dir}")