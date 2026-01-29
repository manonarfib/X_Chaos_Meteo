import torch
import torch.nn as nn
import shap
import numpy as np
import os

from models.utils.ERA5_dataset_from_local import  ERA5Dataset
from models.ConvLSTM.convlstm import PrecipConvLSTM

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

    model = PrecipConvLSTM(
        input_channels=C_in,
        hidden_channels=[32, 64],
        kernel_size=3,
    ).to(device)

    ckpt_path = "explainability/epoch3_full.pt"
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

    background_size = 10
    background = []

    for k in range(background_size):
        X_bg, *_ = dataset_train[k]
        background.append(X_bg)

    background = torch.stack(background).to(device).float()

    print("Fitting shap with background")

    explainer = shap.DeepExplainer(pixel_model,background)

    shap_values = explainer.shap_values(X)

    print("SHAP values shape:", shap_values.shape)

    out_dir = "shap_maps"
    os.makedirs(out_dir, exist_ok=True)

    shap_np = shap_values[0].detach().cpu().numpy()  # (T, C, H, W)

    np.save(os.path.join(out_dir, "shap_values.npy"), shap_np)