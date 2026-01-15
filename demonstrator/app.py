import os
import glob
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from models.utils.ERA5_dataset_from_local import ERA5Dataset
from models.ConvLSTM.convlstm import PrecipConvLSTM


# -----------------------------
# Utils
# -----------------------------
def plot_triplet(y_true: np.ndarray, y_pred: np.ndarray, title_prefix: str = ""):
    err = np.abs(y_pred - y_true)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im0 = axes[0].imshow(y_true)
    axes[0].set_title(f"{title_prefix}Truth tp_6h")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(y_pred)
    axes[1].set_title(f"{title_prefix}Pred tp_6h")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(err)
    axes[2].set_title(f"{title_prefix}|Error|")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    return fig


@st.cache_resource
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_dataset(dataset_path: str, T: int, lead: int):
    ds = ERA5Dataset(dataset_path, T=T, lead=lead)
    input_vars = list(ds.X.coords["channel"].values)
    return ds, input_vars


@st.cache_resource
def build_model(C_in: int, hidden_channels, kernel_size: int, ckpt_path: str, device):
    model = PrecipConvLSTM(
        input_channels=C_in,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    # supporte soit un dict {"model_state_dict": ...} soit un state_dict direct
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    epoch = ckpt.get("epoch", None) if isinstance(ckpt, dict) else None
    return model, epoch


def main():
    st.set_page_config(page_title="ConvLSTM Precip Demo", layout="wide")
    st.title("ConvLSTM — Démonstrateur précipitations (ERA5 Europe)")

    device = get_device()
    st.sidebar.write(f"**Device:** {device}")

    # -----------------------------
    # Sidebar: config
    # -----------------------------
    st.sidebar.header("Configuration")

    dataset_path = st.sidebar.text_input(
        "Dataset path (.zarr)",
        value="/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_validation.zarr"
    )

    T = st.sidebar.number_input("T (input steps)", min_value=1, max_value=10, value=8, step=1)
    lead = st.sidebar.number_input("Lead steps", min_value=1, max_value=8, value=1, step=1)

    kernel_size = st.sidebar.selectbox("Kernel size", options=[3, 5], index=0)

    hidden_str = st.sidebar.text_input("Hidden channels (comma-separated)", value="32,64")
    hidden_channels = [int(x.strip()) for x in hidden_str.split(",") if x.strip()]

    ckpt_glob = st.sidebar.text_input("Checkpoint glob", value="x_chaos/checkpoints/*.pt")
    ckpt_candidates = sorted(glob.glob(ckpt_glob))
    if len(ckpt_candidates) == 0:
        st.sidebar.warning("No checkpoints found with this glob.")
        ckpt_path = st.sidebar.text_input("Checkpoint path", value="")
    else:
        ckpt_path = st.sidebar.selectbox("Checkpoint", ckpt_candidates, index=min(0, len(ckpt_candidates)-1))
    sample_idx = st.sidebar.number_input("Sample index (dataset)", min_value=0, value=0, step=1)

    run = st.sidebar.button("Run prediction")

    # -----------------------------
    # Load dataset
    # -----------------------------
    try:
        dataset, input_vars = load_dataset(dataset_path, int(T), int(lead))
    except Exception as e:
        st.error(f"Dataset loading failed: {e}")
        return

    C_in = len(input_vars)
    st.sidebar.write(f"**C_in:** {C_in}")

    # -----------------------------
    # Main: info
    # -----------------------------
    st.write("### Inputs")
    st.write(f"- Dataset: `{dataset_path}`")
    st.write(f"- T={T}, lead={lead}, kernel={kernel_size}, hidden={hidden_channels}")
    st.write(f"- Checkpoint: `{ckpt_path}`")

    if not run:
        st.info("Configure settings in the sidebar, then click **Run prediction**.")
        return

    if not ckpt_path or not os.path.exists(ckpt_path):
        st.error("Checkpoint path invalid or not found.")
        return

    # -----------------------------
    # Build model
    # -----------------------------
    try:
        model, epoch = build_model(C_in, hidden_channels, int(kernel_size), ckpt_path, device)
        st.success(f"Model loaded. Checkpoint epoch={epoch if epoch is not None else 'unknown'}")
    except Exception as e:
        st.error(f"Model loading failed (architecture must match checkpoint): {e}")
        return

    # -----------------------------
    # Get one sample (deterministic)
    # -----------------------------
    try:
        X, y = dataset[int(sample_idx)]
        # dataset[idx] retourne (T,C,H,W) et (H,W) souvent -> on ajoute batch dim
        if X.ndim == 4:
            X = X.unsqueeze(0)  # (1,T,C,H,W)
        if y.ndim == 2:
            y = y.unsqueeze(0)  # (1,H,W)
    except Exception as e:
        st.error(f"Sampling failed: {e}")
        return

    X = X.to(device)
    y = y.to(device)

    # -----------------------------
    # Predict
    # -----------------------------
    with torch.no_grad():
        y_hat = model(X).squeeze(1)  # (B,H,W)

    mse = nn.MSELoss()(y_hat, y).item()

    # -----------------------------
    # Display
    # -----------------------------
    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("MSE", f"{mse:.6f}")
        st.write(f"X shape: {tuple(X.shape)}")
        st.write(f"y shape: {tuple(y.shape)}")

    with col2:
        y_true = y[0].detach().cpu().numpy()
        y_pred = y_hat[0].detach().cpu().numpy()
        fig = plot_triplet(y_true, y_pred, title_prefix=f"Sample {sample_idx} - ")
        st.pyplot(fig, clear_figure=True)


if __name__ == "__main__":
    main()
