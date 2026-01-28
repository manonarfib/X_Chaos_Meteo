import os
import glob
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

import base64
@st.cache_resource
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    /* Full-page background */
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# -----------------------------
# Splash screen
# -----------------------------
def splash_screen():
    # Set background
    app_dir = os.path.dirname(__file__)
    img_path = os.path.join(app_dir, "assets", "bg_accueil.png")
    if not os.path.exists(img_path):
        st.error(f"Background image not found: {img_path}")
        return False
    set_png_as_page_bg(img_path)

    # Initialize session state
    if "splash_done" not in st.session_state:
        st.session_state.splash_done = False

    if not st.session_state.splash_done:
        # Centered text + button
        st.markdown(
            """
            <div style="
                display:flex;
                flex-direction:column;
                justify-content:center;
                align-items:center;
                height:50vh;
                color:white;
                font-family: Arial, sans-serif;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
            ">
                <h1 style='font-size:60px;text-align:center;'>Prévisions de Précipitations - XChaos Météo</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Use Streamlit button (click detection works)
        col1, col2, col3 = st.columns([1, 0.2, 1])
        with col2:
            if st.button("Start", key="start_button"):
                st.session_state.splash_done = True
                # st.runtime.legacy_caching.rerun()  # Force refresh to show next page immediately
                return True

    return False


def main():
    st.set_page_config(page_title="ConvLSTM Precip Demo", layout="wide")

    # -----------------------------
    # Initialisation du state
    # -----------------------------
    # Initialisation du state
    if "splash_done" not in st.session_state:
        st.session_state.splash_done = False

    # Affiche le splash tant que le bouton Start n'a pas été cliqué
    if not st.session_state.splash_done:
        splash_screen()
        st.stop()  # stoppe l'exécution ici tant que Start n'est pas cliqué



    # -----------------------------
    # Menu principal
    # -----------------------------
    page = st.sidebar.selectbox("Choisir une page", ["Accueil", "Démo ConvLSTM"])

    # -----------------------------
    # Page d'accueil accessible via menu
    # -----------------------------
    if page == "Accueil":
        st.title("Bienvenue dans le démonstrateur XChaos Météo")
        st.markdown("""
        Ce démonstrateur permet de visualiser les prévisions de précipitations
        sur l'Europe.

        Deux modèles ont été entraînés : un ConvLSTM et un UNet.  
        Les prédictions des deux réseaux sont disponibles. Des résultats comparatifs
        des 2 réseaux sont présentés dans la suite du démonstrateur.

        **Instructions :**
        - Sélectionnez une démo dans le menu pour voir les résultats.

        **Auteurs :**
        Louisa Arfib, Manon Arfib et Nathan Morin - élèves à CentraleSupélec
        """)
        return  # pas de sidebar sur la page d'accueil, juste info

    # -----------------------------
    # Démo ConvLSTM
    # -----------------------------
    st.title("ConvLSTM — Démonstrateur précipitations (ERA5 Europe)")

    device = get_device()
    st.sidebar.write(f"**Device:** {device}")
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

    st.write("### Inputs")
    st.write(f"- Dataset: `{dataset_path}`")
    st.write(f"- T={T}, lead={lead}, kernel={kernel_size}, hidden={hidden_channels}")
    st.write(f"- Checkpoint: `{ckpt_path}`")

    if not run:
        st.info("Configurez les paramètres dans la sidebar, puis cliquez sur **Run prediction**.")
        return




if __name__ == "__main__":
    main()
