import os
import glob
import base64
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import plotly.express as px

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.utils.ERA5_dataset_from_local import ERA5Dataset
from models.ConvLSTM.convlstm import PrecipConvLSTM

# =====================================================
# Utils
# =====================================================

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
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    epoch = ckpt.get("epoch", None) if isinstance(ckpt, dict) else None
    return model, epoch


def run_inference(model, dataset, sample_idx, device):
    X, y, _ = dataset[sample_idx]
    with torch.no_grad():
        X = X.unsqueeze(0).to(device)
        y_hat = model(X)

    y_pred = y_hat.squeeze().detach().cpu().numpy()
    y_true = y.squeeze().detach().cpu().numpy()
    return y_true, y_pred


# =====================================================
# Plotly map (zoom + hover pixel)
# =====================================================

def plot_interactive_map_geo(arr: np.ndarray, title: str,
                             lon_min=-12.5, lon_max=42.5,
                             lat_min=35, lat_max=72):

    H, W = arr.shape

    lons = np.linspace(lon_min, lon_max, W)
    lats = np.linspace(lat_min, lat_max, H)

    fig = px.imshow(
        arr,
        x=lons,
        y=lats,
        origin="lower",
        aspect="auto",
        title=title,
        labels=dict(x="Longitude", y="Latitude", color="Value")
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        dragmode="zoom",
    )

    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig

import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from matplotlib import cm


def array_to_png_overlay(arr, vmin=None, vmax=None, cmap_name="Blues"):
    arr = np.array(arr)

    if vmin is None:
        vmin = np.nanmin(arr)
    if vmax is None:
        vmax = np.nanmax(arr)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    rgba = cmap(norm(arr))  # (H,W,4)
    rgba[..., 3] = np.where(np.isfinite(arr), 0.85, 0.0)  # alpha mask

    return (rgba * 255).astype(np.uint8)

def plot_folium_raster(arr, title,
                       region=(-12.5, 42.5, 35, 72),
                       cmap="Blues"):

    lon_min, lon_max, lat_min, lat_max = region

    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles="OpenStreetMap"
    )

    img = array_to_png_overlay(arr, cmap_name=cmap)

    folium.raster_layers.ImageOverlay(
        image=img,
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        opacity=1.0,
        interactive=True,
        cross_origin=False,
    ).add_to(m)

    folium.LayerControl().add_to(m)

    return m

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm
import numpy as np
import streamlit as st

def plot_clean_map(arr: np.ndarray, title: str,
                   lon_min=-12.5, lon_max=42.5,
                   lat_min=35, lat_max=72,
                   cmap="Blues",
                   figsize=(10, 8)):  # <-- ajouter figsize ici
    
    H, W = arr.shape
    lons = np.linspace(lon_min, lon_max, W)
    lats = np.linspace(lat_min, lat_max, H)

    fig = plt.figure(figsize=figsize)  # <-- utiliser figsize ici
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Ajouter fond de carte simplifié
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')

    # Afficher la donnée
    img = ax.imshow(arr, origin='lower', extent=[lon_min, lon_max, lat_min, lat_max],
                    cmap=cmap, alpha=0.8, aspect='auto')

    # Colorbar
    plt.colorbar(img, ax=ax, orientation='vertical', label='Valeur')

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    return fig



# =====================================================
# Background helpers
# =====================================================

@st.cache_resource
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
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
    
def clear_page_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: none;
            background-color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# Splash screen
# =====================================================

def splash_screen():
    app_dir = os.path.dirname(__file__)
    img_path = os.path.join(app_dir, "assets", "bg_accueil.png")
    if os.path.exists(img_path):
        set_png_as_page_bg(img_path)

    if "splash_done" not in st.session_state:
        st.session_state.splash_done = False

    if not st.session_state.splash_done:
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
                <h1 style='font-size:60px;text-align:center;'>Prévisions de Précipitations — XChaos Météo</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns([1, 0.2, 1])
        with col2:
            if st.button("Start", key="start_button"):
                st.session_state.splash_done = True
                st.rerun()

        return False

    return True


# =====================================================
# Pages
# =====================================================

def page_home():
    st.title("Bienvenue — Démonstrateur Prévision Météo")
    st.markdown(
        """
        ## Objectif du projet
        Démonstrateur de modèles deep learning pour la prévision de précipitations à partir de données ERA5.

        ## Fonctionnalités
        - Choix du modèle
        - Choix des paramètres temporels
        - Inférence à la demande
        - Visualisation interactive avec zoom
        - Inspection des pixels par survol

        ## Modèles
        - ConvLSTM
        - (UNet — à intégrer)

        ## Auteurs
        Louisa Arfib — Manon Arfib — Nathan Morin
        """
    )



def page_inference():
    st.title("Page Inférence — Visualisation interactive")

    device = get_device()
    st.write(f"Device: {device}")

    if "has_prediction" not in st.session_state:
        st.session_state.has_prediction = False

    # ---------------- Dataset selection ----------------
    st.header("Sélection des données et modèle")
    
    dataset_choice = st.selectbox(
        "Dataset",
        ["train", "validation", "test"]
    )

    # Exemple : adapter le chemin selon ton choix
    base_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/"
    dataset_files = {
        "train": "era5_europe_ml_train.zarr",
        "validation": "era5_europe_ml_validation.zarr",
        "test": "era5_europe_ml_test.zarr"
    }
    dataset_path = os.path.join(base_path, dataset_files[dataset_choice])
    st.write(f"Dataset sélectionné : {dataset_path}")

    # ---------------- Checkpoint selection ----------------
    st.write("Choisir le modèle :")
    ckpt_choice = st.radio(
        "Checkpoint",
        ["convlstm", "unet"]
    )

    ckpt_paths = {
        "convlstm": "epoch3_full.pt",
        "unet": "/mounts/models/unet_best.pt"
    }
    ckpt_path = ckpt_paths[ckpt_choice]
    st.write(f"Checkpoint utilisé : {ckpt_path}")

    # ---------------- Sample selection ----------------
    sample_idx = st.number_input("Sample index", 0, 100000, 0)

    run = st.button("Run inference")

    if run:
        st.success("Inférence en cours...")

        try:
            with st.spinner("Chargement du dataset..."):
                # Pour simplifier, on met des valeurs fixes
                T = 8
                lead = 1
                dataset, input_vars = load_dataset(dataset_path, T, lead)
                C_in = len(input_vars)

            with st.spinner("Chargement du modèle..."):
                # Pour simplifier, paramètres fixes du modèle
                if ckpt_choice == "convlstm":
                    hidden_channels = [32, 64]
                    kernel_size = 3
                    model_type = "convlstm"
                else:
                    # placeholder si UNet
                    hidden_channels = [64, 128]
                    kernel_size = 3
                    model_type = "unet"

                model, epoch = build_model(
                    C_in,
                    hidden_channels,
                    kernel_size,
                    ckpt_path,
                    device
                )

            with st.spinner("Exécution de l'inférence..."):
                y_true, y_pred = run_inference(model, dataset, sample_idx, device)

        except Exception as e:
            st.error(f"Erreur lors de l'inférence : {e}")
            return

        # S'assurer que c'est 2D
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)

        if y_true.ndim != 2 or y_pred.ndim != 2:
            st.error(f"Sortie non 2D — formes {y_true.shape} / {y_pred.shape}")
            return

        st.session_state.y_true = y_true
        st.session_state.y_pred = y_pred
        st.session_state.has_prediction = True

    # ---------------- Display ----------------
    if not st.session_state.has_prediction:
        st.info("Clique sur 'Run inference' pour générer une prédiction.")
        return

    y_true = st.session_state.y_true
    y_pred = st.session_state.y_pred
    err = np.abs(y_pred - y_true)

    # ---------------- Interactive maps ----------------
    tab1, tab2, tab3 = st.tabs(["Prediction", "Truth", "Error"])

    with tab1:
        st.pyplot(plot_clean_map(y_pred, "Prediction", cmap="Blues", lon_min=-12.5, lon_max=42.5,
                                 lat_min=35, lat_max=72, figsize=(6,5)))
    with tab2:
        st.pyplot(plot_clean_map(y_true, "Truth", cmap="Blues", lon_min=-12.5, lon_max=42.5,
                                 lat_min=35, lat_max=72, figsize=(6,5)))
    with tab3:
        st.pyplot(plot_clean_map(err, "Error", cmap="Reds", lon_min=-12.5, lon_max=42.5,
                                 lat_min=35, lat_max=72, figsize=(6,5)))



# =====================================================
# Main
# =====================================================

def main():
    st.set_page_config(page_title="XChaos Météo Demo", layout="wide")

    if not splash_screen():
        st.stop()

    clear_page_bg()
    
    page = st.sidebar.selectbox(
        "Navigation",
        ["Accueil", "Inférence"]
    )

    if page == "Accueil":
        page_home()
    else:
        page_inference()


if __name__ == "__main__":
    main()
