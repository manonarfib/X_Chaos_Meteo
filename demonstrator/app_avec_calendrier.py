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
    ds = ERA5Dataset(dataset_path, T=T, max_lead=lead)
    input_vars = list(ds.X.coords["channel"].values)

    times = ds.X.coords["time"].values
    times = np.array(times).astype("datetime64[s]").astype(object)

    return ds, input_vars, times


@st.cache_resource
def build_model(C_in: int, hidden_channels, kernel_size: int, ckpt_path: str, device, output_size=1):
    model = PrecipConvLSTM(
        input_channels=C_in,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        output_size=output_size
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    epoch = ckpt.get("epoch", None) if isinstance(ckpt, dict) else None
    return model, epoch


def run_inference(model, dataset, sample_idx, device, lead):
    X, y, _ = dataset[sample_idx]
    with torch.no_grad():
        X = X.unsqueeze(0).to(device)
        y_hat = model(X)
    y_hat = torch.clamp(y_hat, min=0.0)
    y_pred = y_hat.squeeze().detach().cpu().numpy()
    y_true = y.squeeze().detach().cpu().numpy()
    return y_true, y_pred

def extract_available_datetimes(times):
    # convertir en datetime64 lisible (ns → python datetime)
    datetimes = np.array(times).astype('datetime64[s]').astype(object)

    return datetimes

def get_index_from_datetime(times, selected_datetime):
    times_sec = np.array(times).astype('datetime64[s]')
    selected = np.datetime64(selected_datetime)

    idx = np.where(times_sec == selected)[0]

    if len(idx) == 0:
        return None
    return int(idx[0])

# def build_target_index(times, T, lead_hours):
#     times = np.array(times).astype('datetime64[h]')
#     py_datetimes = times.astype(object)

#     target_to_index = {}

#     for t0_idx in range(len(times)):

#         t0 = times[t0_idx]

#         # vérifier qu'on a assez de données pour l'input
#         if t0_idx + T + lead_hours -1 > len(times):
#             break

#         target_time = t0 + np.timedelta64((T+lead_hours-1)*6, 'h')

#         # trouver cet instant dans le dataset
#         matches = np.where(times == target_time)[0]

#         if len(matches) == 0:
#             continue

#         target_idx = matches[0]
#         target_dt = py_datetimes[target_idx]

#         target_to_index[target_dt] = t0_idx

#     return target_to_index

def build_target_index(times, T, lead):
    target_to_index = {}

    max_start = len(times) - T - lead + 1

    for t0 in range(max_start):
        first_pred_time = times[t0 + T]  # ✅ début des prédictions
        target_to_index[first_pred_time] = t0

    return target_to_index

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
                   figsize=(10, 8),
                   vmin=None,
                   vmax=None):

    H, W = arr.shape
    lons = np.linspace(lon_min, lon_max, W)
    lats = np.linspace(lat_min, lat_max, H)

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')

    img = ax.imshow(
        arr,
        origin='lower',
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap=cmap,
        alpha=0.8,
        aspect='auto',
        vmin=vmin,
        vmax=vmax
    )
    
    fig.colorbar(img, ax=ax, orientation='vertical', label='Précipitations (mm)')
    ax.set_title(title)

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
    st.title("Démonstrateur du Projet XChaos : Explicabilité d'un Système Chaotique - Prévisions Météorologiques 🌧️")
    
    # Bannière centrale (GIF)
    # st.markdown(
    #     """
    #     <div style="text-align:center;">
    #         <img src="era5_visuals/figures/gifs/alex_europe.gif" width="500">
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )

    st.markdown("---")

    # Section Aperçu
    st.header("💡 Aperçu")
    st.markdown(
        """
        Ce démonstrateur vous permet d'explorer des modèles de deep learning pour la prévision des précipitations en Europe
        sur un horizon de 6 heures à partir des données ERA5.  
        Vous pouvez également analyser **l'explicabilité des modèles**.
        """
    )

    # Section Fonctionnalités avec colonnes
    st.header("⚙️ Fonctionnalités")
    col1, col2 = st.columns(2)
    col1.markdown("""
    - Choix du modèle (ConvLSTM / U-Net)  
    - Sélection des paramètres temporels  
    - Lancer l'inférence à la demande
    """)
    col2.markdown("""
    - Zoom et survol des pixels  (TO DO)
    - Visualisation de l'influence des variables d'entrée  (TO DO)
    - Autre explicabilité (TO DO)
    """)
 

    st.markdown("---")

    # Section Modèles
    st.header("⚙️ Modèles")
    st.subheader("ConvLSTM")
    st.markdown(
        """
        Le réseau **ConvLSTM** est particulièrement adapté à la prévision spatio-temporelle.  
        Il combine des convolutions spatiales avec des cellules LSTM pour capturer à la fois les dépendances 
        temporelles et spatiales des précipitations.
        """
    )

    st.subheader("U-Net 3D")
    st.markdown(
        """
        Le réseau **U-Net 3D** implémenté est une architecture classique en U adaptée au cas de la prévision des précipitations :  
        - Toutes les variables météorologiques passées (33 canaux) sont traitées simultanément.  
        - Les skip connections conservent l'information temporelle complète, contrairement aux travaux de Kaparakis et al.  
        - La dernière couche est une convolution 3D qui réduit la dimension temporelle pour prédire un pas unique (t+6h).  

        Les convolutions utilisées sont des **Conv3D** avec un padding qui conserve la taille d'entrée. 
        """
    )

    st.markdown("---")

    # Section Auteurs et remerciements
    st.header("🤝 Auteurs & Remerciements")
    st.markdown(
        """
        **Louisa Arfib** — [GitHub](https://github.com/arfiblouisa)  
        **Manon Arfib** — [GitHub](https://github.com/manonarfib)  
        **Nathan Morin** — [GitHub](https://github.com/Nathan9842)  

        Un grand merci à **Florestan Fontaine** de HeadMind Partners pour ses conseils et son accompagnement.
        """
    )
    
    # Bouton pour aller directement à la page d'inférence
    if st.button("▶ Aller à l'inférence"):
        st.session_state.page = "Inférence"
        st.rerun()


def page_inference():
    st.title("Inférence : Prévisions de précipitations")

    device = get_device()
    # st.write(f"Device: {device}")

    if "has_prediction" not in st.session_state:
        st.session_state.has_prediction = False

    # ---------------- Dataset selection ----------------
    st.header("Sélection des données et du modèle")
    
    st.markdown(
        """
        Seules environ deux semaines du dataset de test ont été chargées sur le git, les données disponibles pour l'inférence correspondent donc seulement aux données 
        disponibles (du 01/01/2020 au 16/01/2020). Si d'autres données sont téléchargées avec le fichier "download_dataset_from_gcs/download_dataset.py",
        elles peuvent être utilisées. Il suffit alors de modifier le chemin d'accès aux données en sélectionnant 'mon dataset' ci-dessous.
        
        48 heures avant de données avant la date d'inférence souhaitée sont nécessaires.
        """)
    
    dataset_choice = st.selectbox(
        "Dataset :",
        [
            "16 jours du dataset de test (du 01/01/2020 au 16/01/2020)",
            "mon dataset"
        ]
    )

    dataset_files = {
        "16 jours du dataset de test (du 01/01/2020 au 16/01/2020)": "demonstrator/era5_europe_ml_test_2_weeks.zarr",
    }

    # ---------------- Choix du chemin ----------------
    if dataset_choice == "mon dataset":
        dataset_path = st.text_input(
            "Chemin vers le dataset (.zarr)",
            placeholder="/path/to/your/dataset.zarr"
        )

        if dataset_path == "":
            st.warning("Veuillez entrer un chemin valide")
            st.stop()

        # Option sécurité (très recommandé)
        if not os.path.exists(dataset_path):
            st.error("Le chemin n'existe pas")
            st.stop()

    else:
        dataset_path = dataset_files[dataset_choice]

    st.write(f"Dataset sélectionné : {dataset_path}")
    # dataset_path = "/usr/users/x_chaos_meteo/arfib_lou/X_Chaos_Meteo/demonstrator/era5_europe_ml_test_2_weeks.zarr"
    # st.write(f"Dataset sélectionné : {dataset_path}")
    
    st.subheader("Paramètres de prédiction")
    lead = st.radio(
        "Nombre de pas de temps prédits :",
        [1, 8],
        format_func=lambda x: f"{x} pas de temps"
    )

    # ---------------- Model selection ----------------
    
    if lead == 1:
        ckpt_choice = st.radio("Modèle :", ["ConvLSTM", "UNet"])
    else:
        st.info("Pour lead_time = 8, seul ConvLSTM est disponible")
        ckpt_choice = "ConvLSTM"
    
    # ckpt_choice = st.radio("Modèle", ["convlstm", "unet"])

    ckpt_paths = {
        ("ConvLSTM", 1): "checkpoints/convlstm/mse/epoch3_full.pt",
        ("ConvLSTM", 8): "checkpoints/convlstm/mse_multi/epoch3_full.pt",
        ("UNet", 1): "/mounts/models/unet_best.pt"
    }
    
    ckpt_path = ckpt_paths[(ckpt_choice, lead)]
    # st.write(f"Checkpoint utilisé : {ckpt_path}")

    # ---------------- Load dataset (UNE FOIS) ----------------
    T = 8
    # lead = 1
    
    dataset, input_vars, times = load_dataset(dataset_path, T, lead)
    C_in = len(input_vars)

    # ---------------- Datetime selection ----------------
    st.subheader("Sélection de la date à laquelle prédire les précipitations :")

    target_to_index = build_target_index(times, T, lead)

    # dates disponibles (target)
    available_targets = sorted(target_to_index.keys())

    # construire mapping date → datetimes
    date_to_targets = {}
    for dt in available_targets:
        date = dt.date()
        date_to_targets.setdefault(date, []).append(dt)

    available_dates = sorted(date_to_targets.keys())

    # 📅 calendrier
    selected_date = st.date_input(
        "Choisir une date cible :",
        min_value=min(available_dates),
        max_value=max(available_dates),
        value=min(available_dates)
    )

    if selected_date not in date_to_targets:
        st.warning("Aucune prédiction possible pour cette date.")
        return

    # 🕒 heures disponibles (targets)
    available_datetimes = date_to_targets[selected_date]

    selected_dt = st.selectbox(
        "Choisir l'heure cible :",
        available_datetimes,
        format_func=lambda x: x.strftime("%H:%M")
    )

    # 🎯 récupérer t0
    sample_idx = target_to_index[selected_dt]

    st.write(f"Début de la prédiction : {selected_dt}")
    # st.write(f"Index utilisé (t0) : {sample_idx}")

    # ---------------- Run button ----------------
    run = st.button("Lancer l'inférence")

    if run:
        # st.success("Inférence en cours...")

        try:
            with st.spinner("Chargement du modèle..."):

                if ckpt_choice == "ConvLSTM":
                    hidden_channels = [32, 64]
                    kernel_size = 3
                else:
                    hidden_channels = [64, 128]
                    kernel_size = 3

                model, epoch = build_model(
                    C_in,
                    hidden_channels,
                    kernel_size,
                    ckpt_path,
                    device,
                    output_size=lead
                )

            with st.spinner("Exécution de l'inférence..."):
                y_true, y_pred = run_inference(model, dataset, sample_idx, device, lead)

        except Exception as e:
            st.error(f"Erreur lors de l'inférence : {e}")
            return

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # 🔥 garantir (lead, H, W)
        if y_pred.ndim == 2:
            y_pred = y_pred[None, ...]
            y_true = y_true[None, ...]

        elif y_pred.ndim == 3:
            pass  # déjà bon

        else:
            st.error(f"Shape inattendue : {y_pred.shape}")
            return
        
        st.session_state.sample_idx = sample_idx
        st.session_state.selected_dt = selected_dt

        # if y_true.ndim != 2 or y_pred.ndim != 2:
        #     st.error(f"Sortie non 2D — formes {y_true.shape} / {y_pred.shape}")
        #     return
        
        st.session_state.y_true_seq = y_true
        st.session_state.y_pred_seq = y_pred
        st.session_state.has_prediction = True

    # ---------------- Display ----------------
    if not st.session_state.get("has_prediction", False):
        st.info("Clique sur 'Run inference' pour générer une prédiction.")
        return

    y_true_seq = st.session_state.y_true_seq
    y_pred_seq = st.session_state.y_pred_seq
    sample_idx = st.session_state.sample_idx
    selected_dt = st.session_state.selected_dt

    lead = y_pred_seq.shape[0]
    
    if lead>1:
        # intervalle complet
        end_time = times[sample_idx + T + lead - 1]
        st.write(f"Intervalle prédit : {selected_dt} → {end_time}")

    # 🎯 slider temporel (toujours présent)
    if lead > 1:
        step = st.slider(
            "Pas de prédiction",
            min_value=0,
            max_value=lead - 1,
            value=0
        )
    else:
        step = 0
    
    y_pred_step = y_pred_seq[step]
    y_true_step = y_true_seq[step]

    err = np.abs(y_pred_step - y_true_step)

    # temps réel affiché
    current_time = times[sample_idx + T + step]

    if lead>1:
        st.write(f"Temps affiché : {current_time}")

    tab1, tab2, tab3 = st.tabs(["Prediction", "Truth", "Error"])
    vmin_true = np.min(y_true_seq)
    vmax_true = np.max(y_true_seq)
    vmin_pred = np.min(y_pred_seq)
    vmax_pred = np.max(y_pred_seq)

    with tab1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.pyplot(plot_clean_map(y_pred_step, "Prédiction", figsize=(5,4), vmin=vmin_pred, vmax=vmax_pred))

    with tab2:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.pyplot(plot_clean_map(y_true_step, "Truth", figsize=(5,4), vmin=vmin_true, vmax=vmax_true))

    with tab3:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.pyplot(plot_clean_map(err, "Error", cmap="Reds", figsize=(5,4), vmin=vmin_true, vmax=vmax_true))

# =====================================================
# Main
# =====================================================

def main():
    st.set_page_config(page_title="XChaos Météo Demo", layout="wide")

    if not splash_screen():
        st.stop()

    clear_page_bg()

    # récupérer page depuis session_state ou fallback sur sidebar
    if "page" not in st.session_state:
        st.session_state.page = "Accueil"

    # Affichage du sidebar
    page_sidebar = st.sidebar.selectbox(
        "Navigation",
        ["Accueil", "Inférence"],
        index=0 if st.session_state.page == "Accueil" else 1
    )

    # mettre à jour session_state si sidebar change
    st.session_state.page = page_sidebar

    # afficher la page
    if st.session_state.page == "Accueil":
        page_home()
    else:
        page_inference()


if __name__ == "__main__":
    main()
