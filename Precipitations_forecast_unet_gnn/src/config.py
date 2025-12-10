from pathlib import Path


# Data
ERA5_ZARR_PATH = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
YEARS = [2000]
REGION = {
"lat_slice": (75, 35),
"lon_slices": [(347.5, 360), (0, 42.5)]
}
INPUT_VARS = ["t2m", "z500", "u10", "v10", "q"]
TARGET_VAR = "total_precipitation_6hr"


# Training
BATCH_SIZE = 4
NUM_WORKERS = 4
LR = 1e-4
EPOCHS = 50
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"


# Dask chunking
CHUNKS = {"time": 48, "latitude": 128, "longitude": 128}


# UNet + MPNN hyperparameters
UNET_FEATURES = [64, 128, 256, 512]
MPNN_HIDDEN = 64
CONTEXT_HOURS = 48 # nb d'heures passées utilisées comme input