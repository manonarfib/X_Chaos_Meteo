import os
import pickle
import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
# from weatherbench2.regions import SliceRegion
from dask.diagnostics import ProgressBar
import dask

# ============================================================
# 1. LOAD DATA DIRECTLY FROM GOOGLE CLOUD (lazy)
# ============================================================
path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
ds = xr.open_zarr(path, consolidated=True)
print("Dataset opened (lazy)")

# Select years
years = ["2000"]
ds = ds.sel(time=ds.time.dt.year.isin([int(y) for y in years]))

# ============================================================
# 2. Define Europe region (lazy slicing)
# ============================================================

# tp6 = ds["total_precipitation_6hr"].sel(time=date, method="nearest", tolerance="3H")
ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude")
ds_europe = ds.sel(longitude=slice(-12.5, 42.5), latitude=slice(72, 35))

# europe_region = SliceRegion(
#     lat_slice=slice(72, 35),
#     lon_slice=[slice(347.5, 360), slice(0, 42.5)]
# )

# ds_europe, weights = europe_region.apply(ds, xr.ones_like(ds.isel(time=0)))
print("Region extracted (still lazy)")

# ============================================================
# 3. Prepare variables
# ============================================================
exclude_vars = ["total_precipitation_6hr", "ageostrophic_wind_speed", "divergence"]
constant_vars = ["latitude", "longitude", "orography", "land_sea_mask"]
candidate_vars = [v for v in ds_europe.data_vars if v not in exclude_vars]
print("Candidate variables:", len(candidate_vars))

# Target variable (lazy)
precip = ds_europe["total_precipitation_6hr"]

# Spatial mean (lazy)
precip_mean = precip.mean(dim=("latitude", "longitude"))

# ============================================================
# 4. Lag function
# ============================================================
def lagged_spearman(x, y, lag_steps):
    rhos = []
    for lag in lag_steps:
        if lag > 0:
            rho, _ = spearmanr(x[:-lag], y[lag:], nan_policy="omit")
        else:
            rho, _ = spearmanr(x, y, nan_policy="omit")
        rhos.append(rho)
    return rhos

lags_hours = [0, 6, 12, 24, 36, 48]
lag_steps = [h // 6 for h in lags_hours]

# ============================================================
# 5. Load precip mean into memory
# ============================================================
with ProgressBar():
    precip_np = precip_mean.compute().values
print("\nLoaded precip_mean into memory:", precip_np.shape)

# ============================================================
# 6. Initialize or load partial correlation matrix
# ============================================================
csv_file = "spearman_lagged_correlations_partial.csv"
pkl_file = "spearman_lagged_correlations_partial.pkl"

if os.path.exists(pkl_file):
    # Resume from previous run
    with open(pkl_file, "rb") as f:
        corr_matrix = pickle.load(f)
    print("Resumed from previous partial results")
else:
    corr_matrix = pd.DataFrame(columns=lags_hours, dtype=float)

# ============================================================
# 7. Compute correlations with auto-save
# ============================================================
def var_has_level(var_name):
    """Return True if variable has a 'level' dimension"""
    return "level" in ds_europe[var_name].dims

for var in candidate_vars[30:45]:
    da = ds_europe[var]
    
    corr_matrix = pd.read_csv(csv_file, index_col=0)

    # Si la variable a une dimension 'level', on itère sur chaque niveau
    if "level" in da.dims:
        levels = da.level.values
    else:
        levels = [None]  # variable sans niveau

    for idx, lvl in enumerate(levels):
        # Nom de la ligne dans le DataFrame
        row_name = f"{var}_{lvl}" if lvl is not None else var

        # Skip si déjà calculé
        if row_name in corr_matrix.index and corr_matrix.loc[row_name].notnull().all():
            print(f"{row_name} already computed, skipping...")
            continue

        print(f"\nProcessing {row_name}")

        # Sélection du niveau courant
        if lvl is not None:
            da_lvl = da.isel(level=idx)
        else:
            da_lvl = da

        # Variable statique (pas de temps)
        if "time" not in da_lvl.dims:
            print("  → Static variable")
            da_stack = da_lvl.stack(z=("latitude", "longitude"))
            precip_stack = precip.mean(dim="time").stack(z=("latitude", "longitude"))

            with ProgressBar():
                x, y = dask.compute(da_stack, precip_stack)

            rho, _ = spearmanr(x, y, nan_policy="omit")
            
            if np.isnan(rho):
                corr_matrix.loc[row_name, :] = [0] * len(lags_hours)
            else:
                corr_matrix.loc[row_name, :] = [rho] * len(lags_hours)

        # Variable temporelle
        else:
            print("  → Temporal variable")
            da_mean = da_lvl.mean(dim=("latitude", "longitude"))

            with ProgressBar():
                da_np = da_mean.compute().values  # 1D time series

            rhos = lagged_spearman(da_np, precip_np, lag_steps)
                        
            if all(np.isnan(rhos)):
                corr_matrix.loc[row_name, :] = [0] * len(lags_hours)
            else:
                corr_matrix.loc[row_name, :] = rhos
            print(rhos)

        # Sauvegarde partielle après chaque niveau
        corr_matrix.to_csv(csv_file)
        with open(pkl_file, "wb") as f:
            pickle.dump(corr_matrix, f)
        print(f"Saved partial results for {row_name}")

# ============================================================
# 8. Save final results
# ============================================================
corr_matrix.to_csv("spearman_lagged_correlations_final.csv")
with open("spearman_lagged_correlations_final.pkl", "wb") as f:
    pickle.dump(corr_matrix, f)
print("Saved final correlation matrix")

# ============================================================
# 9. Plot
# ============================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Spearman ρ")
plt.xticks(ticks=np.arange(len(lags_hours)), labels=lags_hours)
plt.yticks(ticks=np.arange(len(corr_matrix.index)), labels=corr_matrix.index)
plt.xlabel("Lag (hours)")
plt.title("Lagged Spearman Correlation: Predictors vs Precipitation")
plt.tight_layout()
plt.show()
