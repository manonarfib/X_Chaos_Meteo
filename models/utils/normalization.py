# import xarray as xr
# import dask
# from dask.diagnostics import ProgressBar
# import numpy as np

# train_dataset_path = "/mounts/datasets/datasets/x_chaos_meteo/dataset_era5/era5_europe_ml_train.zarr"

# # ouverture avec chunking sur le temps pour ne pas tout charger
# ds = xr.open_zarr(train_dataset_path, chunks={"time": 32})  # tu peux ajuster 10 -> 20/50
# X = ds["X"]  # dims: (time, channel, lat, lon)

# print("start lazy calculation")
# # lazy calculation
# mean = X.mean(dim=("time", "latitude", "longitude"))
# print("mean done")
# std = X.std(dim=("time", "latitude", "longitude"))
# print("std done")
# # Progress bar Dask
# with ProgressBar():
#     mean_values = mean.compute()  # affichera la progression
#     std_values = std.compute()

# # sauvegarde
# np.save("era5_mean.npy", mean_values)
# np.save("era5_std.npy", std_values)


import numpy as np

mean_values = np.load("era5_mean.npy")
std_values  = np.load("era5_std.npy")

print("Shape mean:", mean_values.shape)
print("Shape std:", std_values.shape)

print("Mean first 5 channels:", mean_values[:5])
print("Std first 5 channels:", std_values[:5])