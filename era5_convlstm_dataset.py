import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
from typing import List


class ERA5ConvLSTMDataset(Dataset):
    """
    Dataset PyTorch pour entraîner un ConvLSTM sur ERA5 Europe (WeatherBench2).

    - Entrée : séquence X de shape (T_in, C_in, H, W)
        T_in = n_input_steps (par ex. 9)
        C_in = nb de variables d'entrée (toutes sauf tp_6h)
    - Cible : y = tp_6h à un temps futur (lead_steps * 6h), shape (H, W)

    ds : xarray.Dataset avec dimensions (time, latitude, longitude)
         et variables :
           - 'tp_6h' (target)
           - plusieurs variables features (entrées)
    """

    def __init__(
        self,
        ds: xr.Dataset,
        input_vars: List[str],
        target_var: str = "tp_6h",
        n_input_steps: int = 9,    # nombre de pas temporels en entrée (par ex. 9 => -48h...0h)
        lead_steps: int = 1,       # lead time (en indices de pas de temps : 1 => +6h)
    ):
        super().__init__()

        self.ds = ds
        self.input_vars = input_vars
        self.target_var = target_var
        self.n_input_steps = n_input_steps
        self.lead_steps = lead_steps

        times = ds.time.values
        n_time = len(times)

        # temps "centre" t0 pour lesquels on a :
        #  - n_input_steps-1 pas dans le passé
        #  - lead_steps pas dans le futur pour la cible
        # index minimal t0 = (n_input_steps - 1)
        # index maximal t0 = n_time - 1 - lead_steps
        t_min = n_input_steps - 1
        t_max = n_time - 1 - lead_steps

        if t_max <= t_min:
            raise ValueError(
                f"Pas assez de pas de temps pour n_input_steps={n_input_steps} "
                f"et lead_steps={lead_steps} (n_time={n_time})"
            )

        self.valid_time_indices = np.arange(t_min, t_max + 1)

        # Juste pour sanity check : mémoriser tailles spatiales
        self.H = ds.dims["latitude"]
        self.W = ds.dims["longitude"]

    def __len__(self):
        return len(self.valid_time_indices)

    def __getitem__(self, idx):
        """
        Retourne :
            X : (T_in, C_in, H, W)
            y : (H, W)
        """
        t0 = int(self.valid_time_indices[idx])

        t_start = t0 - (self.n_input_steps - 1)  # index du temps le plus ancien
        t_end = t0 + 1                           # slice exclusif, donc [t_start, ..., t0]
        t_target = t0 + self.lead_steps          # instant futur pour la cible

        # --- Entrée X : features sur la fenêtre temporelle ---
        ds_in = self.ds[self.input_vars].isel(time=slice(t_start, t_end))
        # dims : time, latitude, longitude pour chaque variable

        # On empile les variables en dimension "channel"
        # -> DataArray (channel, time, lat, lon)
        da_in = ds_in.to_array("channel")

        # On réordonne en (time, channel, lat, lon)
        da_in = da_in.transpose("time", "channel", "latitude", "longitude")

        # Conversion en numpy puis torch (déclenche un compute pour cette petite slice)
        x_np = da_in.values.astype("float32")  # (T_in, C_in, H, W)
        X = torch.from_numpy(x_np)

        # --- Cible y : tp_6h à t_target ---
        da_out = self.ds[self.target_var].isel(time=t_target)  # (lat, lon)
        y_np = da_out.values.astype("float32")  # (H, W)
        y = torch.from_numpy(y_np)

        return X, y
