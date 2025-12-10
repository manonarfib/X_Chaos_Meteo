import torch
from torch.utils.data import Dataset
import numpy as np
from .config import CONTEXT_HOURS


class ERA5FieldDataset(Dataset):
    def __init__(self, ds_inputs, ds_target, time_indices, mean_dict=None, std_dict=None, transform=None):
        self.ds_inputs = ds_inputs
        self.ds_target = ds_target
        self.time_indices = np.array(time_indices)
        self.mean = mean_dict
        self.std = std_dict
        self.transform = transform
        
        self.vars_with_time = []
        self.vars_without_time = []
        
        for v in ds_inputs.data_vars:
            if "time" in ds_inputs[v].dims:
                self.vars_with_time.append(v)
            else:
                self.vars_without_time.append(v)
        
    def __len__(self):
        return len(self.time_indices) - CONTEXT_HOURS + 1
    
    def __getitem__(self, idx):
        t_start = self.time_indices[idx]
        t_end = t_start + CONTEXT_HOURS
        
        dyn_list = []
        static_list = []

        # Load time-dependent variables
        for v in self.vars_with_time:
            da_var = self.ds_inputs[v]
            arr = da_var.isel(time=slice(t_start, t_end)).values.astype("float32")  # (T,H,W)
            dyn_list.append(arr)

        # Load static variables (shape: H,W)
        for v in self.vars_without_time:
            da_var = self.ds_inputs[v]
            arr = da_var.values.astype("float32")  # (H,W)
            static_list.append(arr)

        # Stack dynamic variables: list of (T,H,W) => (C_dyn,T,H,W)
        if len(dyn_list) > 0:
            dyn_vars = np.stack(dyn_list, axis=0)
            C_dyn, T, H, W = dyn_vars.shape
            dyn_vars = dyn_vars.reshape(C_dyn * T, H, W)
        else:
            dyn_vars = np.zeros((0, H, W), dtype="float32")

        # Stack static variables: list of (H,W) => (C_static,H,W)
        if len(static_list) > 0:
            static_vars = np.stack(static_list, axis=0)
        else:
            static_vars = np.zeros((0, H, W), dtype="float32")

        # Concatenate: (C_dyn*T + C_static, H, W)
        x = np.concatenate([dyn_vars, static_vars], axis=0)

        FORECAST_HORIZON = 6 # A VERIFIER
        y = self.ds_target.isel(time=t_end + FORECAST_HORIZON - 1).values.astype('float32')
        y = np.log1p(y)[None, ...] # (1,H,W)
        
        # normalize
        if self.mean is not None and self.std is not None:
            # dynamic vars: names repeated T times
            full_var_list = (
                [v for v in self.vars_with_time for _ in range(CONTEXT_HOURS)]
                + self.vars_without_time
            )
            for i, var in enumerate(full_var_list):
                m = self.mean.get(var, 0.0)
                s = self.std.get(var, 1.0)
                x[i] = (x[i] - m) / (s + 1e-8)


        x = torch.from_numpy(x)
        y = torch.from_numpy(y)


        if self.transform:
            x, y = self.transform(x, y)
        return x, y