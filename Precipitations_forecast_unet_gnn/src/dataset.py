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
        
    def __len__(self):
        return len(self.time_indices) - CONTEXT_HOURS + 1
    
    def __getitem__(self, idx):
        t_start = self.time_indices[idx]
        t_end = t_start + CONTEXT_HOURS


        # X: (C, T, H, W)
        x_seq = self.ds_inputs.isel(time=slice(t_start, t_end)).to_array().values.astype('float32')
        # Rearrange to (C*T, H, W)
        C, T, H, W = x_seq.shape
        x = x_seq.reshape(C*T, H, W)


        y = self.ds_target.isel(time=t_end-1).values.astype('float32')
        y = np.log1p(y)[None, ...] # (1,H,W)
        
        # normalize
        if self.mean is not None and self.std is not None:
            for i, var in enumerate(self.ds_inputs.data_vars):
                m = self.mean.get(var, 0.0)
                s = self.std.get(var, 1.0)
                x[i::C] = (x[i::C] - m) / (s + 1e-8)


        x = torch.from_numpy(x)
        y = torch.from_numpy(y)


        if self.transform:
            x, y = self.transform(x, y)
        return x, y