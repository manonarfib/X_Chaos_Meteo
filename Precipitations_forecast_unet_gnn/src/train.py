import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataloader import open_era5_region, compute_norms
from dataset import ERA5FieldDataset
from models.unet import UNet
from models.mpnn import GridMPNN
from config import BATCH_SIZE, NUM_WORKERS, LR, EPOCHS, DEVICE, CHECKPOINT_DIR, UNET_FEATURES, MPNN_HIDDEN
from utils import rmse, save_checkpoint


# 1. Data
ds_inputs, ds_target = open_era5_region()
mean, std = compute_norms(ds_inputs)
ntime = ds_inputs.sizes['time']
idx = np.arange(ntime)
train_idx = idx[:int(0.8*ntime)]
val_idx = idx[int(0.8*ntime):int(0.9*ntime)]


train_ds = ERA5FieldDataset(ds_inputs, ds_target, train_idx, mean, std)
val_ds = ERA5FieldDataset(ds_inputs, ds_target, val_idx, mean, std)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# 2. Models
unet = UNet(in_channels=len(ds_inputs.data_vars)*48).to(DEVICE)
# Suppose we select K points to apply MPNN (could be uniform sampling or points of interest)
# Edge index construit selon voisinage 4-connexions ou KNN
mpnn = GridMPNN(in_channels=UNET_FEATURES[-1], hidden_channels=MPNN_HIDDEN).to(DEVICE)


# 3. Optimizer
optimizer = torch.optim.Adam(list(unet.parameters()) + list(mpnn.parameters()), lr=LR)
criterion = torch.nn.MSELoss()


# 4. Training loop
for epoch in range(1, EPOCHS+1):
    # ----- Training -----
    unet.train(); mpnn.train()
    train_losses = []
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} train"):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        feat = unet(xb)  # global feature map

        # Flatten features to nodes (NxF) for MPNN
        B,C,H,W = feat.shape
        nodes = feat.permute(0,2,3,1).reshape(-1, C)
        edge_index = ... # TO DO
        out_nodes = mpnn(nodes, edge_index)
        out = out_nodes.reshape(B,1,H,W)

        # Pour simplifier, si pas de MPNN, on peut utiliser UNet directement
        out = feat[:,0:1,...]

        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    print(f"Epoch {epoch}, train loss: {np.mean(train_losses):.6f}")

    # ----- Validation -----
    unet.eval(); mpnn.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch} val"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            feat = unet(xb)
            out = feat[:,0:1,...]
            val_loss = criterion(out, yb)
            val_losses.append(val_loss.item())

    print(f"Epoch {epoch}, val loss: {np.mean(val_losses):.6f}")

    # Checkpoints
    save_checkpoint(unet, optimizer, epoch, CHECKPOINT_DIR/f"unet_epoch_{epoch}.pt")
