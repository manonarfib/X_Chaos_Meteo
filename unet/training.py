import numpy as np
import torch
import torch.nn as nn
from model import WFUNet_with_train
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

np.random.seed(1)
torch.manual_seed(2)
torch.cuda.manual_seed_all(2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Hyperparameters ---- #
lags = 12
lat = 96
long = 96
feats = 1
feats_out = 1
filters = 16
dropout = 0.5
batch_size = 2
epochs = 200
learning_rate = 1e-3
# choisir entre w_mse_and_w_dice, w_mse, w_dice or mse
loss_type = "w_mse_and_w_dice"

# ---- Model ---- #
model = WFUNet_with_train(lags, lat, long, feats, feats_out,
                          filters, dropout).to(device)

# ---- Optimizer / Scheduler ---- #
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(
    optimizer, factor=0.5, patience=10, min_lr=1e-4, verbose=True)

# ---- Train ---- #
train_losses, val_losses = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=epochs,
    loss_type=loss_type,
    device=device,
    save_path="best_model.pt"
)
