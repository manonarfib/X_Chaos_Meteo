import numpy as np
import torch
import torch.nn as nn
from model import WFUNet
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

# ---- Model ---- #
model = WFUNet(lags, lat, long, feats, feats_out, filters, dropout).to(device)

# ---- Loss / Optimizer ---- #
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(
    optimizer, factor=0.5, patience=10, min_lr=1e-4, verbose=True)

# ---- Training loop ---- #
train_losses = []
val_losses = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for x, target in train_loader:
        x = x.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(x)
        # output shape: (B, C, D=1, H, W)
        # target shape: (B, C, D=1, H, W)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x[0].size(0)
    epoch_loss /= len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, target in val_loader:
            x = x.to(device)
            target = target.to(device)
            output = model(x)
            loss = criterion(output, target)
            val_loss += loss.item() * x[0].size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

    scheduler.step(val_loss)

    print(
        f"Epoch [{epoch+1}/{epochs}] Train Loss: {epoch_loss:.6f} Val Loss: {val_loss:.6f}")

    # Save best model
    if epoch == 0 or val_loss < min(val_losses[:-1]):
        torch.save(model.state_dict(), "best_model.pt")
        print("Saved new best model!")
