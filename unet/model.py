import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import *


def _xavier_uniform_(module: nn.Module):  # Initialisation des couches
    for m in module.modules():
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class Conv3dSame(nn.Module):
    """
    Conv3d 'same' à la TensorFlow (padding asymétrique si kernel pair).
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv3d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride, bias=bias)

    def forward(self, x):
        kD, kH, kW = self.kernel_size
        pad_d_total = kD - 1
        pad_h_total = kH - 1
        pad_w_total = kW - 1

        pad_d_left = pad_d_total // 2
        pad_d_right = pad_d_total - pad_d_left
        pad_h_left = pad_h_total // 2
        pad_h_right = pad_h_total - pad_h_left
        pad_w_left = pad_w_total // 2
        pad_w_right = pad_w_total - pad_w_left

        # F.pad order: (W_left, W_right, H_left, H_right, D_left, D_right)
        x = F.pad(
            x,
            (pad_w_left, pad_w_right, pad_h_left,
             pad_h_right, pad_d_left, pad_d_right)
        )
        return self.conv(x)


class LagConv3d(nn.Module):
    """
    Équivalent de Conv3D(kernel=(lags,1,1), padding='valid') :
    - réduit la dimension temporelle (D) via un noyau couvrant lags positions.
    """

    def __init__(self, in_ch, out_ch, lags, bias=True):
        super().__init__()
        self.lags = int(lags)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=(
            self.lags, 1, 1), padding=0, bias=bias)

    def forward(self, x):
        # x: (B, C, D, H, W) ; sortie D' = D - lags + 1 (si D == lags => D' = 1)
        return F.relu(self.conv(x))


class DoubleConv3d(nn.Module):
    """Deux convs 3D: (3x3x3 same) + ReLU, comme dans la partie "encoder/decoder" TF."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = Conv3dSame(in_ch, out_ch, kernel_size=3)
        self.c2 = Conv3dSame(out_ch, out_ch, kernel_size=3)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        return x


class UNet3D(nn.Module):
    """
    Un seul U-Net (le bloc répété pour streamA et streamB).
    Entrée: (B, features, lags, latitude, longitude)
    Sortie: (B, features_output, D_out, H_out, W_out)  (D_out dépend de lags et des conv (lags,1,1))
    """

    def __init__(self, lags, features_output, filters=16, dropout=0.0):
        super().__init__()
        self.lags = int(lags)
        self.features_output = int(features_output)
        self.filters = int(filters)

        # --- Encoder ---
        # in_ch défini plus bas via build()
        self.enc1 = DoubleConv3d(in_ch=None, out_ch=self.filters)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.enc2 = DoubleConv3d(
            in_ch=2 * self.filters, out_ch=2 * self.filters)
        self.enc3 = DoubleConv3d(
            in_ch=4 * self.filters, out_ch=4 * self.filters)
        self.enc4 = DoubleConv3d(
            in_ch=8 * self.filters, out_ch=8 * self.filters)
        self.drop4 = nn.Dropout(dropout)

        # --- Bottleneck ---
        self.bottleneck_conv = Conv3dSame(
            16 * self.filters, 16 * self.filters, kernel_size=3)
        self.compress_lags5 = LagConv3d(
            16 * self.filters, 16 * self.filters, lags=self.lags)
        self.bottleneck_post = Conv3dSame(
            16 * self.filters, 16 * self.filters, kernel_size=3)
        self.drop5 = nn.Dropout(dropout)

        # --- Decoder (up blocks) ---
        self.up6 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            Conv3dSame(8 * self.filters, 8 * self.filters, kernel_size=2),
        )
        self.compress6 = LagConv3d(
            8 * self.filters, 8 * self.filters, lags=self.lags)
        self.dec6 = DoubleConv3d(
            in_ch=16 * self.filters, out_ch=8 * self.filters)

        self.up7 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            Conv3dSame(4 * self.filters, 4 * self.filters, kernel_size=2),
        )
        self.compress7 = LagConv3d(
            4 * self.filters, 4 * self.filters, lags=self.lags)
        self.dec7 = DoubleConv3d(
            in_ch=8 * self.filters, out_ch=4 * self.filters)

        self.up8 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            Conv3dSame(2 * self.filters, 2 * self.filters, kernel_size=2),
        )
        self.compress8 = LagConv3d(
            2 * self.filters, 2 * self.filters, lags=self.lags)
        self.dec8 = DoubleConv3d(
            in_ch=4 * self.filters, out_ch=2 * self.filters)

        self.up9 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            Conv3dSame(self.filters, self.filters, kernel_size=2),
        )
        self.compress9 = LagConv3d(self.filters, self.filters, lags=self.lags)
        self.dec9 = DoubleConv3d(in_ch=2 * self.filters, out_ch=self.filters)

        # "conv9" (2*features_output) + "conv10" (features_output) avec ReLU (comme dans ton TF)
        self.final_conv_2 = Conv3dSame(
            self.filters, 2 * self.features_output, kernel_size=3)
        self.final_conv_1 = nn.Conv3d(
            2 * self.features_output, self.features_output, kernel_size=1)

        self._built = False

    def _build(self, in_features: int):
        # Le premier bloc dépend du nombre de canaux d'entrée (features)
        self.enc1 = DoubleConv3d(in_ch=in_features, out_ch=self.filters)
        _xavier_uniform_(self)
        self._built = True

    def forward(self, x):
        # x: (B, C=features, D=lags, H, W)
        if not self._built:
            self._build(in_features=x.shape[1])

        # --- Encoder ---
        conv1 = self.enc1(x)
        pool1 = self.pool(conv1)

        conv2 = self.enc2(pool1)
        pool2 = self.pool(conv2)

        conv3 = self.enc3(pool2)
        pool3 = self.pool(conv3)

        conv4 = self.enc4(pool3)
        drop4 = self.drop4(conv4)

        # --- Bottleneck ---
        pool4 = self.pool(drop4)
        conv5 = F.relu(self.bottleneck_conv(pool4))

        compress_lags5 = self.compress_lags5(conv5)  # (lags,1,1) valid
        conv5 = F.relu(self.bottleneck_post(compress_lags5))
        drop5 = self.drop5(conv5)

        # --- Decoder ---
        up6 = self.up6(drop5)
        compress_lags6 = self.compress6(drop4)
        merge6 = torch.cat([compress_lags6, up6], dim=1)
        conv6 = self.dec6(merge6)

        up7 = self.up7(conv6)
        compress_lags7 = self.compress7(conv3)
        merge7 = torch.cat([compress_lags7, up7], dim=1)
        conv7 = self.dec7(merge7)

        up8 = self.up8(conv7)
        compress_lags8 = self.compress8(conv2)
        merge8 = torch.cat([compress_lags8, up8], dim=1)
        conv8 = self.dec8(merge8)

        up9 = self.up9(conv8)
        compress_lags9 = self.compress9(conv1)
        merge9 = torch.cat([compress_lags9, up9], dim=1)
        conv9 = self.dec9(merge9)

        conv9 = F.relu(self.final_conv_2(conv9))
        out = F.relu(self.final_conv_1(conv9))  # comme TF (relu sur le 1x1)

        return out


class WFUNet(nn.Module):
    """
    x: liste de N entrées (1,...N) -> N UNet identiques -> concat canaux -> 1x1 linear vers features_output.
    Entrées attendues: [x1, ..., xN] chacune en (B, features, lags, latitude, longitude)
    """

    def __init__(self, lags, latitude, longitude, features, features_output, filters=16, dropout=0.0):
        super().__init__()
        self.lags = int(lags)
        self.latitude = int(latitude)
        self.longitude = int(longitude)
        self.features = int(features)
        self.features_output = int(features_output)
        self.streams = []

        for i_feature in range(self.features):
            self.streams.append(UNet3D(lags=self.lags, features_output=self.features_output,
                                       filters=filters, dropout=dropout))

        # self.streamA = UNet3D(lags=self.lags, features_output=self.features_output,
        #                       filters=filters, dropout=dropout)
        # self.streamB = UNet3D(lags=self.lags, features_output=self.features_output,
        #                       filters=filters, dropout=dropout)

        # fusion: concat -> 1x1 linear (pas d'activation), comme TF (activation='linear')
        self.fusion = nn.Conv3d(2 * self.features_output,
                                self.features_output, kernel_size=1)

    def forward(self, x):
        outs = []
        for i_feature in range(self.features):
            outs.append(self.streams[i_feature](x[i_feature]))
        fused = torch.cat(outs, dim=1)  # concat sur les canaux
        return self.fusion(fused)


class WFUNet_with_train(WFUNet):
    def __init__(self, loss_fn=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Si aucun criterion n'est fourni :
        self.criterion = loss_fn if loss_fn is not None else nn.MSELoss()

    def compute_loss(self, output, target, loss_type="w_mse_and_w_dice"):
        if loss_type in ["w_mse", "w_dice", "w_mse_and_w_dice"]:
            weight = torch.where(
                target > 0.1,
                torch.tensor(5.0, device=target.device),
                torch.tensor(1.0, device=target.device)
            )
        else:
            weight = None

        if loss_type == "w_mse_and_w_dice":
            criterion_mse = WeightedMSELoss(weight)
            criterion_dice = WeightedDiceRegressionLoss(weight)
            loss_mse = criterion_mse(output, target)
            loss_dice = criterion_dice(output, target)
            return 0.7 * loss_mse + 0.3 * loss_dice

        elif loss_type == "w_dice":
            criterion_dice = WeightedDiceRegressionLoss(weight)
            return criterion_dice(output, target)

        elif loss_type == "w_mse":
            criterion_mse = WeightedMSELoss(weight)
            return criterion_mse(output, target)

        else:  # mse only
            criterion_mse = WeightedMSELoss()
            return criterion_mse(output, target)

    # ---------------------------------------------------------------------
    #  ONE EPOCH TRAINING
    # ---------------------------------------------------------------------
    def train_one_epoch(self, train_loader, optimizer, loss_type, device):
        self.train()
        total_loss = 0

        for x, target in train_loader:
            x = x.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = self(x)
            loss = self.compute_loss(output, target, loss_type)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        return total_loss / len(train_loader.dataset)

    # ---------------------------------------------------------------------
    #  VALIDATION
    # ---------------------------------------------------------------------
    def evaluate(self, val_loader, loss_type, device):
        self.eval()
        total_loss = 0

        with torch.no_grad():
            for x, target in val_loader:
                x = x.to(device)
                target = target.to(device)
                output = self(x)
                loss = self.compute_loss(output, target, loss_type)
                total_loss += loss.item() * x.size(0)

        return total_loss / len(val_loader.dataset)

    # ---------------------------------------------------------------------
    #  COMPLETE TRAINING LOOP
    # ---------------------------------------------------------------------
    def fit(self, train_loader, val_loader, optimizer, scheduler,
            epochs, loss_type, device, save_path="best_model.pt"):

        train_losses = []
        val_losses = []
        best_val = float("inf")

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(
                train_loader, optimizer, loss_type, device
            )
            val_loss = self.evaluate(val_loader, loss_type, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}"
            )

            # save best
            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.state_dict(), save_path)
                print("Saved new best model!")

        return train_losses, val_losses
