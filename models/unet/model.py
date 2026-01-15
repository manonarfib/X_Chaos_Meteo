import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import csv
import pandas as pd
from tqdm import tqdm
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

    def __init__(self, lags, features,features_output, filters=16, dropout=0.0):
        super().__init__()
        self.lags = int(lags)
        self.features=int(features)
        self.features_output = int(features_output)
        self.filters = int(filters)

        # --- Encoder ---
        # self.enc1 = DoubleConv3d(in_ch=1, out_ch=self.filters)
        self.enc1 = DoubleConv3d(in_ch=features, out_ch=self.filters)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.enc2 = DoubleConv3d(
            in_ch=self.filters, out_ch=2 * self.filters)
        self.enc3 = DoubleConv3d(
            in_ch=2 * self.filters, out_ch=4 * self.filters)
        self.enc4 = DoubleConv3d(
            in_ch=4 * self.filters, out_ch=8 * self.filters)
        self.drop4 = nn.Dropout(dropout)

        # --- Bottleneck ---
        self.bottleneck_conv = Conv3dSame(
            8 * self.filters, 16 * self.filters, kernel_size=3)
        self.compress_lags5 = LagConv3d(
            16 * self.filters, 16 * self.filters, lags=self.lags)
        self.bottleneck_post = Conv3dSame(
            16 * self.filters, 16 * self.filters, kernel_size=3)
        self.drop5 = nn.Dropout(dropout)

        # --- Decoder (up blocks) ---
        self.up6 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            Conv3dSame(16 * self.filters, 8 * self.filters, kernel_size=2),
        )
        self.compress6 = LagConv3d(
            8 * self.filters, 8 * self.filters, lags=self.lags)
        self.dec6 = DoubleConv3d(
            in_ch=16 * self.filters, out_ch=8 * self.filters)

        self.up7 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            Conv3dSame(8 * self.filters, 4 * self.filters, kernel_size=2),
        )
        self.compress7 = LagConv3d(
            4 * self.filters, 4 * self.filters, lags=self.lags)
        self.dec7 = DoubleConv3d(
            in_ch=8 * self.filters, out_ch=4 * self.filters)

        self.up8 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            Conv3dSame(4 * self.filters, 2 * self.filters, kernel_size=2),
        )
        self.compress8 = LagConv3d(
            2 * self.filters, 2 * self.filters, lags=self.lags)
        self.dec8 = DoubleConv3d(
            in_ch=4 * self.filters, out_ch=2 * self.filters)

        self.up9 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            Conv3dSame(2*self.filters, self.filters, kernel_size=2),
        )
        self.compress9 = LagConv3d(self.filters, self.filters, lags=self.lags)
        self.dec9 = DoubleConv3d(in_ch=2 * self.filters, out_ch=self.filters)

        # "conv9" (2*features_output) + "conv10" (features_output) avec ReLU (comme dans ton TF)
        self.final_conv_2 = Conv3dSame(
            self.filters, 2 * self.features_output, kernel_size=3)
        self.final_conv_1 = nn.Conv3d(
            2 * self.features_output, self.features_output, kernel_size=1)

        self._built = False

    def _build(self):
        _xavier_uniform_(self)
        self._built = True

    def forward(self, x):
        # x: (B, C=features, D=lags, H, W)
        if not self._built:
            self._build()

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

        compress_lags5 = self.compress_lags5(conv5)
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
        # out = F.relu(self.final_conv_1(conv9))
        out = self.final_conv_1(conv9)

        return out



class WFUNet(nn.Module):
    """
    x: liste de N entrées (1,...N) -> N UNet identiques -> concat canaux -> 1x1 linear vers features_output.
    Entrées attendues: [x1, ..., xN] chacune en (B, features, lags, latitude, longitude)
    """

    def __init__(self, lags, latitude, longitude, features, features_output, batch_size, filters=16, dropout=0.0):
        super().__init__()
        self.batch_size=int(batch_size)
        self.lags = int(lags)
        self.latitude = int(latitude)
        self.longitude = int(longitude)
        self.features = int(features)
        self.features_output = int(features_output)
        # self.streams = nn.ModuleList()

        # for i_feature in range(self.features):
        #     # Mémoire allouée par PyTorch (actuellement utilisée)
        #     print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        #     print(f"Creating Unet model for variable {i_feature}")
        #     self.streams.append(UNet3D(lags=self.lags, features=self.features, features_output=self.features_output,
        #                                filters=filters, dropout=dropout))

        # self.fusion = nn.Conv3d(self.features * self.features_output,
        #                         self.features_output, kernel_size=1)
        
        self.unet = UNet3D(
            lags=self.lags,
            features=self.features,           # toutes les features en entrée
            features_output=self.features_output,
            filters=filters,
            dropout=dropout
        )
        

    def forward(self, x):

        b, T, C, H, W = x.shape
        H_orig, W_orig = H, W
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        # padding: (left, right, top, bottom) pour les 2 dernières dimensions
        x = F.pad(
            x,
            pad=(0, pad_w, 0, pad_h),  # W puis H
            mode="constant",
            value=0
        )
        x = x.permute(0, 2, 1, 3, 4)

        # outs = []
        # for i_feature in range(self.features):
        #     x_feature_i = x[:, :, i_feature, :, :].unsqueeze(1)
        #     out=self.streams[i_feature](x_feature_i)
        #     out = out[..., :H_orig, :W_orig]
        #     outs.append(out)
        # fused = torch.cat(outs, dim=1)  # concat sur les canaux
        
        # return self.fusion(fused)

        out = self.unet(x)  # UNet traite toutes les features en une seule passe
        out = out[..., :H_orig, :W_orig]  # enlever le padding

        return out


class WFUNet_with_train(WFUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            criterion_mse = WeightedMSELoss()
            criterion_dice = WeightedDiceRegressionLoss()
            loss_mse = criterion_mse(output, target, weight)
            loss_dice = criterion_dice(output, target, weight)
            return 0.7 * loss_mse + 0.3 * loss_dice

        elif loss_type == "w_dice":
            criterion_dice = WeightedDiceRegressionLoss()
            return criterion_dice(output, target,weight)

        elif loss_type == "w_mse":
            criterion_mse = WeightedMSELoss()
            return criterion_mse(output, target,weight)

        else:  # mse only
            criterion_mse = WeightedMSELoss()
            return criterion_mse(output, target)


    # ---------------------------------------------------------------------
    #  VALIDATION
    # ---------------------------------------------------------------------
    def evaluate(self, val_loader, loss_type, device):
        self.eval()
        total_loss = 0

        num_batches_val = len(val_loader)

        with torch.no_grad():
            pbar_val = tqdm(val_loader, total=num_batches_val, leave=True)
            for batchid, (x, target, i) in enumerate(pbar_val):
                x = x.to(device)
                target = target.to(device)
                output = self(x)
                print("OUTPUT", output)
                print("TARGET", target)
                loss = self.compute_loss(output, target, loss_type)
                total_loss += loss.item() * x.size(0)

        return total_loss / len(val_loader.dataset)

    # ---------------------------------------------------------------------
    #  COMPLETE TRAINING LOOP
    # ---------------------------------------------------------------------
    def fit(self, train_loader, val_loader, optimizer, scheduler,
            epochs, loss_type, device, weight_update_interval, val_loss_calculation_interval, save_path="best_model.pt"):

        last_checkpoint = os.path.join(save_path, "checkpoint_last.pt")
        os.makedirs(os.path.dirname(last_checkpoint), exist_ok=True)

        start_epoch=1

        if os.path.exists(last_checkpoint):
            checkpoint = torch.load(last_checkpoint, map_location=device)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"[CHECKPOINT] Chargé depuis {last_checkpoint}, reprise à l'époque {start_epoch}")
        else:
            print("[CHECKPOINT] Aucun checkpoint trouvé, entraînement depuis le début")


        csv_path_train = os.path.join(save_path,"train_log.csv")
        csv_path_val = os.path.join(save_path,"validation_log.csv")

        previous_val_b_loss = float("inf")
        val_batches_loss=0

        total_nb_train_batches=len(train_loader)

        accumulation_steps=weight_update_interval//self.batch_size
        val_calculation_steps=val_loss_calculation_interval//self.batch_size

        for epoch in range(start_epoch, epochs+1):

            self.train()
            optimizer.zero_grad()
            epoch_loss = 0.0
            accumulation_step_loss = 0

            epoch_start=time.time()

            pbar = tqdm(train_loader, total=total_nb_train_batches, desc=f"Epoch {epoch}/{epochs}", leave=True)
            for batch_idx, (x, target, i) in enumerate(pbar):

                batch_start = time.time()
                print(f"Training for batch {batch_idx}")
                x = x.to(device)
                target = target.to(device)

                output = self(x)

                raw_loss = self.compute_loss(output, target, loss_type)

                if torch.isnan(raw_loss).any():
                    date = pd.to_datetime(train_loader.dataset.Y.time.values[i])
                    print(f"NaN détecté à la date {date}")
                    continue

                (raw_loss / accumulation_steps).backward()
                epoch_loss += raw_loss.item()
                accumulation_step_loss += raw_loss / accumulation_steps

                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_nb_train_batches:
                    optimizer.step()
                    optimizer.zero_grad()

                    batch_time = time.time() - batch_start
                    print(
                        f"[Epoch {epoch}/{epochs}] "
                        f"Batch {batch_idx+1}/{total_nb_train_batches} "
                        f"- Loss: {accumulation_step_loss.item():.4e} "
                        f"- Batch time: {batch_time:.2f}s"
                    )

                    with open(csv_path_train, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, batch_idx, accumulation_step_loss])

                    accumulation_step_loss=0

                if (batch_idx + 1) % val_calculation_steps == 0 or (batch_idx + 1) == total_nb_train_batches:
                    self.eval()
                    previous_val_b_loss = val_batches_loss
                    val_batches_loss = self.evaluate(val_loader, loss_type, device)

                    if previous_val_b_loss>val_batches_loss:
                        best_checkpoint = os.path.join(save_path, f"best_checkpoint_epoch{epoch}_batch_idx{batch_idx}.pt")
                        torch.save({
                            "epoch": epoch,
                            "model_state_dict": self.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        }, best_checkpoint)

                    with open(csv_path_val, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, batch_idx, "val", val_batches_loss])

                    scheduler.step(val_batches_loss)
                    self.train()

                    

                batch_end = time.time()
                print(f"event : training batch end, batch {batch_idx}, duration :{batch_end - batch_start}")

                       
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / total_nb_train_batches
            pbar.set_postfix(loss=f"{epoch_loss:.3e}", avg=f"{avg_loss:.3e}", bt=f"{batch_time:.2f}s")
            print(
                f"\n>>> Epoch {epoch}/{epochs} terminée "
                f"- Loss moyen: {avg_loss:.4e} "
                f"- Temps epoch: {epoch_time:.1f}s\n"
            )

            torch.save(self.state_dict(), os.path.join(save_path, f"epoch_{epoch}.pt"))
            print(f"Saved model for epoch {epoch}!")

            epoch_checkpoint = os.path.join(save_path, f"epoch{epoch}_full.pt")
            torch.save({'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, epoch_checkpoint)
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, last_checkpoint)
            print(f"[CHECKPOINT] Sauvegardé à {epoch_checkpoint}")

        return 
