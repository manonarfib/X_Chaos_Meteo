import torch
import torch.nn as nn
from typing import Tuple, List, Optional


class ConvLSTMCell(nn.Module):
    """
    Cellule ConvLSTM 2D.

    Entrée :
        - x_t : (B, C_in, H, W)
        - h_t : (B, C_hidden, H, W)
        - c_t : (B, C_hidden, H, W)

    Sortie :
        - h_{t+1} : (B, C_hidden, H, W)
        - c_{t+1} : (B, C_hidden, H, W)
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2  # pour garder H, W constants

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # Convolution qui produit les 4 "portes" en une fois
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_t : (B, C_in, H, W)
        # h_prev, c_prev : (B, C_hidden, H, W)

        combined = torch.cat([x_t, h_prev], dim=1)  # (B, C_in + C_hidden, H, W)
        conv_out = self.conv(combined)  # (B, 4 * C_hidden, H, W)

        # On découpe en 4 blocs de canaux : i, f, o, g
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(
        self, batch_size: int, spatial_size: Tuple[int, int], device=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W = spatial_size
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        return h, c


class ConvLSTM(nn.Module):
    """
    Bloc ConvLSTM multi-couches.

    Entrée :
        x : (B, T, C_in, H, W)  si batch_first=True

    Sortie (mode par défaut = last_state_only=True) :
        - h_T : (B, C_hidden_last, H, W)

      ou, si return_sequence=True :
        - H_seq : (B, T, C_hidden_last, H, W)
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        kernel_size: int = 3,
        num_layers: Optional[int] = None,
        batch_first: bool = True,
        bias: bool = True,
        return_sequence: bool = False,
    ):
        super().__init__()

        assert len(hidden_channels) > 0, "hidden_channels ne doit pas être vide."
        self.batch_first = batch_first
        self.return_sequence = return_sequence

        if num_layers is None:
            num_layers = len(hidden_channels)
        assert num_layers == len(
            hidden_channels
        ), "num_layers doit correspondre à la longueur de hidden_channels."

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        cells = []
        for layer_idx in range(num_layers):
            in_ch = input_channels if layer_idx == 0 else hidden_channels[layer_idx - 1]
            cell = ConvLSTMCell(
                input_channels=in_ch,
                hidden_channels=hidden_channels[layer_idx],
                kernel_size=kernel_size,
                bias=bias,
            )
            cells.append(cell)

        self.cells = nn.ModuleList(cells)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, C_in, H, W) si batch_first=True
        """
        if not self.batch_first:
            # (T, B, C, H, W) -> (B, T, C, H, W)
            x = x.permute(1, 0, 2, 3, 4)

        B, T, C_in, H, W = x.shape
        device = x.device

        # États cachés initiaux (un par couche)
        h_states = []
        c_states = []
        for layer_idx in range(self.num_layers):
            h, c = self.cells[layer_idx].init_hidden(
                batch_size=B, spatial_size=(H, W), device=device
            )
            h_states.append(h)
            c_states.append(c)

        # On va stocker les sorties de la dernière couche si on veut la séquence
        outputs_last_layer = []

        # Boucle temporelle
        for t in range(T):
            x_t = x[:, t]  # (B, C_in, H, W) ou (B, C_{l-1}, H, W)
            # Propagation à travers les couches ConvLSTM
            for layer_idx, cell in enumerate(self.cells):
                h_prev, c_prev = h_states[layer_idx], c_states[layer_idx]
                h_t, c_t = cell(x_t, h_prev, c_prev)
                h_states[layer_idx], c_states[layer_idx] = h_t, c_t
                # La sortie de cette couche devient l'entrée de la suivante
                x_t = h_t

            # À la fin de la dernière couche, x_t = h_t^{(dernier layer)}
            outputs_last_layer.append(x_t)

        # Empilement temporel
        H_seq = torch.stack(outputs_last_layer, dim=1)  # (B, T, C_last, H, W)

        if self.return_sequence:
            return H_seq  # séquence complète
        else:
            # Dernière sortie temporelle seulement
            h_T = H_seq[:, -1]  # (B, C_last, H, W)
            return h_T

class PrecipConvLSTM(nn.Module):
    """
    Modèle ConvLSTM pour prédire une carte de précipitations (tp_6h)
    à partir d'une séquence d'entrées (multivariées).

    Entrée :
        X : (B, T_in, C_in, H, W)

    Sortie :
        y_hat : (B, 1, H, W)  — carte de précipitations prédite pour l'instant cible
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int] = [64, 32, 16],
        kernel_size: int = 3,
    ):
        super().__init__()

        self.convlstm = ConvLSTM(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            batch_first=True,
            return_sequence=False,  # on ne récupère que la dernière sortie temporelle
        )

        # Projection finale en 1 canal (précipitation)
        self.head = nn.Conv2d(
            in_channels=hidden_channels[-1],
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T_in, C_in, H, W)
        """
        # Encode la séquence spatio-temporelle
        h_T = self.convlstm(x)  # (B, C_last, H, W)

        # Projection vers une carte de précipitations
        y_hat = self.head(h_T)  # (B, 1, H, W)

        return y_hat

if __name__ == "__main__":
    B = 2          # batch size
    T_in = 4       # nombre de pas de temps en entrée
    C_in = 10      # nombre de variables d'entrée (features)
    H, W = 64, 96  # taille spatiale

    model = PrecipConvLSTM(input_channels=C_in)
    x = torch.randn(B, T_in, C_in, H, W)

    y_hat = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", y_hat.shape)  # attendu: (2, 1, 64, 96)
