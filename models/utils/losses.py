import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """
    Weighted pixel-wise MSE Loss.
    y_pred, y_true, weight : shape [B, C, H, W] or [B, 1, H, W]
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, weight=None):
        if weight is None:
            weight = torch.ones_like(y_true)

        # squared error
        se = (y_pred - y_true) ** 2

        # weighted mean
        loss = torch.sum(weight * se) / torch.sum(weight)
        return loss


class WeightedDiceRegressionLoss(nn.Module):
    """
    Dice-like loss adapted for regression.
    Works with arbitrary real-valued y_pred (no sigmoid needed).
    """

    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true, weight=None):
        if weight is None:
            weight = torch.ones_like(y_true)

        # Dice numerator
        intersection = torch.sum(weight * y_pred * y_true)

        # Dice denominator (L2-based for regression stability)
        denom = torch.sum(weight * (y_pred ** 2 + y_true ** 2))

        dice = (2 * intersection + self.epsilon) / (denom + self.epsilon)
        return 1 - dice



class AdvancedTorrentialLoss(nn.Module):
    """
    2025 AT loss - with annealing (PyTorch version)
    Version pytorch de la loss codée en keras https://github.com/kaetsraeb/at-loss/blob/main/at_loss.py
    y_pred, y_true : arbitrary real-valued tensors
    """

    def __init__(
        self,
        threshold,
        call_per_epoch,
        tau_init=1.0,
        tau_decay=0.005,
        tau_min=0.05,
        scale=0.05,
        device=None
    ):
        super().__init__()

        self.threshold = threshold
        self.call_per_epoch = int(call_per_epoch)

        self.tau_init = tau_init
        self.tau_decay = tau_decay
        self.tau_min = tau_min
        self.scale = scale

        # persistent state (like tf.Variable)
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("num_decay", torch.zeros((), dtype=torch.float))

        self.device = device

    def forward(self, y_pred, y_true):
        """
        y_pred, y_true : tensors of same shape
        """

        if self.device is None:
            self.device = y_pred.device

        # (count % call_per_epoch) == 0
        condition = (self.count % self.call_per_epoch) == 0

        if condition:
            self.num_decay += 1.0

        # annealed temperature
        tau = max(
            self.tau_init - self.tau_decay * (self.num_decay.item() - 1.0),
            self.tau_min
        )

        # increment call counter
        self.count += 1

        # z_true (binary thresholding)
        z_true = torch.where(
            y_true >= self.threshold,
            torch.tensor(1.0, device=y_true.device),
            torch.tensor(0.0, device=y_true.device),
        )

        # logistic noise
        l = self._sampling_logistic_noise(y_pred.shape, device=y_pred.device)

        # z_pred
        z_pred = torch.sigmoid(
            ((2.0 * y_pred) - (2.0 * self.threshold) + l) / tau
        )

        # squared error
        z = (z_true - z_pred) ** 2

        return z.mean()

    def _sampling_logistic_noise(self, shape, device):
        """
        Logistic(0, 1) noise with deterministic seed (like tf.random.stateless_uniform)
        """

        # emulate stateless seed using num_decay
        seed = int(self.num_decay.item())
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        u = torch.rand(
            shape,
            generator=g,
            device=device
        ).clamp_(min=1e-10, max=1.0 - 1e-10)

        logistic = torch.log(u) - torch.log(1.0 - u)
        return logistic * self.scale