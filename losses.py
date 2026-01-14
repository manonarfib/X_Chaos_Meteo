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
