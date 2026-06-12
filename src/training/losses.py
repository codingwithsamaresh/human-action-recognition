import torch
import torch.nn as nn
import torch.nn.functional as F


def get_cross_entropy_loss():
    """
    Standard classification loss.
    """
    return nn.CrossEntropyLoss()


class FocalLoss(nn.Module):
    """
    Useful for class-imbalanced datasets.
    """

    def __init__(
        self,
        alpha=1.0,
        gamma=2.0,
        reduction="mean"
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits,
        targets
    ):

        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none"
        )

        pt = torch.exp(-ce_loss)

        focal_loss = (
            self.alpha
            * (1 - pt) ** self.gamma
            * ce_loss
        )

        if self.reduction == "mean":
            return focal_loss.mean()

        if self.reduction == "sum":
            return focal_loss.sum()

        return focal_loss


def get_focal_loss():
    return FocalLoss()