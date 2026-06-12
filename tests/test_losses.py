import torch

from src.training.losses import (
    get_cross_entropy_loss,
    get_focal_loss
)

logits = torch.randn(4, 5)

targets = torch.tensor(
    [0, 1, 2, 3]
)

ce = get_cross_entropy_loss()
focal = get_focal_loss()

print("=" * 50)

print(
    "CrossEntropy:",
    ce(logits, targets).item()
)

print(
    "FocalLoss:",
    focal(logits, targets).item()
)

print("=" * 50)