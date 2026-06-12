import torch

from src.training.metrics import (
    top1_accuracy,
    topk_accuracy,
    precision_recall_f1
)

logits = torch.tensor([
    [5.0, 1.0, 0.5],
    [0.1, 4.0, 0.2],
    [0.3, 0.4, 3.5],
    [2.0, 1.0, 0.5]
])

targets = torch.tensor([
    0,
    1,
    2,
    1
])

print("=" * 50)

print(
    "Top1:",
    top1_accuracy(
        logits,
        targets
    )
)

print(
    "Top3:",
    topk_accuracy(
        logits,
        targets,
        k=3
    )
)

metrics = precision_recall_f1(
    logits,
    targets
)

print(metrics)

print("=" * 50)