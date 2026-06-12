import torch

from src.models.cnn_lstm_baseline import (
    CNNLSTMBaseline
)

model = CNNLSTMBaseline(
    num_classes=5
)

dummy = torch.randn(
    2,
    16,
    3,
    224,
    224
)

output = model(dummy)

print("=" * 50)

print("Input Shape :", dummy.shape)

print("Output Shape:", output.shape)

print(output)

print("=" * 50)