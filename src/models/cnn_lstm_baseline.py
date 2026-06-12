import torch
import torch.nn as nn

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)


class CNNLSTMBaseline(nn.Module):

    def __init__(
        self,
        num_classes,
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
        pretrained=True
    ):
        super().__init__()

        # -------------------------
        # CNN Backbone
        # -------------------------

        if pretrained:
            weights = MobileNet_V3_Small_Weights.DEFAULT
        else:
            weights = None

        backbone = mobilenet_v3_small(
            weights=weights
        )

        self.feature_extractor = backbone.features

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        feature_dim = 576

        # -------------------------
        # Temporal Encoder
        # -------------------------

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # -------------------------
        # Classifier
        # -------------------------

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size,
                num_classes
            )
        )

    def forward(self, x):

        # x:
        # (B, T, C, H, W)

        batch_size, seq_len, C, H, W = x.shape

        # Merge batch and time

        x = x.view(
            batch_size * seq_len,
            C,
            H,
            W
        )

        # CNN

        features = self.feature_extractor(x)

        features = self.avgpool(features)

        features = features.flatten(1)

        # shape:
        # (B*T, 576)

        features = features.view(
            batch_size,
            seq_len,
            -1
        )

        # shape:
        # (B, T, 576)

        lstm_out, (hidden, cell) = self.lstm(
            features
        )

        # last layer hidden state

        hidden = hidden[-1]

        logits = self.classifier(
            hidden
        )

        return logits