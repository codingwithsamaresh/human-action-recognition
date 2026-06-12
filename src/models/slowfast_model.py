"""
SlowFast Model

Input Shape:
(B, T, C, H, W)

Example:
(B, 16, 3, 224, 224)

Output:
(B, num_classes)
"""

import torch
import torch.nn as nn

from torchvision.models import (
    resnet18
)


class SlowPathway(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = resnet18(
            weights=None
        )

        self.features = nn.Sequential(
            *list(backbone.children())[:-1]
        )

    def forward(self, x):

        b, t, c, h, w = x.shape

        x = x.reshape(
            b * t,
            c,
            h,
            w
        )

        x = self.features(x)

        x = x.flatten(1)

        x = x.reshape(
            b,
            t,
            -1
        )

        x = x.mean(dim=1)

        return x


class FastPathway(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = resnet18(
            weights=None
        )

        self.features = nn.Sequential(
            *list(backbone.children())[:-1]
        )

    def forward(self, x):

        b, t, c, h, w = x.shape

        x = x.reshape(
            b * t,
            c,
            h,
            w
        )

        x = self.features(x)

        x = x.flatten(1)

        x = x.reshape(
            b,
            t,
            -1
        )

        x = x.mean(dim=1)

        return x


class SlowFastModel(nn.Module):

    def __init__(
        self,
        num_classes,
        alpha=4
    ):

        super().__init__()

        self.alpha = alpha

        self.slow_path = SlowPathway()

        self.fast_path = FastPathway()

        self.classifier = nn.Sequential(

            nn.Linear(
                512 + 512,
                512
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.5
            ),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(
        self,
        x
    ):
        """
        x:
        (B,T,C,H,W)
        """

        slow_frames = x[
            :,
            ::self.alpha,
            :,
            :,
            :
        ]

        fast_frames = x

        slow_feat = (
            self.slow_path(
                slow_frames
            )
        )

        fast_feat = (
            self.fast_path(
                fast_frames
            )
        )

        features = torch.cat(
            [
                slow_feat,
                fast_feat
            ],
            dim=1
        )

        logits = self.classifier(
            features
        )

        return logits