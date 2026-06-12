import torch

from src.models.slowfast_model import (
    SlowFastModel
)


def main():

    model = SlowFastModel(
        num_classes=101
    )

    x = torch.randn(
        2,
        16,
        3,
        224,
        224
    )

    y = model(x)

    print(
        "Output Shape:",
        y.shape
    )


if __name__ == "__main__":
    main()