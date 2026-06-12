"""
Train SlowFast Model

Uses:

data/train/
data/val/

Outputs:

weights/checkpoints/slowfast_best.pth
"""

import torch

from torch.utils.data import DataLoader

from src.data.dataset import (
    ActionSequenceDataset
)

from src.data.augmentations import (
    get_train_transforms,
    get_val_transforms
)

from src.models.slowfast_model import (
    SlowFastModel
)

from src.training.losses import (
    get_cross_entropy_loss
)

from src.training.trainer import (
    Trainer
)

from src.utils.device import (
    get_device
)


def main():

    # -------------------------
    # Device
    # -------------------------

    device = get_device()

    print(
        f"\nUsing Device: {device}\n"
    )

    # -------------------------
    # Dataset
    # -------------------------

    train_dataset = (
        ActionSequenceDataset(
            sequence_root="data/train",
            transform=get_train_transforms()
        )
    )

    val_dataset = (
        ActionSequenceDataset(
            sequence_root="data/val",
            transform=get_val_transforms()
        )
    )

    num_classes = (
        train_dataset
        .get_num_classes()
    )

    class_names = (
        train_dataset
        .get_class_names()
    )

    print(
        f"Train Samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation Samples: "
        f"{len(val_dataset)}"
    )

    print(
        f"Classes: "
        f"{num_classes}"
    )

    print()

    for idx, name in enumerate(
        class_names
    ):
        print(
            f"{idx}: {name}"
        )

    # -------------------------
    # DataLoader
    # -------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    # -------------------------
    # Model
    # -------------------------

    model = SlowFastModel(
        num_classes=num_classes,
        alpha=4
    )

    model.to(device)

    print(
        "\nSlowFast Model Loaded\n"
    )

    # -------------------------
    # Loss
    # -------------------------

    criterion = (
        get_cross_entropy_loss()
    )

    # -------------------------
    # Optimizer
    # -------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    # -------------------------
    # Trainer
    # -------------------------

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=
        "weights/checkpoints"
    )

    # -------------------------
    # Train
    # -------------------------

    epochs = 10

    print(
        f"\nTraining SlowFast "
        f"for {epochs} epochs...\n"
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs
    )

    print(
        "\nTraining Finished."
    )


if __name__ == "__main__":
    main()