"""
Main Training Script

Trains the CNN-LSTM baseline model
using train/validation splits.
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

from src.models.cnn_lstm_baseline import (
    CNNLSTMBaseline
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

from src.utils.config_loader import (
    ConfigLoader
)


def main():

    # -------------------------
    # Config
    # -------------------------

    config = ConfigLoader.load(
        "configs/colab_config.yaml"
    )

    # -------------------------
    # Device
    # -------------------------

    device = get_device()

    print(
        f"\nUsing Device: {device}\n"
    )

    # -------------------------
    # Datasets
    # -------------------------

    train_dataset = (
        ActionSequenceDataset(
            sequence_root=
            config.dataset.train_dir,

            transform=
            get_train_transforms()
        )
    )

    val_dataset = (
        ActionSequenceDataset(
            sequence_root=
            config.dataset.val_dir,

            transform=
            get_val_transforms()
        )
    )

    # -------------------------
    # Dataset Stats
    # -------------------------

    print(
        f"Train Samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation Samples: "
        f"{len(val_dataset)}"
    )

    class_names = (
        train_dataset
        .get_class_names()
    )

    num_classes = (
        train_dataset
        .get_num_classes()
    )

    print(
        f"\nNumber of Classes: "
        f"{num_classes}"
    )

    print("\nClasses:")

    for idx, class_name in enumerate(
        class_names
    ):
        print(
            f"{idx}: {class_name}"
        )

    print()

    # -------------------------
    # DataLoaders
    # -------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=
        config.dataset.num_workers,
        pin_memory=
        torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=
        config.dataset.num_workers,
        pin_memory=
        torch.cuda.is_available()
    )

    # -------------------------
    # Model
    # -------------------------

    model = CNNLSTMBaseline(
        num_classes=num_classes,

        hidden_size=
        config.model.hidden_size,

        num_layers=
        config.model.num_layers,

        dropout=
        config.model.dropout
    )

    model.to(device)

    print(
        "Model Loaded Successfully\n"
    )

    # -------------------------
    # Loss Function
    # -------------------------

    criterion = (
        get_cross_entropy_loss()
    )

    # -------------------------
    # Optimizer
    # -------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=
        config.training.learning_rate,
        weight_decay=
        config.training.weight_decay
    )

    # -------------------------
    # Checkpoint Directory
    # -------------------------

    checkpoint_dir = (
        "/content/drive/MyDrive/"
        "human_action_recognition/"
        "weights/checkpoints"
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
        checkpoint_dir
    )

    # -------------------------
    # Train
    # -------------------------

    epochs = (
        config.training.epochs
    )

    print(
        f"Starting Training "
        f"for {epochs} epochs...\n"
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs
    )

    print(
        "\nTraining Complete."
    )

    print(
        "\nBest model saved to:"
    )

    print(
        checkpoint_dir +
        "/best_model.pth\n"
    )


if __name__ == "__main__":
    main()