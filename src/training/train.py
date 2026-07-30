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

    # ==================================================
    # Configuration
    # ==================================================

    config = ConfigLoader.load(
        "configs/colab_config.yaml"
    )

    # ==================================================
    # Device
    # ==================================================

    device = get_device()

    print(
        "\n========================================"
    )

    print(
        f"Using Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA Version: "
            f"{torch.version.cuda}"
        )

    print(
        "========================================\n"
    )

    # ==================================================
    # Datasets
    # ==================================================

    print(
        "Loading training dataset..."
    )

    train_dataset = (
        ActionSequenceDataset(
            sequence_root=
            config.dataset.train_dir,

            transform=
            get_train_transforms()
        )
    )

    print(
        "Loading validation dataset..."
    )

    val_dataset = (
        ActionSequenceDataset(
            sequence_root=
            config.dataset.val_dir,

            transform=
            get_val_transforms()
        )
    )

    # ==================================================
    # Dataset Statistics
    # ==================================================

    train_samples = len(
        train_dataset
    )

    val_samples = len(
        val_dataset
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
        "\n========================================"
    )

    print(
        f"Train Samples     : {train_samples}"
    )

    print(
        f"Validation Samples: {val_samples}"
    )

    print(
        f"Number of Classes : {num_classes}"
    )

    print(
        "========================================"
    )

    # ==================================================
    # Verify Train/Validation Classes
    # ==================================================

    val_class_names = (
        val_dataset
        .get_class_names()
    )

    if class_names != val_class_names:

        raise RuntimeError(
            "Train and validation datasets "
            "do not contain the same classes."
        )

    print(
        "\nDataset classes verified."
    )

    # ==================================================
    # DataLoaders
    # ==================================================

    batch_size = (
        config.training.batch_size
    )

    num_workers = (
        config.dataset.num_workers
    )

    pin_memory = (
        torch.cuda.is_available()
    )

    print(
        "\nCreating DataLoaders..."
    )

    train_loader = DataLoader(
        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers,

        pin_memory=pin_memory,

        persistent_workers=(
            num_workers > 0
        )
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=pin_memory,

        persistent_workers=(
            num_workers > 0
        )
    )

    print(
        f"Train batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Val batches: "
        f"{len(val_loader)}"
    )

    # ==================================================
    # Model
    # ==================================================

    print(
        "\nCreating CNN-LSTM model..."
    )

    model = CNNLSTMBaseline(
        num_classes=num_classes,

        hidden_size=
        config.model.hidden_size,

        num_layers=
        config.model.num_layers,

        dropout=
        config.model.dropout
    )

    model = model.to(
        device
    )

    print(
        "Model loaded successfully."
    )

    # ==================================================
    # Loss Function
    # ==================================================

    criterion = (
        get_cross_entropy_loss()
    )

    # ==================================================
    # Optimizer
    # ==================================================

    optimizer_name = (
        config.training.optimizer
        .lower()
    )

    if optimizer_name == "adam":

        optimizer = torch.optim.Adam(
            model.parameters(),

            lr=
            config.training.learning_rate,

            weight_decay=
            config.training.weight_decay
        )

    elif optimizer_name == "adamw":

        optimizer = torch.optim.AdamW(
            model.parameters(),

            lr=
            config.training.learning_rate,

            weight_decay=
            config.training.weight_decay
        )

    else:

        raise ValueError(
            f"Unsupported optimizer: "
            f"{optimizer_name}"
        )

    # ==================================================
    # Learning Rate Scheduler
    # ==================================================

    scheduler = None

    scheduler_name = (
        config.training.scheduler
        .lower()
    )

    epochs = (
        config.training.epochs
    )

    if scheduler_name == "cosine":

        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs
            )
        )

        print(
            "\nUsing CosineAnnealingLR scheduler."
        )

    elif scheduler_name in (
        "none",
        ""
    ):

        print(
            "\nLearning-rate scheduler disabled."
        )

    else:

        raise ValueError(
            f"Unsupported scheduler: "
            f"{scheduler_name}"
        )

    # ==================================================
    # Checkpoint Directory
    # ==================================================

    checkpoint_dir = (
        config.checkpoint.save_dir
    )

    print(
        "\nCheckpoint directory:"
    )

    print(
        checkpoint_dir
    )

    # ==================================================
    # Trainer
    # ==================================================

    trainer = Trainer(
        model=model,

        criterion=criterion,

        optimizer=optimizer,

        scheduler=scheduler,

        device=device,

        checkpoint_dir=
        checkpoint_dir
    )

    # ==================================================
    # Training
    # ==================================================

    print(
        "\n========================================"
    )

    print(
        f"Starting Training "
        f"for {epochs} epochs"
    )

    print(
        "========================================\n"
    )

    trainer.fit(
        train_loader=train_loader,

        val_loader=val_loader,

        epochs=epochs
    )

    # ==================================================
    # Completion
    # ==================================================

    print(
        "\n========================================"
    )

    print(
        "Training Complete."
    )

    print(
        f"Best checkpoint:"
    )

    print(
        f"{checkpoint_dir}/best_model.pth"
    )

    print(
        f"\nLast checkpoint:"
    )

    print(
        f"{checkpoint_dir}/last_model.pth"
    )

    print(
        "========================================\n"
    )


if __name__ == "__main__":

    main()