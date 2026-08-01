"""
Main Training Script

Trains the CNN-LSTM baseline model
using train/validation splits.

Optimized for Google Colab / NVIDIA T4

Features
--------
- Mixed Precision (AMP)
- Resume Training
- Balanced Subset Sampling
- Persistent Workers
- Fast DataLoader
- Automatic Checkpointing
"""

import random

import torch

from torch.utils.data import (
    DataLoader,
    SubsetRandomSampler
)

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


# ==================================================
# Reproducibility
# ==================================================

def set_seed(seed):

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ==================================================
# Balanced Sampler
# ==================================================

def create_balanced_sampler(
    dataset,
    samples_per_epoch
):
    """
    Creates a balanced subset of samples.

    Every class contributes approximately
    the same number of samples.
    """

    class_indices = dataset.get_class_indices()

    num_classes = dataset.get_num_classes()

    if samples_per_epoch <= 0:

        raise ValueError(
            "samples_per_epoch must be positive."
        )

    base = samples_per_epoch // num_classes

    remainder = samples_per_epoch % num_classes

    sampled_indices = []

    for class_idx in range(num_classes):

        indices = class_indices[class_idx]

        if len(indices) == 0:
            continue

        count = base

        if class_idx < remainder:
            count += 1

        selected = torch.randint(
            low=0,
            high=len(indices),
            size=(count,)
        ).tolist()

        sampled_indices.extend(
            indices[i]
            for i in selected
        )

    random.shuffle(sampled_indices)

    return sampled_indices


# ==================================================
# Main
# ==================================================

def main():

    config = ConfigLoader.load(
        "configs/colab_config.yaml"
    )

    set_seed(
        getattr(config, "seed", 42)
    )

    device = get_device()

    print("\n========================================")
    print(f"Using Device: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA Version: {torch.version.cuda}"
        )

    print("========================================\n")

        # ==================================================
    # Dataset
    # ==================================================

    print("Loading training dataset...")

    train_dataset = ActionSequenceDataset(
        sequence_root=config.dataset.train_dir,
        transform=get_train_transforms()
    )

    print("Loading validation dataset...")

    val_dataset = ActionSequenceDataset(
        sequence_root=config.dataset.val_dir,
        transform=get_val_transforms()
    )

    # ==================================================
    # Dataset Statistics
    # ==================================================

    train_samples = len(train_dataset)
    val_samples = len(val_dataset)

    num_classes = train_dataset.get_num_classes()

    class_names = train_dataset.get_class_names()

    print("\n========================================")

    print(f"Train Samples     : {train_samples}")

    print(f"Validation Samples: {val_samples}")

    print(f"Number of Classes : {num_classes}")

    print("========================================")

    if class_names != val_dataset.get_class_names():

        raise RuntimeError(
            "Train and validation classes do not match."
        )

    print("\nDataset classes verified.")

    # ==================================================
    # Samples Per Epoch
    # ==================================================

    samples_per_epoch = getattr(
        config.training,
        "samples_per_epoch",
        train_samples
    )

    if samples_per_epoch > train_samples:

        samples_per_epoch = train_samples

    print("\nTraining samples per epoch:")

    print(samples_per_epoch)

    # ==================================================
    # Balanced Sampling
    # ==================================================

    if samples_per_epoch < train_samples:

        print("\nUsing balanced subset sampling.")

        sampled_indices = create_balanced_sampler(
            train_dataset,
            samples_per_epoch
        )

        train_sampler = SubsetRandomSampler(
            sampled_indices
        )

        shuffle = False

    else:

        print("\nUsing complete training dataset.")

        train_sampler = None

        shuffle = True

    # ==================================================
    # DataLoaders
    # ==================================================

    batch_size = config.training.batch_size

    num_workers = config.dataset.num_workers

    pin_memory = torch.cuda.is_available()

    print("\nCreating DataLoaders...")

    train_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if train_sampler is not None:
        train_loader_kwargs["sampler"] = train_sampler

    if num_workers > 0:

        train_loader_kwargs[
            "persistent_workers"
        ] = True

        train_loader_kwargs[
            "prefetch_factor"
        ] = 2

    train_loader = DataLoader(
        train_dataset,
        **train_loader_kwargs
    )

    val_loader_kwargs = {

        "batch_size": batch_size,

        "shuffle": False,

        "num_workers": num_workers,

        "pin_memory": pin_memory

    }

    if num_workers > 0:

        val_loader_kwargs[
            "persistent_workers"
        ] = True

        val_loader_kwargs[
            "prefetch_factor"
        ] = 2

    val_loader = DataLoader(
        val_dataset,
        **val_loader_kwargs
    )

    print(
        f"Train batches: {len(train_loader)}"
    )

    print(
        f"Val batches: {len(val_loader)}"
    )

        # ==================================================
    # Model
    # ==================================================

    print("\nCreating CNN-LSTM model...")

    model = CNNLSTMBaseline(

        num_classes=num_classes,

        hidden_size=config.model.hidden_size,

        num_layers=config.model.num_layers,

        dropout=config.model.dropout

    )

    model = model.to(device)

    print("Model loaded successfully.")

    # ==================================================
    # Loss Function
    # ==================================================

    criterion = get_cross_entropy_loss()

    # ==================================================
    # Optimizer
    # ==================================================

    optimizer_name = (
        config.training.optimizer.lower()
    )

    if optimizer_name == "adam":

        optimizer = torch.optim.Adam(

            model.parameters(),

            lr=config.training.learning_rate,

            weight_decay=config.training.weight_decay

        )

    elif optimizer_name == "adamw":

        optimizer = torch.optim.AdamW(

            model.parameters(),

            lr=config.training.learning_rate,

            weight_decay=config.training.weight_decay

        )

    else:

        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}"
        )

    # ==================================================
    # Scheduler
    # ==================================================

    scheduler = None

    scheduler_name = (
        config.training.scheduler.lower()
    )

    epochs = config.training.epochs

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
            f"Unsupported scheduler: {scheduler_name}"
        )

    # ==================================================
    # Mixed Precision
    # ==================================================

    mixed_precision = getattr(
        config.training,
        "mixed_precision",
        True
    )

    if (
        mixed_precision
        and
        torch.cuda.is_available()
    ):

        print(
            "\nMixed Precision: ENABLED"
        )

    else:

        mixed_precision = False

        print(
            "\nMixed Precision: DISABLED"
        )

    # ==================================================
    # Resume Checkpoint
    # ==================================================

    resume_checkpoint = getattr(
        config.training,
        "resume_checkpoint",
        None
    )

    if resume_checkpoint in (
        "",
        "None",
        "none"
    ):

        resume_checkpoint = None

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

        checkpoint_dir=checkpoint_dir,

        mixed_precision=mixed_precision,

        resume_checkpoint=resume_checkpoint,

        samples_per_epoch=samples_per_epoch

    )

    # ==================================================
    # Training
    # ==================================================

    print(
        "\n========================================"
    )

    print(
        f"Starting Training for {epochs} epochs"
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
        "\nBest checkpoint:"
    )

    print(
        f"{checkpoint_dir}/best_model.pth"
    )

    print(
        "\nLast checkpoint:"
    )

    print(
        f"{checkpoint_dir}/last_model.pth"
    )

    print(
        "========================================\n"
    )


# ==================================================
# Entry Point
# ==================================================

if __name__ == "__main__":

    main()

    