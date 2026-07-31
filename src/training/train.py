"""
Main Training Script

Trains the CNN-LSTM baseline model
using train/validation splits.

Optimized for Google Colab / NVIDIA T4:

* Mixed precision
* Persistent DataLoader workers
* Pinned memory
* Non-blocking transfers
* Optional balanced subset per epoch
* Checkpoint resume
  """

import random

import torch

from torch.utils.data import (
DataLoader
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

def set_seed(seed):

```
random.seed(seed)

torch.manual_seed(seed)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        seed
    )
```

def create_balanced_sampler(
dataset,
samples_per_epoch
):
"""
Creates a balanced sampler across
all classes.

```
Every class receives approximately
the same number of samples per epoch.

Sampling is with replacement, allowing
us to reduce the number of samples
processed in each epoch without
dropping any of the 101 classes.
"""

class_indices = (
    dataset.get_class_indices()
)

num_classes = (
    dataset.get_num_classes()
)

if samples_per_epoch <= 0:

    raise ValueError(
        "samples_per_epoch must "
        "be greater than zero."
    )

# --------------------------------------------------
# Equal target count per class
# --------------------------------------------------

base_count = (
    samples_per_epoch //
    num_classes
)

remainder = (
    samples_per_epoch %
    num_classes
)

target_counts = []

for class_index in range(
    num_classes
):

    count = base_count

    if class_index < remainder:

        count += 1

    target_counts.append(
        count
    )

# --------------------------------------------------
# Build sampled indices
# --------------------------------------------------

sampled_indices = []

for class_index in range(
    num_classes
):

    indices = class_indices[
        class_index
    ]

    if not indices:

        continue

    count = target_counts[
        class_index
    ]

    selected = torch.randint(
        low=0,
        high=len(indices),
        size=(count,)
    ).tolist()

    sampled_indices.extend(
        indices[index]
        for index in selected
    )

# --------------------------------------------------
# Shuffle sampled indices
# --------------------------------------------------

random.shuffle(
    sampled_indices
)

# --------------------------------------------------
# Convert to weights
#
# WeightedRandomSampler is not necessary
# here because we already constructed a
# balanced index list.
# --------------------------------------------------

return sampled_indices
```

def main():

```
# ==================================================
# Configuration
# ==================================================

config = ConfigLoader.load(
    "configs/colab_config.yaml"
)

seed = getattr(
    config,
    "seed",
    42
)

set_seed(seed)

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
# Dataset
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
    f"Train Samples     : "
    f"{train_samples}"
)

print(
    f"Validation Samples: "
    f"{val_samples}"
)

print(
    f"Number of Classes : "
    f"{num_classes}"
)

print(
    "========================================"
)

# ==================================================
# Verify Classes
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
# Training Sample Count
# ==================================================

samples_per_epoch = getattr(
    config.training,
    "samples_per_epoch",
    train_samples
)

if samples_per_epoch > train_samples:

    samples_per_epoch = (
        train_samples
    )

print(
    "\nTraining samples per epoch:"
)

print(
    samples_per_epoch
)

# ==================================================
# Build Balanced Sample List
# ==================================================

if samples_per_epoch < train_samples:

    print(
        "\nUsing balanced subset sampling."
    )

    sampled_indices = (
        create_balanced_sampler(
            train_dataset,
            samples_per_epoch
        )
    )

    train_sampler = (
        torch.utils.data.SubsetRandomSampler(
            sampled_indices
        )
    )

    train_shuffle = False

else:

    print(
        "\nUsing the complete training dataset."
    )

    train_sampler = None
    train_shuffle = True

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

    shuffle=train_shuffle,

    sampler=train_sampler,

    num_workers=num_workers,

    pin_memory=pin_memory,

    persistent_workers=(
        num_workers > 0
    ),

    prefetch_factor=(
        2
        if num_workers > 0
        else None
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
    ),

    prefetch_factor=(
        2
        if num_workers > 0
        else None
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
# Loss
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
# Scheduler
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
# AMP
# ==================================================

use_amp = getattr(
    config.training,
    "mixed_precision",
    True
)

# ==================================================
# Resume
# ==================================================

resume_checkpoint = getattr(
    config.training,
    "resume_checkpoint",
    None
)

if (
    resume_checkpoint
    in (
        "",
        "none",
        "None"
    )
):

    resume_checkpoint = None

if resume_checkpoint is not None:

    print(
        "\nResume checkpoint:"
    )

    print(
        resume_checkpoint
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
    checkpoint_dir,

    use_amp=use_amp
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

    epochs=epochs,

    resume_checkpoint=
    resume_checkpoint
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
    "Best checkpoint:"
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
```

if **name** == "**main**":

```
main()
```
