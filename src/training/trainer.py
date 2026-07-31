from pathlib import Path

import torch
from tqdm import tqdm

from src.training.metrics import top1_accuracy
from src.utils.logger import get_logger

class Trainer:
"""
Training and validation manager for HAR models.

```
Features:
- Mixed precision training on CUDA
- Non-blocking GPU transfers
- Learning-rate scheduling
- Best-model checkpointing
- Last-model checkpointing
- Resume-from-checkpoint support
- Interrupted-training checkpointing
"""

def __init__(
    self,
    model,
    criterion,
    optimizer,
    device,
    checkpoint_dir="weights/checkpoints",
    scheduler=None,
    use_amp=True
):

    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.device = device
    self.scheduler = scheduler

    self.logger = get_logger(
        "trainer"
    )

    self.checkpoint_dir = Path(
        checkpoint_dir
    )

    self.checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    self.best_val_loss = float(
        "inf"
    )

    # ==================================================
    # AMP
    # ==================================================

    self.use_amp = (
        use_amp
        and
        self.device.type == "cuda"
    )

    if self.use_amp:

        self.scaler = torch.amp.GradScaler(
            "cuda"
        )

        self.logger.info(
            "Automatic Mixed Precision enabled."
        )

    else:

        self.scaler = None

        self.logger.info(
            "Automatic Mixed Precision disabled."
        )

# ==================================================
# Forward Pass
# ==================================================

def _forward(
    self,
    frames,
    labels
):

    if self.use_amp:

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            outputs = self.model(
                frames
            )

            loss = self.criterion(
                outputs,
                labels
            )

    else:

        outputs = self.model(
            frames
        )

        loss = self.criterion(
            outputs,
            labels
        )

    return (
        outputs,
        loss
    )

# ==================================================
# Training
# ==================================================

def train_one_epoch(
    self,
    dataloader
):

    self.model.train()

    running_loss = 0.0
    running_correct = 0
    running_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False
    )

    for frames, labels in progress_bar:

        frames = frames.to(
            self.device,
            non_blocking=True
        )

        labels = labels.to(
            self.device,
            non_blocking=True
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        outputs, loss = self._forward(
            frames,
            labels
        )

        # ==================================================
        # Backward
        # ==================================================

        if self.use_amp:

            self.scaler.scale(
                loss
            ).backward()

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

        else:

            loss.backward()

            self.optimizer.step()

        # ==================================================
        # Metrics
        # ==================================================

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        correct = (
            predictions == labels
        ).sum().item()

        batch_size = (
            labels.size(0)
        )

        running_correct += correct
        running_samples += batch_size
        running_loss += (
            loss.item() *
            batch_size
        )

        current_accuracy = (
            correct /
            batch_size
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{current_accuracy:.4f}"
        )

    epoch_loss = (
        running_loss /
        running_samples
    )

    epoch_accuracy = (
        running_correct /
        running_samples
    )

    return (
        epoch_loss,
        epoch_accuracy
    )

# ==================================================
# Validation
# ==================================================

@torch.no_grad()
def validate(
    self,
    dataloader
):

    self.model.eval()

    running_loss = 0.0
    running_correct = 0
    running_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        leave=False
    )

    for frames, labels in progress_bar:

        frames = frames.to(
            self.device,
            non_blocking=True
        )

        labels = labels.to(
            self.device,
            non_blocking=True
        )

        if self.use_amp:

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = self.model(
                    frames
                )

                loss = self.criterion(
                    outputs,
                    labels
                )

        else:

            outputs = self.model(
                frames
            )

            loss = self.criterion(
                outputs,
                labels
            )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        correct = (
            predictions == labels
        ).sum().item()

        batch_size = (
            labels.size(0)
        )

        running_correct += correct
        running_samples += batch_size

        running_loss += (
            loss.item() *
            batch_size
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = (
        running_loss /
        running_samples
    )

    epoch_accuracy = (
        running_correct /
        running_samples
    )

    return (
        epoch_loss,
        epoch_accuracy
    )

# ==================================================
# Save Checkpoint
# ==================================================

def save_checkpoint(
    self,
    epoch,
    val_loss,
    filename="last_model.pth"
):

    checkpoint_path = (
        self.checkpoint_dir /
        filename
    )

    checkpoint = {

        "epoch":
            epoch,

        "model_state_dict":
            self.model.state_dict(),

        "optimizer_state_dict":
            self.optimizer.state_dict(),

        "val_loss":
            val_loss,

        "best_val_loss":
            self.best_val_loss
    }

    if self.scheduler is not None:

        checkpoint[
            "scheduler_state_dict"
        ] = (
            self.scheduler.state_dict()
        )

    if self.scaler is not None:

        checkpoint[
            "scaler_state_dict"
        ] = (
            self.scaler.state_dict()
        )

    torch.save(
        checkpoint,
        checkpoint_path
    )

    self.logger.info(
        f"Checkpoint saved: "
        f"{checkpoint_path}"
    )

# ==================================================
# Load Checkpoint
# ==================================================

def load_checkpoint(
    self,
    checkpoint_path
):

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=self.device
    )

    self.model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    if (
        "optimizer_state_dict"
        in checkpoint
    ):

        self.optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    if (
        self.scheduler is not None
        and
        "scheduler_state_dict"
        in checkpoint
    ):

        self.scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

    if (
        self.scaler is not None
        and
        "scaler_state_dict"
        in checkpoint
    ):

        self.scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )

    self.best_val_loss = checkpoint.get(
        "best_val_loss",
        checkpoint.get(
            "val_loss",
            float("inf")
        )
    )

    start_epoch = (
        checkpoint.get(
            "epoch",
            0
        )
    )

    self.logger.info(
        f"Checkpoint loaded: "
        f"{checkpoint_path}"
    )

    self.logger.info(
        f"Resuming from epoch "
        f"{start_epoch}"
    )

    return start_epoch

# ==================================================
# Best Checkpoint
# ==================================================

def save_best_checkpoint(
    self,
    epoch,
    val_loss
):

    if val_loss < self.best_val_loss:

        self.best_val_loss = val_loss

        self.save_checkpoint(
            epoch=epoch,
            val_loss=val_loss,
            filename="best_model.pth"
        )

# ==================================================
# Main Training Loop
# ==================================================

def fit(
    self,
    train_loader,
    val_loader,
    epochs,
    resume_checkpoint=None
):

    start_epoch = 0

    last_val_loss = float(
        "inf"
    )

    # ==================================================
    # Resume
    # ==================================================

    if resume_checkpoint is not None:

        start_epoch = (
            self.load_checkpoint(
                resume_checkpoint
            )
        )

        self.logger.info(
            f"Training will continue "
            f"from epoch {start_epoch + 1}."
        )

    # ==================================================
    # Training
    # ==================================================

    try:

        for epoch in range(
            start_epoch,
            epochs
        ):

            current_epoch = (
                epoch + 1
            )

            train_loss, train_acc = (
                self.train_one_epoch(
                    train_loader
                )
            )

            val_loss, val_acc = (
                self.validate(
                    val_loader
                )
            )

            last_val_loss = val_loss

            # ==================================================
            # Scheduler
            # ==================================================

            if self.scheduler is not None:

                self.scheduler.step()

            current_lr = (
                self.optimizer
                .param_groups[0]["lr"]
            )

            # ==================================================
            # Logging
            # ==================================================

            self.logger.info(
                f"Epoch "
                f"{current_epoch}/{epochs} | "
                f"Train Loss: "
                f"{train_loss:.4f} | "
                f"Train Acc: "
                f"{train_acc:.4f} | "
                f"Val Loss: "
                f"{val_loss:.4f} | "
                f"Val Acc: "
                f"{val_acc:.4f} | "
                f"LR: "
                f"{current_lr:.6f}"
            )

            # ==================================================
            # Save Best
            # ==================================================

            self.save_best_checkpoint(
                epoch=current_epoch,
                val_loss=val_loss
            )

            # ==================================================
            # Save Last
            # ==================================================

            self.save_checkpoint(
                epoch=current_epoch,
                val_loss=val_loss,
                filename="last_model.pth"
            )

        self.logger.info(
            "Training Complete."
        )

    except KeyboardInterrupt:

        self.logger.warning(
            "Training interrupted."
        )

        self.save_checkpoint(
            epoch=current_epoch,
            val_loss=last_val_loss,
            filename="interrupted_model.pth"
        )

        self.logger.info(
            "Interrupted checkpoint saved."
        )
```
