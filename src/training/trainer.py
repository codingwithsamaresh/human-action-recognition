from pathlib import Path

import torch
from tqdm import tqdm

from src.training.metrics import top1_accuracy
from src.utils.logger import get_logger


class Trainer:
    """
    Training and validation manager for HAR models.

    Handles:
    - Training loop
    - Validation loop
    - Learning-rate scheduling
    - Best-model checkpointing
    - Last-model checkpointing
    - Interrupted-training checkpointing
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        checkpoint_dir="weights/checkpoints",
        scheduler=None
    ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        self.logger = get_logger("trainer")

        self.checkpoint_dir = Path(
            checkpoint_dir
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.best_val_loss = float("inf")

    # ==================================================
    # Training
    # ==================================================

    def train_one_epoch(
        self,
        dataloader
    ):

        self.model.train()

        running_loss = 0.0
        running_acc = 0.0

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

            # ------------------------------
            # Clear gradients
            # ------------------------------

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------
            # Forward pass
            # ------------------------------

            outputs = self.model(
                frames
            )

            # ------------------------------
            # Loss
            # ------------------------------

            loss = self.criterion(
                outputs,
                labels
            )

            # ------------------------------
            # Backpropagation
            # ------------------------------

            loss.backward()

            self.optimizer.step()

            # ------------------------------
            # Metrics
            # ------------------------------

            acc = top1_accuracy(
                outputs,
                labels
            )

            running_loss += loss.item()
            running_acc += acc

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{acc:.4f}"
            )

        epoch_loss = (
            running_loss /
            len(dataloader)
        )

        epoch_acc = (
            running_acc /
            len(dataloader)
        )

        return (
            epoch_loss,
            epoch_acc
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
        running_acc = 0.0

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

            outputs = self.model(
                frames
            )

            loss = self.criterion(
                outputs,
                labels
            )

            acc = top1_accuracy(
                outputs,
                labels
            )

            running_loss += loss.item()
            running_acc += acc

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{acc:.4f}"
            )

        epoch_loss = (
            running_loss /
            len(dataloader)
        )

        epoch_acc = (
            running_acc /
            len(dataloader)
        )

        return (
            epoch_loss,
            epoch_acc
        )

    # ==================================================
    # Checkpoint
    # ==================================================

    def save_checkpoint(
        self,
        epoch,
        val_loss,
        filename="best_model.pth"
    ):

        checkpoint_path = (
            self.checkpoint_dir /
            filename
        )

        checkpoint = {
            "epoch": epoch,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "val_loss":
                val_loss
        }

        if self.scheduler is not None:

            checkpoint[
                "scheduler_state_dict"
            ] = self.scheduler.state_dict()

        torch.save(
            checkpoint,
            checkpoint_path
        )

        self.logger.info(
            f"Checkpoint saved: "
            f"{checkpoint_path}"
        )

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
    # Training Loop
    # ==================================================

    def fit(
        self,
        train_loader,
        val_loader,
        epochs
    ):

        last_epoch = 0
        last_val_loss = float("inf")

        try:

            for epoch in range(
                epochs
            ):

                last_epoch = epoch + 1

                # ------------------------------
                # Training
                # ------------------------------

                train_loss, train_acc = (
                    self.train_one_epoch(
                        train_loader
                    )
                )

                # ------------------------------
                # Validation
                # ------------------------------

                val_loss, val_acc = (
                    self.validate(
                        val_loader
                    )
                )

                last_val_loss = val_loss

                # ------------------------------
                # Scheduler
                # ------------------------------

                if self.scheduler is not None:

                    self.scheduler.step()

                # ------------------------------
                # Current learning rate
                # ------------------------------

                current_lr = (
                    self.optimizer
                    .param_groups[0]["lr"]
                )

                # ------------------------------
                # Logging
                # ------------------------------

                self.logger.info(
                    f"Epoch {last_epoch}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f} | "
                    f"LR: {current_lr:.6f}"
                )

                # ------------------------------
                # Save best model
                # ------------------------------

                self.save_best_checkpoint(
                    epoch=last_epoch,
                    val_loss=val_loss
                )

            # ------------------------------
            # Save final model
            # ------------------------------

            self.save_checkpoint(
                epoch=last_epoch,
                val_loss=last_val_loss,
                filename="last_model.pth"
            )

            self.logger.info(
                "Training Complete."
            )

        except KeyboardInterrupt:

            self.logger.warning(
                "Training interrupted by user."
            )

            self.save_checkpoint(
                epoch=last_epoch,
                val_loss=last_val_loss,
                filename="interrupted_model.pth"
            )

            self.logger.info(
                "Interrupted checkpoint saved."
            )