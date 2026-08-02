from pathlib import Path

import torch
from tqdm import tqdm

from src.training.metrics import top1_accuracy
from src.utils.logger import get_logger


class Trainer:
    """
    Trainer for CNN-LSTM Human Action Recognition.

    Features
    --------
    • Mixed Precision (AMP)
    • Resume Training
    • Automatic Checkpoints
    • Best Model Saving
    • Last Model Saving
    • Interrupted Training Recovery
    • Learning Rate Scheduler
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        checkpoint_dir="weights/checkpoints",
        scheduler=None,
        mixed_precision=True,
        resume_checkpoint=None,
        samples_per_epoch=None
    ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.logger = get_logger("trainer")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.best_val_loss = float("inf")
        self.start_epoch = 0

        self.samples_per_epoch = samples_per_epoch

        # ----------------------------------
        # Automatic Mixed Precision
        # ----------------------------------

        self.use_amp = (
            mixed_precision
            and torch.cuda.is_available()
        )

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp
        )

        # ----------------------------------
        # Resume Training
        # ----------------------------------

        if resume_checkpoint is not None:

            self.load_checkpoint(
                resume_checkpoint
            )

    # ==================================================
    # Load Checkpoint
    # ==================================================

    def load_checkpoint(
        self,
        checkpoint_path
    ):

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found:\n"
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
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

        self.start_epoch = (
            checkpoint["epoch"]
        )

        self.best_val_loss = (
            checkpoint.get(
                "best_val_loss",
                float("inf")
            )
        )

        self.logger.info(
            f"Resumed from checkpoint: "
            f"{checkpoint_path}"
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
        running_acc = 0.0

        total_samples = 0

        progress_bar = tqdm(
            dataloader,
            desc="Training",
            leave=False
        )

        for frames, labels in progress_bar:

            # ------------------------------------------
            # Stop early if samples_per_epoch reached
            # ------------------------------------------

            if (
                self.samples_per_epoch is not None
                and
                total_samples >= self.samples_per_epoch
            ):
                break

            frames = frames.to(
                self.device,
                non_blocking=True
            )

            labels = labels.to(
                self.device,
                non_blocking=True
            )

            batch_size = labels.size(0)

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------
            # Mixed Precision Forward Pass
            # ------------------------------------------

            with torch.cuda.amp.autocast(
                enabled=self.use_amp
            ):

                outputs = self.model(
                    frames
                )

                loss = self.criterion(
                    outputs,
                    labels
                )

            # ------------------------------------------
            # Backward
            # ------------------------------------------

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

            # ------------------------------------------
            # Metrics
            # ------------------------------------------

            acc = top1_accuracy(
                outputs,
                labels
            )

            running_loss += loss.item()
            running_acc += acc

            total_samples += batch_size

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{acc:.4f}",
                samples=total_samples
            )

        num_batches = max(
            1,
            progress_bar.n
        )

        epoch_loss = (
            running_loss /
            num_batches
        )

        epoch_acc = (
            running_acc /
            num_batches
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

            with torch.cuda.amp.autocast(
                enabled=self.use_amp
            ):

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

        num_batches = max(
            1,
            len(dataloader)
        )

        epoch_loss = (
            running_loss /
            num_batches
        )

        epoch_acc = (
            running_acc /
            num_batches
        )

        return (
            epoch_loss,
            epoch_acc
        )

    # ==================================================
    # Save Checkpoint
    # ==================================================

    def save_checkpoint(
        self,
        epoch,
        val_loss,
        filename
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

            "best_val_loss":
                self.best_val_loss,

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
    # Save Best Model
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

            self.logger.info(
                f"New best validation loss: "
                f"{val_loss:.4f}"
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

        last_val_loss = self.best_val_loss

        try:

            for epoch in range(
                self.start_epoch,
                epochs
            ):

                current_epoch = epoch + 1

                self.logger.info(
                    "-" * 70
                )

                self.logger.info(
                    f"Epoch {current_epoch}/{epochs}"
                )

                # ----------------------------------
                # Training
                # ----------------------------------

                train_loss, train_acc = (
                    self.train_one_epoch(
                        train_loader
                    )
                )

                # ----------------------------------
                # Validation
                # ----------------------------------

                # Validate every 5 epochs
                if (current_epoch % 5 == 0) or (current_epoch == epochs):

                    val_loss, val_acc = self.validate(
                        val_loader
                    )

                    last_val_loss = val_loss

                    self.save_best_checkpoint(
                        epoch=current_epoch,
                        val_loss=val_loss
                    )

                else:

                    val_loss = float("nan")
                    val_acc = float("nan")

                if self.scheduler is not None:
                    self.scheduler.step()

            # ----------------------------------
            # Save Final Model
            # ----------------------------------

            self.save_checkpoint(
                epoch=epochs,
                val_loss=last_val_loss,
                filename="last_model.pth"
            )

            self.logger.info(
                "-" * 70
            )

            self.logger.info(
                "Training completed successfully."
            )

            self.logger.info(
                f"Best Validation Loss: "
                f"{self.best_val_loss:.4f}"
            )

        except KeyboardInterrupt:

            self.logger.warning(
                "Training interrupted by user."
            )

            self.save_checkpoint(
                epoch=current_epoch,
                val_loss=last_val_loss,
                filename="interrupted_model.pth"
            )

            self.logger.info(
                "Interrupted checkpoint saved."
            )

        except Exception as error:

            self.logger.exception(
                error
            )

            self.save_checkpoint(
                epoch=current_epoch,
                val_loss=last_val_loss,
                filename="crashed_model.pth"
            )

            raise