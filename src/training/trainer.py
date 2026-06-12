from pathlib import Path

import torch
from tqdm import tqdm

from src.training.metrics import (
    top1_accuracy
)

from src.utils.logger import (
    get_logger
)


class Trainer:

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        checkpoint_dir="weights/checkpoints"
    ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

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

        self.best_val_loss = float("inf")

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
                self.device
            )

            labels = labels.to(
                self.device
            )

            self.optimizer.zero_grad()

            outputs = self.model(
                frames
            )

            loss = self.criterion(
                outputs,
                labels
            )

            loss.backward()

            self.optimizer.step()

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
                self.device
            )

            labels = labels.to(
                self.device
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

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    self.model.state_dict(),
                "optimizer_state_dict":
                    self.optimizer.state_dict(),
                "val_loss": val_loss
            },
            checkpoint_path
        )

        self.logger.info(
            f"Checkpoint saved: "
            f"{checkpoint_path}"
        )

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

    def fit(
        self,
        train_loader,
        val_loader,
        epochs
    ):

        try:

            for epoch in range(
                epochs
            ):

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

                self.logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )

                self.save_best_checkpoint(
                    epoch,
                    val_loss
                )

            self.save_checkpoint(
                epoch=epochs,
                val_loss=val_loss,
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
                epoch=epoch,
                val_loss=val_loss,
                filename="interrupted_model.pth"
            )

            self.logger.info(
                "Interrupted checkpoint saved."
            )