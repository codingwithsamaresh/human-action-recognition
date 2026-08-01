"""
Evaluation Script

Evaluates a trained CNN-LSTM model on the
test dataset.

Reports:
- Test Loss
- Top-1 Accuracy
- Top-5 Accuracy
- Precision
- Recall
- F1 Score
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ActionSequenceDataset
from src.data.augmentations import get_val_transforms

from src.models.cnn_lstm_baseline import CNNLSTMBaseline

from src.training.losses import get_cross_entropy_loss

from src.training.metrics import (
    top1_accuracy,
    topk_accuracy,
    precision_recall_f1
)

from src.utils.device import get_device
from src.utils.config_loader import ConfigLoader


class Evaluator:

    def __init__(self):

        self.config = ConfigLoader.load(
            "configs/colab_config.yaml"
        )

        self.device = get_device()

        print(f"\nUsing Device: {self.device}")

        # ------------------------------------------
        # Dataset
        # ------------------------------------------

        print("\nLoading test dataset...")

        self.dataset = ActionSequenceDataset(

            sequence_root=self.config.dataset.test_dir,

            transform=get_val_transforms()

        )

        self.dataloader = DataLoader(

            self.dataset,

            batch_size=self.config.training.batch_size,

            shuffle=False,

            num_workers=self.config.dataset.num_workers,

            pin_memory=torch.cuda.is_available()

        )

        # ------------------------------------------
        # Model
        # ------------------------------------------

        print("Creating model...")

        self.model = CNNLSTMBaseline(

            num_classes=self.dataset.get_num_classes(),

            hidden_size=self.config.model.hidden_size,

            num_layers=self.config.model.num_layers,

            dropout=self.config.model.dropout

        )

        self.model.to(self.device)

        self.load_checkpoint()

        self.criterion = get_cross_entropy_loss()

    # ==================================================
    # Load Checkpoint
    # ==================================================

    def load_checkpoint(self):

        checkpoint_path = Path(

            self.config.checkpoint.save_dir

        ) / "best_model.pth"

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

        self.model.eval()

        print(

            f"Loaded checkpoint:\n"

            f"{checkpoint_path}"

        )

    # ==================================================
    # Evaluation
    # ==================================================

    @torch.no_grad()

    def evaluate(self):

        self.model.eval()

        running_loss = 0.0

        all_outputs = []

        all_labels = []

        for frames, labels in self.dataloader:

            frames = frames.to(self.device)

            labels = labels.to(self.device)

            outputs = self.model(frames)

            loss = self.criterion(

                outputs,

                labels

            )

            running_loss += loss.item()

            all_outputs.append(

                outputs.cpu()

            )

            all_labels.append(

                labels.cpu()

            )

        outputs = torch.cat(

            all_outputs,

            dim=0

        )

        labels = torch.cat(

            all_labels,

            dim=0

        )

        avg_loss = (

            running_loss /

            len(self.dataloader)

        )

        top1 = top1_accuracy(

            outputs,

            labels

        )

        top5 = topk_accuracy(

            outputs,

            labels,

            k=self.config.evaluation.top_k

        )

        metrics = precision_recall_f1(

            outputs,

            labels

        )

        print("\n========================================")

        print("Evaluation Results")

        print("========================================")

        print(f"Loss          : {avg_loss:.4f}")

        print(f"Top-1 Acc     : {top1:.4f}")

        print(f"Top-5 Acc     : {top5:.4f}")

        print(f"Precision     : {metrics['precision']:.4f}")

        print(f"Recall        : {metrics['recall']:.4f}")

        print(f"F1 Score      : {metrics['f1']:.4f}")

        print("========================================")


def main():

    evaluator = Evaluator()

    evaluator.evaluate()


if __name__ == "__main__":

    main()