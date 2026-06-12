"""
Evaluation Script

Evaluates a trained HAR model on the
test dataset and reports:

- Loss
- Top-1 Accuracy
- Top-K Accuracy
- Precision
- Recall
- F1 Score
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ActionSequenceDataset
from src.models.cnn_lstm_baseline import CNNLSTMBaseline

from src.training.losses import (
    get_cross_entropy_loss
)

from src.training.metrics import (
    top1_accuracy,
    topk_accuracy,
    precision_recall_f1
)

from src.utils.device import get_device


class Evaluator:

    def __init__(
        self,
        checkpoint_path,
        test_dir,
        batch_size=8,
        image_size=224
    ):

        self.device = get_device()

        # -------------------------
        # Dataset
        # -------------------------

        self.dataset = ActionSequenceDataset(
            sequence_root=test_dir,
            image_size=image_size
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        # -------------------------
        # Model
        # -------------------------

        self.model = CNNLSTMBaseline(
            num_classes=self.dataset.get_num_classes(),
            pretrained=False
        )

        self._load_checkpoint(
            checkpoint_path
        )

        # -------------------------
        # Loss
        # -------------------------

        self.criterion = (
            get_cross_entropy_loss()
        )

    def _load_checkpoint(
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

        # Training checkpoint format

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:

            self.model.load_state_dict(
                checkpoint
            )

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            f"Loaded checkpoint from: "
            f"{checkpoint_path}"
        )

    @torch.no_grad()
    def evaluate(self):

        total_loss = 0.0

        all_logits = []
        all_targets = []

        for frames, targets in self.dataloader:

            frames = frames.to(
                self.device
            )

            targets = targets.to(
                self.device
            )

            logits = self.model(
                frames
            )

            loss = self.criterion(
                logits,
                targets
            )

            total_loss += loss.item()

            all_logits.append(
                logits.cpu()
            )

            all_targets.append(
                targets.cpu()
            )

        all_logits = torch.cat(
            all_logits,
            dim=0
        )

        all_targets = torch.cat(
            all_targets,
            dim=0
        )

        avg_loss = (
            total_loss /
            len(self.dataloader)
        )

        top1 = top1_accuracy(
            all_logits,
            all_targets
        )

        top5 = topk_accuracy(
            all_logits,
            all_targets,
            k=5
        )

        prf = precision_recall_f1(
            all_logits,
            all_targets
        )

        results = {
            "loss": avg_loss,
            "top1_accuracy": top1,
            "top5_accuracy": top5,
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"]
        }

        return results


def main():

    evaluator = Evaluator(
        checkpoint_path=
        "weights/checkpoints/best_model.pth",

        test_dir=
        "data/test"
    )

    results = evaluator.evaluate()

    print("\nEvaluation Results")
    print("-" * 40)

    for key, value in results.items():

        print(
            f"{key}: "
            f"{value:.4f}"
        )


if __name__ == "__main__":
    main()