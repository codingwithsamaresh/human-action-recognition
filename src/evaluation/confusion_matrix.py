"""
Confusion Matrix Generator

Loads trained model,
runs inference on dataset,
builds confusion matrix,
and saves visualization.

Output:
outputs/visualizations/confusion_matrix.png
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from src.data.dataset import ActionSequenceDataset
from src.models.cnn_lstm_baseline import CNNLSTMBaseline
from src.utils.device import get_device
from src.utils.config_loader import ConfigLoader


class ConfusionMatrixEvaluator:

    def __init__(
        self,
        checkpoint_path,
        dataset_dir,
        output_path="outputs/visualizations/confusion_matrix.png",
        batch_size=8,
        image_size=224
    ):

        self.device = get_device()

        self.dataset = ActionSequenceDataset(
            sequence_root=dataset_dir,
            image_size=image_size
        )

        self.class_names = (
            self.dataset.get_class_names()
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        self.model = CNNLSTMBaseline(
            num_classes=self.dataset.get_num_classes(),
            pretrained=False
        )

        self.output_path = Path(output_path)

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self._load_checkpoint(
            checkpoint_path
        )

    def _load_checkpoint(
        self,
        checkpoint_path
    ):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

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

        self.model.to(self.device)

        self.model.eval()

        print(
            f"Loaded checkpoint from: "
            f"{checkpoint_path}"
        )

    @torch.no_grad()
    def generate(self):

        all_preds = []
        all_targets = []

        for frames, targets in self.dataloader:

            frames = frames.to(
                self.device
            )

            logits = self.model(
                frames
            )

            preds = torch.argmax(
                logits,
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_targets.extend(
                targets.numpy()
            )

        cm = confusion_matrix(
            all_targets,
            all_preds
        )

        plt.figure(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")

        plt.tight_layout()

        plt.savefig(
            self.output_path,
            dpi=300
        )

        plt.close()

        print(
            f"Saved confusion matrix to:\n"
            f"{self.output_path}"
        )

        return cm


def main():

    config = ConfigLoader.load(
        "configs/colab_config.yaml"
    )

    evaluator = ConfusionMatrixEvaluator(
        checkpoint_path=(
            f"{config.checkpoint.save_dir}/best_model.pth"
        ),
        dataset_dir=config.dataset.test_dir,
        batch_size=config.training.batch_size,
        image_size=config.dataset.image_size
    )

    evaluator.generate()


if __name__ == "__main__":
    main()