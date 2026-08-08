"""
Confusion Matrix Generator

Loads the trained CNN-LSTM model,
runs inference on the test dataset,
generates a confusion matrix,
and saves the visualization.

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
        output_path=(
            "outputs/visualizations/"
            "confusion_matrix.png"
        ),
        batch_size=8,
        image_size=224
    ):

        self.device = get_device()

        # ---------------------------------
        # Dataset
        # ---------------------------------

        print("\nLoading test dataset...")

        self.dataset = ActionSequenceDataset(
            sequence_root=dataset_dir,
            image_size=image_size
        )

        self.class_names = (
            self.dataset.get_class_names()
        )

        self.num_classes = (
            self.dataset.get_num_classes()
        )

        print(
            f"Loaded {len(self.dataset)} sequences "
            f"from {self.num_classes} classes."
        )

        # ---------------------------------
        # DataLoader
        # ---------------------------------

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        # ---------------------------------
        # Model
        # ---------------------------------

        print("Creating model...")

        self.model = CNNLSTMBaseline(
            num_classes=self.num_classes,
            pretrained=False
        )

        # ---------------------------------
        # Output
        # ---------------------------------

        self.output_path = Path(
            output_path
        )

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # ---------------------------------
        # Load checkpoint
        # ---------------------------------

        self._load_checkpoint(
            checkpoint_path
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
                f"Checkpoint not found:\n"
                f"{checkpoint_path}"
            )

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

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            f"Loaded checkpoint from:\n"
            f"{checkpoint_path}"
        )

    @torch.no_grad()
    def generate(self):

        all_preds = []
        all_targets = []

        print("\nGenerating confusion matrix...")

        # ---------------------------------
        # Inference
        # ---------------------------------

        for frames, targets in self.dataloader:

            frames = frames.to(
                self.device
            )

            logits = self.model(
                frames
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_preds.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                targets.cpu().numpy()
            )

        if not all_preds:

            print(
                "No predictions were generated."
            )

            return None

        # ---------------------------------
        # Confusion Matrix
        # ---------------------------------

        labels = list(
            range(self.num_classes)
        )

        cm = confusion_matrix(
            all_targets,
            all_preds,
            labels=labels
        )

        # ---------------------------------
        # Plot
        # ---------------------------------

        plt.figure(
            figsize=(18, 16)
        )

        sns.heatmap(
            cm,
            annot=False,
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            square=True,
            cbar=True
        )

        plt.xlabel(
            "Predicted Label"
        )

        plt.ylabel(
            "True Label"
        )

        plt.title(
            "Confusion Matrix - CNN-LSTM"
        )

        plt.xticks(
            rotation=90,
            fontsize=6
        )

        plt.yticks(
            rotation=0,
            fontsize=6
        )

        plt.tight_layout()

        plt.savefig(
            self.output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"\nSaved confusion matrix to:\n"
            f"{self.output_path}"
        )

        return cm


def main():

    # ---------------------------------
    # Load Colab configuration
    # ---------------------------------

    config = ConfigLoader.load(
        "configs/colab_config.yaml"
    )

    # ---------------------------------
    # Checkpoint
    # ---------------------------------

    checkpoint_path = (
        Path(
            config.checkpoint.save_dir
        )
        / "best_model.pth"
    )

    # ---------------------------------
    # Generate confusion matrix
    # ---------------------------------

    evaluator = ConfusionMatrixEvaluator(
        checkpoint_path=checkpoint_path,
        dataset_dir=config.dataset.test_dir,
        output_path=(
            "outputs/visualizations/"
            "confusion_matrix.png"
        ),
        batch_size=config.training.batch_size,
        image_size=config.dataset.image_size
    )

    evaluator.generate()


if __name__ == "__main__":
    main()