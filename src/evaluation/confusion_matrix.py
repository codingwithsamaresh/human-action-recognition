"""
Confusion Matrix Generator

Generates a normalized confusion matrix
for the 101-class CNN-LSTM model.

Visualization strategy:
- Row-normalized confusion matrix
- No cell annotations
- Show every Nth class label
- High-resolution output
- Save raw and normalized matrices as CSV

Outputs:
outputs/visualizations/confusion_matrix.png
outputs/reports/confusion_matrix_raw.csv
outputs/reports/confusion_matrix_normalized.csv
"""

from pathlib import Path

import numpy as np

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
        image_size=224,
        label_step=5
    ):

        self.device = get_device()

        self.label_step = label_step

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
        # Output paths
        # ---------------------------------

        self.output_path = Path(
            output_path
        )

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.report_dir = Path(
            "outputs/reports"
        )

        self.report_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ---------------------------------
        # Load checkpoint
        # ---------------------------------

        self._load_checkpoint(
            checkpoint_path
        )

    # =====================================
    # Load checkpoint
    # =====================================

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

    # =====================================
    # Generate confusion matrix
    # =====================================

    @torch.no_grad()
    def generate(self):

        all_preds = []
        all_targets = []

        print(
            "\nGenerating confusion matrix..."
        )

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
        # Raw confusion matrix
        # ---------------------------------

        labels = np.arange(
            self.num_classes
        )

        cm = confusion_matrix(
            all_targets,
            all_preds,
            labels=labels
        )

        # ---------------------------------
        # Row normalization
        # ---------------------------------

        row_sums = cm.sum(
            axis=1,
            keepdims=True
        )

        cm_normalized = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(
                cm,
                dtype=float
            ),
            where=row_sums != 0
        )

        # ---------------------------------
        # Save raw matrix
        # ---------------------------------

        raw_csv = (
            self.report_dir
            / "confusion_matrix_raw.csv"
        )

        np.savetxt(
            raw_csv,
            cm,
            delimiter=",",
            fmt="%d"
        )

        # ---------------------------------
        # Save normalized matrix
        # ---------------------------------

        normalized_csv = (
            self.report_dir
            / "confusion_matrix_normalized.csv"
        )

        np.savetxt(
            normalized_csv,
            cm_normalized,
            delimiter=",",
            fmt="%.6f"
        )

        # ---------------------------------
        # Plot
        # ---------------------------------

        plt.figure(
            figsize=(16, 14)
        )

        ax = sns.heatmap(
            cm_normalized,
            cmap="Blues",
            vmin=0,
            vmax=1,
            square=True,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={
                "label": "Normalized Frequency"
            }
        )

        # ---------------------------------
        # Select labels
        # ---------------------------------

        tick_positions = np.arange(
            0,
            self.num_classes,
            self.label_step
        )

        tick_labels = [
            self.class_names[i]
            for i in tick_positions
        ]

        ax.set_xticks(
            tick_positions + 0.5
        )

        ax.set_yticks(
            tick_positions + 0.5
        )

        ax.set_xticklabels(
            tick_labels,
            rotation=90,
            fontsize=8
        )

        ax.set_yticklabels(
            tick_labels,
            rotation=0,
            fontsize=8
        )

        # ---------------------------------
        # Labels
        # ---------------------------------

        plt.xlabel(
            "Predicted Label",
            fontsize=13
        )

        plt.ylabel(
            "True Label",
            fontsize=13
        )

        plt.title(
            "Normalized Confusion Matrix - "
            "CNN-LSTM (101-Class UCF101)",
            fontsize=16,
            fontweight="bold",
            pad=15
        )

        # ---------------------------------
        # Save
        # ---------------------------------

        plt.tight_layout()

        plt.savefig(
            self.output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        # ---------------------------------
        # Summary
        # ---------------------------------

        print(
            "\n========================================"
        )

        print(
            "Confusion Matrix Results"
        )

        print(
            "========================================"
        )

        print(
            f"Classes : "
            f"{self.num_classes}"
        )

        print(
            f"Samples : "
            f"{len(all_targets)}"
        )

        print(
            "========================================"
        )

        print(
            f"\nSaved visualization to:\n"
            f"{self.output_path}"
        )

        print(
            f"\nSaved raw matrix to:\n"
            f"{raw_csv}"
        )

        print(
            f"\nSaved normalized matrix to:\n"
            f"{normalized_csv}"
        )

        return cm_normalized


# =========================================
# Main
# =========================================

def main():

    config = ConfigLoader.load(
        "configs/colab_config.yaml"
    )

    checkpoint_path = (
        Path(
            config.checkpoint.save_dir
        )
        / "best_model.pth"
    )

    evaluator = ConfusionMatrixEvaluator(
        checkpoint_path=checkpoint_path,
        dataset_dir=config.dataset.test_dir,
        output_path=(
            "outputs/visualizations/"
            "confusion_matrix.png"
        ),
        batch_size=config.training.batch_size,
        image_size=config.dataset.image_size,
        label_step=5
    )

    evaluator.generate()


if __name__ == "__main__":
    main()