"""
Confusion Matrix Generator
==========================

Generates a clean normalized confusion matrix for the
101-class UCF101 Human Action Recognition model.

Features
--------
- Loads trained CNN-LSTM checkpoint
- Automatically resolves the test dataset directory
- Supports 101 classes
- Row-normalized confusion matrix
- No cell annotations
- Sparse class labels for readability
- Saves directly to Google Drive
- Also saves a high-resolution PNG

Output
------
Google Drive:
human_action_recognition/outputs/visualizations/confusion_matrix.png
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


# ============================================================
# Paths
# ============================================================

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/human_action_recognition"
)

DRIVE_OUTPUT_DIR = (
    DRIVE_ROOT
    / "outputs"
    / "visualizations"
)


# ============================================================
# Resolve test directory
# ============================================================

def resolve_test_directory(config):
    """
    Find the test dataset directory.

    Checks:
    1. config.dataset.test_dir
    2. processed_sequences_dir/test
    3. /content/processed/sequences/test
    4. Google Drive processed sequences/test
    """

    candidates = []

    # --------------------------------------------------------
    # Config test_dir
    # --------------------------------------------------------

    if hasattr(config.dataset, "test_dir"):

        candidates.append(
            Path(config.dataset.test_dir)
        )

    # --------------------------------------------------------
    # Config processed sequences directory
    # --------------------------------------------------------

    if hasattr(
        config.dataset,
        "processed_sequences_dir"
    ):

        processed_dir = Path(
            config.dataset.processed_sequences_dir
        )

        candidates.append(
            processed_dir / "test"
        )

    # --------------------------------------------------------
    # Common Colab paths
    # --------------------------------------------------------

    candidates.extend(
        [
            Path(
                "/content/processed/sequences/test"
            ),
            Path(
                "/content/test"
            ),
            DRIVE_ROOT
            / "datasets"
            / "processed"
            / "sequences"
            / "test",
        ]
    )

    # --------------------------------------------------------
    # Return first valid directory
    # --------------------------------------------------------

    for path in candidates:

        if path.exists() and path.is_dir():

            print(
                f"Using test dataset:\n{path}"
            )

            return path

    # --------------------------------------------------------
    # Nothing found
    # --------------------------------------------------------

    print("\nChecked the following locations:")

    for path in candidates:
        print(f"  - {path}")

    raise FileNotFoundError(
        "\nCould not find the test dataset directory.\n"
        "Please verify where your processed test sequences "
        "are stored."
    )


# ============================================================
# Confusion Matrix Evaluator
# ============================================================

class ConfusionMatrixEvaluator:

    def __init__(
        self,
        checkpoint_path,
        dataset_dir,
        output_path,
        batch_size=8,
        image_size=224,
    ):

        self.device = get_device()

        print(
            f"\nUsing device: {self.device}"
        )

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        print(
            "\nLoading test dataset..."
        )

        self.dataset = ActionSequenceDataset(
            sequence_root=str(dataset_dir),
            image_size=image_size,
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

        # ----------------------------------------------------
        # DataLoader
        # ----------------------------------------------------

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
            if self.device.type == "cuda"
            else False,
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        print(
            "\nCreating model..."
        )

        self.model = CNNLSTMBaseline(
            num_classes=self.num_classes,
            pretrained=False,
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        self.output_path = Path(
            output_path
        )

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        self._load_checkpoint(
            checkpoint_path
        )

    # ========================================================
    # Load checkpoint
    # ========================================================

    def _load_checkpoint(
        self,
        checkpoint_path,
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
            map_location=self.device,
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        else:

            state_dict = checkpoint

        self.model.load_state_dict(
            state_dict
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            f"\nLoaded checkpoint from:\n"
            f"{checkpoint_path}"
        )

    # ========================================================
    # Generate predictions
    # ========================================================

    @torch.no_grad()
    def _get_predictions(self):

        all_predictions = []
        all_targets = []

        print(
            "\nGenerating predictions..."
        )

        for frames, targets in self.dataloader:

            frames = frames.to(
                self.device,
                non_blocking=True,
            )

            logits = self.model(
                frames
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                targets.cpu().numpy()
            )

        return (
            np.asarray(all_targets),
            np.asarray(all_predictions),
        )

    # ========================================================
    # Generate confusion matrix
    # ========================================================

    def generate(self):

        y_true, y_pred = (
            self._get_predictions()
        )

        # ----------------------------------------------------
        # Raw confusion matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=np.arange(
                self.num_classes
            ),
        )

        # ----------------------------------------------------
        # Row normalization
        # ----------------------------------------------------

        row_sums = cm.sum(
            axis=1,
            keepdims=True,
        )

        cm_normalized = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(
                cm,
                dtype=float,
            ),
            where=row_sums != 0,
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(20, 18)
        )

        sns.heatmap(
            cm_normalized,
            cmap="Blues",
            vmin=0,
            vmax=1,
            square=True,
            cbar=True,
            xticklabels=False,
            yticklabels=False,
            linewidths=0,
            ax=ax,
        )

        # ----------------------------------------------------
        # Sparse labels
        # ----------------------------------------------------

        label_step = max(
            1,
            self.num_classes // 20
        )

        tick_positions = np.arange(
            0,
            self.num_classes,
            label_step,
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
            fontsize=8,
        )

        ax.set_yticklabels(
            tick_labels,
            rotation=0,
            fontsize=8,
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        ax.set_xlabel(
            "Predicted Label",
            fontsize=14,
        )

        ax.set_ylabel(
            "True Label",
            fontsize=14,
        )

        ax.set_title(
            "Normalized Confusion Matrix — UCF101",
            fontsize=18,
            pad=15,
        )

        # ----------------------------------------------------
        # Colorbar
        # ----------------------------------------------------

        cbar = ax.collections[0].colorbar

        cbar.set_label(
            "Classification Rate",
            fontsize=12,
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        plt.tight_layout()

        plt.savefig(
            self.output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "\n========================================"
        )

        print(
            "Confusion Matrix Generated"
        )

        print(
            "========================================"
        )

        print(
            f"Classes     : {self.num_classes}"
        )

        print(
            f"Samples     : {len(y_true)}"
        )

        print(
            f"Saved to    :\n{self.output_path}"
        )

        print(
            "========================================\n"
        )

        return cm_normalized


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    config = ConfigLoader.load(
        "configs/colab_config.yaml"
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint_path = (
        Path(
            config.checkpoint.save_dir
        )
        / "best_model.pth"
    )

    # --------------------------------------------------------
    # Resolve dataset
    # --------------------------------------------------------

    test_dir = resolve_test_directory(
        config
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_path = (
        DRIVE_OUTPUT_DIR
        / "confusion_matrix.png"
    )

    # --------------------------------------------------------
    # Evaluator
    # --------------------------------------------------------

    evaluator = (
        ConfusionMatrixEvaluator(
            checkpoint_path=checkpoint_path,
            dataset_dir=test_dir,
            output_path=output_path,
            batch_size=config.training.batch_size,
            image_size=config.dataset.image_size,
        )
    )

    evaluator.generate()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()