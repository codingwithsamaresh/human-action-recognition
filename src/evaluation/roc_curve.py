"""
ROC Curve Generator

Generates ROC curves and AUC scores
for the trained CNN-LSTM action recognition model.

Output:
outputs/visualizations/roc_curve.png
"""

from pathlib import Path

import numpy as np

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

from src.data.dataset import ActionSequenceDataset
from src.models.cnn_lstm_baseline import CNNLSTMBaseline
from src.utils.device import get_device
from src.utils.config_loader import ConfigLoader


class ROCEvaluator:

    def __init__(
        self,
        checkpoint_path,
        dataset_dir,
        output_path="outputs/visualizations/roc_curve.png",
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

        if self.num_classes < 2:

            print(
                "\nROC curve requires "
                "at least 2 classes."
            )

            return None

        # ---------------------------------
        # Collect predictions
        # ---------------------------------

        all_probs = []
        all_targets = []

        print("\nGenerating ROC data...")

        for frames, targets in self.dataloader:

            frames = frames.to(
                self.device
            )

            logits = self.model(
                frames
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            all_probs.append(
                probabilities.cpu().numpy()
            )

            all_targets.append(
                targets.cpu().numpy()
            )

        if not all_probs:

            print(
                "No predictions were generated."
            )

            return None

        y_score = np.concatenate(
            all_probs,
            axis=0
        )

        y_true = np.concatenate(
            all_targets,
            axis=0
        )

        # ---------------------------------
        # Convert labels to one-vs-rest
        # ---------------------------------

        y_true_bin = label_binarize(
            y_true,
            classes=np.arange(
                self.num_classes
            )
        )

        # ---------------------------------
        # Plot
        # ---------------------------------

        plt.figure(
            figsize=(12, 9)
        )

        valid_curves = 0

        for i in range(
            self.num_classes
        ):

            # Skip classes that do not contain
            # both positive and negative samples.
            positives = np.sum(
                y_true_bin[:, i]
            )

            negatives = (
                len(y_true_bin[:, i])
                - positives
            )

            if positives == 0 or negatives == 0:

                print(
                    f"Skipping ROC for "
                    f"{self.class_names[i]} "
                    f"(insufficient samples)."
                )

                continue

            fpr, tpr, _ = roc_curve(
                y_true_bin[:, i],
                y_score[:, i]
            )

            roc_auc = auc(
                fpr,
                tpr
            )

            plt.plot(
                fpr,
                tpr,
                linewidth=1.2,
                label=(
                    f"{self.class_names[i]} "
                    f"(AUC={roc_auc:.3f})"
                )
            )

            valid_curves += 1

        # ---------------------------------
        # Random classifier baseline
        # ---------------------------------

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1.5,
            label="Random Classifier"
        )

        plt.xlabel(
            "False Positive Rate"
        )

        plt.ylabel(
            "True Positive Rate"
        )

        plt.title(
            "ROC Curves - CNN-LSTM"
        )

        plt.xlim(
            0,
            1
        )

        plt.ylim(
            0,
            1.05
        )

        # 101 classes makes a normal legend
        # extremely large, so place it outside.
        if valid_curves > 0:

            plt.legend(
                loc="upper left",
                bbox_to_anchor=(
                    1.02,
                    1
                ),
                fontsize=7
            )

        plt.grid(
            alpha=0.2
        )

        plt.tight_layout()

        plt.savefig(
            self.output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"\nSaved ROC curve to:\n"
            f"{self.output_path}"
        )

        return self.output_path


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
    # Generate ROC curve
    # ---------------------------------

    evaluator = ROCEvaluator(
        checkpoint_path=checkpoint_path,
        dataset_dir=config.dataset.test_dir,
        output_path=(
            "outputs/visualizations/"
            "roc_curve.png"
        ),
        batch_size=config.training.batch_size,
        image_size=config.dataset.image_size
    )

    evaluator.generate()


if __name__ == "__main__":
    main()