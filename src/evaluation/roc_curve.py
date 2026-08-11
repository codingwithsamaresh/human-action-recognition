"""
ROC Curve Generator

Generates ROC curves and AUC scores
for the trained CNN-LSTM action recognition model.

For 101 classes:
- Shows Micro-average ROC
- Shows Macro-average ROC
- Shows Top 15 classes by AUC
- Avoids an unreadable 101-item legend

Output:
outputs/visualizations/roc_curve.png
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
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

        self.class_names = self.dataset.get_class_names()
        self.num_classes = self.dataset.get_num_classes()

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

        self.output_path = Path(output_path)

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # ---------------------------------
        # Load checkpoint
        # ---------------------------------

        self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found:\n{checkpoint_path}"
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
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        print(
            f"Loaded checkpoint from:\n"
            f"{checkpoint_path}"
        )

    @torch.no_grad()
    def generate(self):

        if self.num_classes < 2:
            print(
                "\nROC curve requires at least 2 classes."
            )
            return None

        # ---------------------------------
        # Collect predictions
        # ---------------------------------

        all_probs = []
        all_targets = []

        print("\nGenerating ROC data...")

        for frames, targets in self.dataloader:

            frames = frames.to(self.device)

            logits = self.model(frames)

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
            print("No predictions were generated.")
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
        # One-vs-rest labels
        # ---------------------------------

        y_true_bin = label_binarize(
            y_true,
            classes=np.arange(self.num_classes)
        )

        # ---------------------------------
        # Calculate per-class ROC/AUC
        # ---------------------------------

        class_results = []

        for i in range(self.num_classes):

            positives = np.sum(
                y_true_bin[:, i]
            )

            negatives = (
                len(y_true_bin[:, i])
                - positives
            )

            if positives == 0 or negatives == 0:
                continue

            fpr, tpr, _ = roc_curve(
                y_true_bin[:, i],
                y_score[:, i]
            )

            roc_auc = auc(
                fpr,
                tpr
            )

            class_results.append(
                {
                    "index": i,
                    "name": self.class_names[i],
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": roc_auc
                }
            )

        if not class_results:
            print("No valid ROC curves.")
            return None

        # ---------------------------------
        # Sort classes by AUC
        # ---------------------------------

        class_results.sort(
            key=lambda x: x["auc"],
            reverse=True
        )

        # ---------------------------------
        # Micro-average ROC
        # ---------------------------------

        micro_fpr, micro_tpr, _ = roc_curve(
            y_true_bin.ravel(),
            y_score.ravel()
        )

        micro_auc = auc(
            micro_fpr,
            micro_tpr
        )

        # ---------------------------------
        # Macro-average ROC
        # ---------------------------------

        all_fpr = np.unique(
            np.concatenate(
                [
                    result["fpr"]
                    for result in class_results
                ]
            )
        )

        mean_tpr = np.zeros_like(all_fpr)

        for result in class_results:

            mean_tpr += np.interp(
                all_fpr,
                result["fpr"],
                result["tpr"]
            )

        mean_tpr /= len(class_results)

        macro_auc = auc(
            all_fpr,
            mean_tpr
        )

        # ---------------------------------
        # Plot
        # ---------------------------------

        plt.figure(
            figsize=(12, 9)
        )

        # Micro-average
        plt.plot(
            micro_fpr,
            micro_tpr,
            linewidth=2.5,
            label=f"Micro-average (AUC = {micro_auc:.3f})"
        )

        # Macro-average
        plt.plot(
            all_fpr,
            mean_tpr,
            linewidth=2.5,
            linestyle="--",
            label=f"Macro-average (AUC = {macro_auc:.3f})"
        )

        # ---------------------------------
        # Top 15 classes
        # ---------------------------------

        top_k = min(
            15,
            len(class_results)
        )

        for result in class_results[:top_k]:

            plt.plot(
                result["fpr"],
                result["tpr"],
                linewidth=1.2,
                alpha=0.75,
                label=(
                    f"{result['name']} "
                    f"(AUC={result['auc']:.3f})"
                )
            )

        # ---------------------------------
        # Random classifier
        # ---------------------------------

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle=":",
            linewidth=1.5,
            label="Random Classifier"
        )

        # ---------------------------------
        # Formatting
        # ---------------------------------

        plt.xlabel(
            "False Positive Rate",
            fontsize=12
        )

        plt.ylabel(
            "True Positive Rate",
            fontsize=12
        )

        plt.title(
            "ROC Curves - CNN-LSTM on UCF101",
            fontsize=16
        )

        plt.xlim(
            0,
            1
        )

        plt.ylim(
            0,
            1.05
        )

        plt.grid(
            alpha=0.2
        )

        plt.legend(
            loc="lower right",
            fontsize=8,
            frameon=True
        )

        plt.tight_layout()

        # ---------------------------------
        # Save
        # ---------------------------------

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

        print(
            f"\nMicro-average AUC: {micro_auc:.4f}"
        )

        print(
            f"Macro-average AUC: {macro_auc:.4f}"
        )

        print(
            f"\nTop {top_k} classes by AUC:"
        )

        for result in class_results[:top_k]:

            print(
                f"{result['name']}: "
                f"{result['auc']:.4f}"
            )

        return self.output_path


def main():

    # ---------------------------------
    # Load configuration
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
    # Generate ROC
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