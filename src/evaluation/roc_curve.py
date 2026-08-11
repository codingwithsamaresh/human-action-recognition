"""
ROC Curve Generator

Generates ROC curves for the 101-class CNN-LSTM
Human Action Recognition model.

Visualization strategy:
- All class ROC curves are shown faintly
- No 101-item legend
- Macro-average ROC is highlighted
- Micro-average ROC is highlighted
- Random classifier baseline is shown
- Per-class AUC values are saved separately

Outputs:
outputs/visualizations/roc_curve.png
outputs/reports/roc_auc_per_class.csv
"""

from pathlib import Path

import numpy as np

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score
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
        output_path=(
            "outputs/visualizations/"
            "roc_curve.png"
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
    # Generate ROC
    # =====================================

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

        print(
            "\nGenerating ROC predictions..."
        )

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
        # One-vs-rest labels
        # ---------------------------------

        y_true_bin = label_binarize(
            y_true,
            classes=np.arange(
                self.num_classes
            )
        )

        # ---------------------------------
        # Per-class ROC
        # ---------------------------------

        class_aucs = []

        fpr_dict = {}
        tpr_dict = {}

        for i in range(
            self.num_classes
        ):

            positives = np.sum(
                y_true_bin[:, i]
            )

            negatives = (
                len(y_true_bin[:, i])
                - positives
            )

            # Skip invalid classes
            if (
                positives == 0
                or negatives == 0
            ):
                continue

            fpr, tpr, _ = roc_curve(
                y_true_bin[:, i],
                y_score[:, i]
            )

            roc_auc = auc(
                fpr,
                tpr
            )

            fpr_dict[i] = fpr
            tpr_dict[i] = tpr

            class_aucs.append(
                (
                    i,
                    self.class_names[i],
                    roc_auc
                )
            )

        # ---------------------------------
        # Macro-average ROC
        # ---------------------------------

        all_fpr = np.unique(
            np.concatenate(
                [
                    fpr_dict[i]
                    for i in fpr_dict
                ]
            )
        )

        mean_tpr = np.zeros_like(
            all_fpr
        )

        for i in fpr_dict:

            mean_tpr += np.interp(
                all_fpr,
                fpr_dict[i],
                tpr_dict[i]
            )

        mean_tpr /= len(
            fpr_dict
        )

        macro_auc = auc(
            all_fpr,
            mean_tpr
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
        # Save per-class AUC
        # ---------------------------------

        auc_csv = (
            self.report_dir
            / "roc_auc_per_class.csv"
        )

        class_aucs_sorted = sorted(
            class_aucs,
            key=lambda x: x[2],
            reverse=True
        )

        with open(
            auc_csv,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "class_index,class_name,auc\n"
            )

            for (
                index,
                class_name,
                score
            ) in class_aucs_sorted:

                f.write(
                    f"{index},"
                    f"{class_name},"
                    f"{score:.6f}\n"
                )

        # ---------------------------------
        # Plot
        # ---------------------------------

        plt.figure(
            figsize=(11, 8)
        )

        # All class curves
        for i in fpr_dict:

            plt.plot(
                fpr_dict[i],
                tpr_dict[i],
                linewidth=0.8,
                alpha=0.18
            )

        # Random classifier
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1.5,
            label="Random Classifier"
        )

        # Micro-average
        plt.plot(
            micro_fpr,
            micro_tpr,
            linewidth=2.5,
            label=(
                f"Micro-average "
                f"(AUC = {micro_auc:.3f})"
            )
        )

        # Macro-average
        plt.plot(
            all_fpr,
            mean_tpr,
            linewidth=3.0,
            label=(
                f"Macro-average "
                f"(AUC = {macro_auc:.3f})"
            )
        )

        # ---------------------------------
        # Formatting
        # ---------------------------------

        plt.xlabel(
            "False Positive Rate",
            fontsize=13
        )

        plt.ylabel(
            "True Positive Rate",
            fontsize=13
        )

        plt.title(
            "ROC Curves - CNN-LSTM "
            "(101-Class UCF101)",
            fontsize=16,
            fontweight="bold"
        )

        plt.xlim(
            0,
            1
        )

        plt.ylim(
            0,
            1.02
        )

        plt.grid(
            alpha=0.2
        )

        plt.legend(
            loc="lower right",
            fontsize=11,
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

        # ---------------------------------
        # Print summary
        # ---------------------------------

        print(
            "\n========================================"
        )

        print(
            "ROC Evaluation Results"
        )

        print(
            "========================================"
        )

        print(
            f"Classes evaluated : "
            f"{len(class_aucs)}"
        )

        print(
            f"Macro AUC         : "
            f"{macro_auc:.4f}"
        )

        print(
            f"Micro AUC         : "
            f"{micro_auc:.4f}"
        )

        print(
            "========================================"
        )

        print(
            f"\nSaved ROC curve to:\n"
            f"{self.output_path}"
        )

        print(
            f"\nSaved per-class AUC report to:\n"
            f"{auc_csv}"
        )

        return {
            "macro_auc": macro_auc,
            "micro_auc": micro_auc,
            "output_path": self.output_path,
            "auc_csv": auc_csv
        }


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