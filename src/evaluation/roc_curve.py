"""
ROC Curve Generator
===================

Generates a clean ROC visualization for the
101-class UCF101 Human Action Recognition model.

Features
--------
- Loads trained CNN-LSTM checkpoint
- Automatically resolves test dataset
- Computes one-vs-rest ROC
- Computes per-class AUC
- Computes micro-average ROC
- Computes macro-average ROC
- Displays only top classes by AUC
- Saves AUC values to CSV
- Saves visualization directly to Google Drive

Outputs
-------
Google Drive:
human_action_recognition/outputs/visualizations/roc_curve.png
human_action_recognition/outputs/reports/class_auc_scores.csv
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
)

from sklearn.preprocessing import (
    label_binarize,
)

from src.data.dataset import (
    ActionSequenceDataset
)

from src.models.cnn_lstm_baseline import (
    CNNLSTMBaseline
)

from src.utils.device import (
    get_device
)

from src.utils.config_loader import (
    ConfigLoader
)


# ============================================================
# Paths
# ============================================================

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/human_action_recognition"
)

DRIVE_VISUALIZATION_DIR = (
    DRIVE_ROOT
    / "outputs"
    / "visualizations"
)

DRIVE_REPORT_DIR = (
    DRIVE_ROOT
    / "outputs"
    / "reports"
)


# ============================================================
# Resolve test dataset
# ============================================================

def resolve_test_directory(config):

    candidates = []

    # --------------------------------------------------------
    # Config test directory
    # --------------------------------------------------------

    if hasattr(
        config.dataset,
        "test_dir"
    ):

        candidates.append(
            Path(
                config.dataset.test_dir
            )
        )

    # --------------------------------------------------------
    # Processed sequences
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
    # Common Colab locations
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
    # Find valid directory
    # --------------------------------------------------------

    for path in candidates:

        if path.exists() and path.is_dir():

            print(
                f"Using test dataset:\n{path}"
            )

            return path

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    print(
        "\nChecked locations:"
    )

    for path in candidates:

        print(
            f"  - {path}"
        )

    raise FileNotFoundError(
        "\nTest dataset directory could not be found."
    )


# ============================================================
# ROC Evaluator
# ============================================================

class ROCEvaluator:

    def __init__(
        self,
        checkpoint_path,
        dataset_dir,
        output_path,
        auc_csv_path,
        batch_size=8,
        image_size=224,
        top_k=15,
    ):

        self.device = get_device()

        self.top_k = top_k

        print(
            f"\nUsing device: {self.device}"
        )

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        print(
            "\nLoading test dataset..."
        )

        self.dataset = (
            ActionSequenceDataset(
                sequence_root=str(
                    dataset_dir
                ),
                image_size=image_size,
            )
        )

        self.class_names = (
            self.dataset.get_class_names()
        )

        self.num_classes = (
            self.dataset.get_num_classes()
        )

        print(
            f"Loaded {len(self.dataset)} "
            f"sequences from "
            f"{self.num_classes} classes."
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

        self.model = (
            CNNLSTMBaseline(
                num_classes=self.num_classes,
                pretrained=False,
            )
        )

        # ----------------------------------------------------
        # Output paths
        # ----------------------------------------------------

        self.output_path = Path(
            output_path
        )

        self.auc_csv_path = Path(
            auc_csv_path
        )

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.auc_csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Load checkpoint
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
    # Collect probabilities
    # ========================================================

    @torch.no_grad()
    def _collect_predictions(self):

        all_probs = []
        all_targets = []

        print(
            "\nGenerating prediction probabilities..."
        )

        for frames, targets in self.dataloader:

            frames = frames.to(
                self.device,
                non_blocking=True,
            )

            logits = self.model(
                frames
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            all_probs.append(
                probabilities.cpu().numpy()
            )

            all_targets.append(
                targets.cpu().numpy()
            )

        y_score = np.concatenate(
            all_probs,
            axis=0,
        )

        y_true = np.concatenate(
            all_targets,
            axis=0,
        )

        return (
            y_true,
            y_score,
        )

    # ========================================================
    # Generate ROC
    # ========================================================

    def generate(self):

        if self.num_classes < 2:

            raise ValueError(
                "ROC requires at least 2 classes."
            )

        y_true, y_score = (
            self._collect_predictions()
        )

        # ----------------------------------------------------
        # One-hot targets
        # ----------------------------------------------------

        y_true_bin = label_binarize(
            y_true,
            classes=np.arange(
                self.num_classes
            ),
        )

        # ----------------------------------------------------
        # Per-class ROC
        # ----------------------------------------------------

        fpr = {}
        tpr = {}
        roc_auc = {}

        valid_classes = []

        for i in range(
            self.num_classes
        ):

            # Skip classes with no positive examples
            if (
                np.sum(
                    y_true_bin[:, i]
                )
                == 0
            ):
                continue

            fpr[i], tpr[i], _ = (
                roc_curve(
                    y_true_bin[:, i],
                    y_score[:, i],
                )
            )

            roc_auc[i] = auc(
                fpr[i],
                tpr[i],
            )

            valid_classes.append(i)

        # ----------------------------------------------------
        # Micro-average ROC
        # ----------------------------------------------------

        fpr_micro, tpr_micro, _ = (
            roc_curve(
                y_true_bin.ravel(),
                y_score.ravel(),
            )
        )

        auc_micro = auc(
            fpr_micro,
            tpr_micro,
        )

        # ----------------------------------------------------
        # Macro-average ROC
        # ----------------------------------------------------

        all_fpr = np.unique(
            np.concatenate(
                [
                    fpr[i]
                    for i in valid_classes
                ]
            )
        )

        mean_tpr = np.zeros_like(
            all_fpr
        )

        for i in valid_classes:

            mean_tpr += np.interp(
                all_fpr,
                fpr[i],
                tpr[i],
            )

        mean_tpr /= len(
            valid_classes
        )

        fpr_macro = all_fpr
        tpr_macro = mean_tpr

        auc_macro = auc(
            fpr_macro,
            tpr_macro,
        )

        # ----------------------------------------------------
        # Sort classes by AUC
        # ----------------------------------------------------

        sorted_classes = sorted(
            valid_classes,
            key=lambda i: roc_auc[i],
            reverse=True,
        )

        top_classes = (
            sorted_classes[:self.top_k]
        )

        # ----------------------------------------------------
        # Save AUC CSV
        # ----------------------------------------------------

        csv_lines = [
            "class,auc\n"
        ]

        for i in sorted_classes:

            csv_lines.append(
                f"{self.class_names[i]},"
                f"{roc_auc[i]:.6f}\n"
            )

        self.auc_csv_path.write_text(
            "".join(csv_lines),
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        plt.figure(
            figsize=(12, 9)
        )

        # ----------------------------------------------------
        # Plot top classes
        # ----------------------------------------------------

        for i in top_classes:

            plt.plot(
                fpr[i],
                tpr[i],
                linewidth=1.5,
                alpha=0.75,
                label=(
                    f"{self.class_names[i]} "
                    f"(AUC={roc_auc[i]:.3f})"
                ),
            )

        # ----------------------------------------------------
        # Macro average
        # ----------------------------------------------------

        plt.plot(
            fpr_macro,
            tpr_macro,
            linewidth=3,
            label=(
                f"Macro-average "
                f"(AUC={auc_macro:.3f})"
            ),
        )

        # ----------------------------------------------------
        # Micro average
        # ----------------------------------------------------

        plt.plot(
            fpr_micro,
            tpr_micro,
            linewidth=3,
            linestyle="--",
            label=(
                f"Micro-average "
                f"(AUC={auc_micro:.3f})"
            ),
        )

        # ----------------------------------------------------
        # Random classifier
        # ----------------------------------------------------

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle=":",
            linewidth=2,
            label="Random classifier",
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        plt.xlabel(
            "False Positive Rate",
            fontsize=13,
        )

        plt.ylabel(
            "True Positive Rate",
            fontsize=13,
        )

        plt.title(
            "ROC Curve — UCF101 Human Action Recognition",
            fontsize=17,
            pad=15,
        )

        plt.xlim(
            0,
            1,
        )

        plt.ylim(
            0,
            1.02,
        )

        plt.grid(
            alpha=0.25
        )

        # ----------------------------------------------------
        # Legend
        # ----------------------------------------------------

        plt.legend(
            loc="lower right",
            fontsize=8,
            framealpha=0.9,
        )

        plt.tight_layout()

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        plt.savefig(
            self.output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        print(
            "\n========================================"
        )

        print(
            "ROC Evaluation Complete"
        )

        print(
            "========================================"
        )

        print(
            f"Classes           : "
            f"{self.num_classes}"
        )

        print(
            f"Samples           : "
            f"{len(y_true)}"
        )

        print(
            f"Macro AUC         : "
            f"{auc_macro:.4f}"
        )

        print(
            f"Micro AUC         : "
            f"{auc_micro:.4f}"
        )

        print(
            f"Top classes shown : "
            f"{len(top_classes)}"
        )

        print(
            f"\nROC curve saved to:\n"
            f"{self.output_path}"
        )

        print(
            f"\nAUC scores saved to:\n"
            f"{self.auc_csv_path}"
        )

        print(
            "========================================\n"
        )

        return {
            "macro_auc": auc_macro,
            "micro_auc": auc_micro,
            "roc_auc": roc_auc,
        }


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Configuration
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
    # Test directory
    # --------------------------------------------------------

    test_dir = resolve_test_directory(
        config
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    roc_output = (
        DRIVE_VISUALIZATION_DIR
        / "roc_curve.png"
    )

    auc_csv_output = (
        DRIVE_REPORT_DIR
        / "class_auc_scores.csv"
    )

    # --------------------------------------------------------
    # Evaluator
    # --------------------------------------------------------

    evaluator = ROCEvaluator(
        checkpoint_path=checkpoint_path,
        dataset_dir=test_dir,
        output_path=roc_output,
        auc_csv_path=auc_csv_output,
        batch_size=config.training.batch_size,
        image_size=config.dataset.image_size,
        top_k=15,
    )

    evaluator.generate()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()