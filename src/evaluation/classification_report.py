"""
Classification Report Generator

Evaluates the trained CNN-LSTM model on the test dataset.

Generates:
    - Per-class precision
    - Per-class recall
    - Per-class F1-score
    - Support
    - Overall accuracy
    - Macro average
    - Weighted average

Outputs:
    Google Drive:
    outputs/reports/classification_report.csv
    outputs/reports/classification_report.json

Also prints:
    - Overall metrics
    - Top 10 best classes by F1
    - Top 10 worst classes by F1
"""

from pathlib import Path
import json

import numpy as np

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    accuracy_score
)

from src.data.dataset import ActionSequenceDataset
from src.models.cnn_lstm_baseline import CNNLSTMBaseline
from src.utils.device import get_device
from src.utils.config_loader import ConfigLoader


class ClassificationReportEvaluator:

    def __init__(
        self,
        checkpoint_path,
        dataset_dir,
        output_dir,
        batch_size=8,
        image_size=224
    ):

        # ---------------------------------
        # Device
        # ---------------------------------

        self.device = get_device()

        print(
            f"\nUsing device: {self.device}"
        )

        # ---------------------------------
        # Load test dataset
        # ---------------------------------

        print(
            "\nLoading test dataset..."
        )

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
            f"Loaded {len(self.dataset)} "
            f"sequences from "
            f"{self.num_classes} classes."
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

        print(
            "\nCreating model..."
        )

        self.model = CNNLSTMBaseline(
            num_classes=self.num_classes,
            pretrained=False
        )

        # ---------------------------------
        # Output directory
        # ---------------------------------

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ---------------------------------
        # Checkpoint
        # ---------------------------------

        self._load_checkpoint(
            checkpoint_path
        )

    # ==================================================
    # Load checkpoint
    # ==================================================

    def _load_checkpoint(
        self,
        checkpoint_path
    ):

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"\nCheckpoint not found:\n"
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        # ---------------------------------
        # Handle checkpoint dictionary
        # ---------------------------------

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
            f"\nLoaded checkpoint from:\n"
            f"{checkpoint_path}"
        )

    # ==================================================
    # Generate predictions
    # ==================================================

    @torch.no_grad()
    def _get_predictions(self):

        all_predictions = []
        all_targets = []

        print(
            "\nRunning inference on test dataset..."
        )

        for batch_idx, (
            frames,
            targets
        ) in enumerate(self.dataloader):

            frames = frames.to(
                self.device,
                non_blocking=True
            )

            # ---------------------------------
            # Forward pass
            # ---------------------------------

            logits = self.model(
                frames
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            # ---------------------------------
            # Store results
            # ---------------------------------

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                targets.cpu().numpy()
            )

            if (
                batch_idx + 1
            ) % 500 == 0:

                print(
                    f"Processed "
                    f"{batch_idx + 1} batches..."
                )

        return (
            np.asarray(all_targets),
            np.asarray(all_predictions)
        )

    # ==================================================
    # Generate classification report
    # ==================================================

    def generate(self):

        print(
            "\n"
            + "=" * 60
        )

        print(
            "Generating Classification Report"
        )

        print(
            "=" * 60
        )

        # ---------------------------------
        # Predictions
        # ---------------------------------

        y_true, y_pred = (
            self._get_predictions()
        )

        # ---------------------------------
        # Sanity check
        # ---------------------------------

        if len(y_true) != len(y_pred):

            raise RuntimeError(
                "Number of predictions does not "
                "match number of targets."
            )

        print(
            f"\nTotal test samples: "
            f"{len(y_true)}"
        )

        # ---------------------------------
        # Accuracy
        # ---------------------------------

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        # ---------------------------------
        # Classification report
        # ---------------------------------

        report_dict = classification_report(
            y_true,
            y_pred,
            labels=list(
                range(self.num_classes)
            ),
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )

        # ---------------------------------
        # Add overall accuracy
        # ---------------------------------

        report_dict["accuracy"] = float(
            accuracy
        )

        # ---------------------------------
        # Save JSON
        # ---------------------------------

        json_path = (
            self.output_dir
            / "classification_report.json"
        )

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                report_dict,
                f,
                indent=4
            )

        # ---------------------------------
        # Create CSV
        # ---------------------------------

        csv_path = (
            self.output_dir
            / "classification_report.csv"
        )

        self._save_csv(
            report_dict,
            csv_path
        )

        # ---------------------------------
        # Print overall results
        # ---------------------------------

        self._print_summary(
            report_dict
        )

        # ---------------------------------
        # Print best/worst classes
        # ---------------------------------

        self._print_class_analysis(
            report_dict
        )

        # ---------------------------------
        # Final output
        # ---------------------------------

        print(
            "\n"
            + "=" * 60
        )

        print(
            "Classification report generated."
        )

        print(
            f"\nCSV saved to:\n"
            f"{csv_path}"
        )

        print(
            f"\nJSON saved to:\n"
            f"{json_path}"
        )

        print(
            "=" * 60
        )

        return report_dict

    # ==================================================
    # Save CSV
    # ==================================================

    def _save_csv(
        self,
        report_dict,
        csv_path
    ):

        import csv

        fieldnames = [
            "class",
            "precision",
            "recall",
            "f1_score",
            "support"
        ]

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            # ---------------------------------
            # Per-class results
            # ---------------------------------

            for class_name in self.class_names:

                metrics = report_dict[
                    class_name
                ]

                writer.writerow({
                    "class": class_name,
                    "precision": metrics[
                        "precision"
                    ],
                    "recall": metrics[
                        "recall"
                    ],
                    "f1_score": metrics[
                        "f1-score"
                    ],
                    "support": metrics[
                        "support"
                    ]
                })

            # ---------------------------------
            # Summary rows
            # ---------------------------------

            writer.writerow({
                "class": "accuracy",
                "precision": "",
                "recall": "",
                "f1_score": report_dict[
                    "accuracy"
                ],
                "support": len(
                    report_dict
                )
            })

            for name in [
                "macro avg",
                "weighted avg"
            ]:

                metrics = report_dict[
                    name
                ]

                writer.writerow({
                    "class": name,
                    "precision": metrics[
                        "precision"
                    ],
                    "recall": metrics[
                        "recall"
                    ],
                    "f1_score": metrics[
                        "f1-score"
                    ],
                    "support": metrics[
                        "support"
                    ]
                })

    # ==================================================
    # Print summary
    # ==================================================

    def _print_summary(
        self,
        report_dict
    ):

        print(
            "\n"
            + "=" * 60
        )

        print(
            "Overall Classification Results"
        )

        print(
            "=" * 60
        )

        print(
            f"Accuracy       : "
            f"{report_dict['accuracy']:.4f}"
        )

        print(
            f"Accuracy (%)   : "
            f"{report_dict['accuracy'] * 100:.2f}%"
        )

        print(
            f"Macro Precision: "
            f"{report_dict['macro avg']['precision']:.4f}"
        )

        print(
            f"Macro Recall   : "
            f"{report_dict['macro avg']['recall']:.4f}"
        )

        print(
            f"Macro F1       : "
            f"{report_dict['macro avg']['f1-score']:.4f}"
        )

        print(
            f"Weighted F1    : "
            f"{report_dict['weighted avg']['f1-score']:.4f}"
        )

    # ==================================================
    # Best / Worst classes
    # ==================================================

    def _print_class_analysis(
        self,
        report_dict
    ):

        class_results = []

        for class_name in self.class_names:

            metrics = report_dict[
                class_name
            ]

            class_results.append({
                "class": class_name,
                "precision": metrics[
                    "precision"
                ],
                "recall": metrics[
                    "recall"
                ],
                "f1": metrics[
                    "f1-score"
                ],
                "support": metrics[
                    "support"
                ]
            })

        # ---------------------------------
        # Sort by F1
        # ---------------------------------

        class_results_sorted = sorted(
            class_results,
            key=lambda x: x["f1"],
            reverse=True
        )

        # ---------------------------------
        # Best 10
        # ---------------------------------

        print(
            "\n"
            + "=" * 60
        )

        print(
            "Top 10 Classes by F1 Score"
        )

        print(
            "=" * 60
        )

        for i, result in enumerate(
            class_results_sorted[:10],
            start=1
        ):

            print(
                f"{i:2d}. "
                f"{result['class']:<25} "
                f"F1={result['f1']:.4f} "
                f"Precision={result['precision']:.4f} "
                f"Recall={result['recall']:.4f}"
            )

        # ---------------------------------
        # Worst 10
        # ---------------------------------

        print(
            "\n"
            + "=" * 60
        )

        print(
            "Bottom 10 Classes by F1 Score"
        )

        print(
            "=" * 60
        )

        for i, result in enumerate(
            class_results_sorted[-10:],
            start=1
        ):

            print(
                f"{i:2d}. "
                f"{result['class']:<25} "
                f"F1={result['f1']:.4f} "
                f"Precision={result['precision']:.4f} "
                f"Recall={result['recall']:.4f}"
            )


# ======================================================
# Main
# ======================================================

def main():

    # ---------------------------------
    # Load configuration
    # ---------------------------------

    config = ConfigLoader.load(
        "configs/colab_config.yaml"
    )

    # ---------------------------------
    # Google Drive output directory
    # ---------------------------------

    output_dir = (
        "/content/drive/MyDrive/"
        "human_action_recognition/"
        "outputs/reports"
    )

    # ---------------------------------
    # Checkpoint
    # ---------------------------------

    checkpoint_path = (
        f"{config.checkpoint.save_dir}/"
        "best_model.pth"
    )

    # ---------------------------------
    # Evaluator
    # ---------------------------------

    evaluator = (
        ClassificationReportEvaluator(
            checkpoint_path=checkpoint_path,
            dataset_dir=config.dataset.test_dir,
            output_dir=output_dir,
            batch_size=config.training.batch_size,
            image_size=config.dataset.image_size
        )
    )

    # ---------------------------------
    # Generate
    # ---------------------------------

    evaluator.generate()


# ======================================================
# Entry point
# ======================================================

if __name__ == "__main__":
    main()