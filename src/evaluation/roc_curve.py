"""
ROC Curve Generator

Generates ROC curve and AUC score
for trained action recognition model.

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

from src.data.dataset import (
    ActionSequenceDataset
)
from src.models.cnn_lstm_baseline import (
    CNNLSTMBaseline
)
from src.utils.device import (
    get_device
)

from src.utils.config_loader import ConfigLoader

config = ConfigLoader.load("configs/colab_config.yaml")


class ROCEvaluator:

    def __init__(
        self,
        checkpoint_path,
        dataset_dir,
        output_path=
        "outputs/visualizations/roc_curve.png",
        batch_size=8,
        image_size=224
    ):

        self.device = get_device()

        self.dataset = (
            ActionSequenceDataset(
                sequence_root=dataset_dir,
                image_size=image_size
            )
        )

        self.class_names = (
            self.dataset.get_class_names()
        )

        self.num_classes = (
            self.dataset.get_num_classes()
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        self.model = CNNLSTMBaseline(
            num_classes=self.num_classes,
            pretrained=False
        )

        self.output_path = Path(
            output_path
        )

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

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            f"Loaded checkpoint from: "
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

        all_probs = []
        all_targets = []

        for frames, targets in self.dataloader:

            frames = frames.to(
                self.device
            )

            logits = self.model(
                frames
            )

            probs = torch.softmax(
                logits,
                dim=1
            )

            all_probs.append(
                probs.cpu().numpy()
            )

            all_targets.append(
                targets.numpy()
            )

        y_score = np.concatenate(
            all_probs,
            axis=0
        )

        y_true = np.concatenate(
            all_targets,
            axis=0
        )

        y_true_bin = label_binarize(
            y_true,
            classes=list(
                range(self.num_classes)
            )
        )

        plt.figure(
            figsize=(8, 6)
        )

        for i in range(
            self.num_classes
        ):

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
                label=(
                    f"{self.class_names[i]}"
                    f" (AUC={roc_auc:.3f})"
                )
            )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel(
            "False Positive Rate"
        )

        plt.ylabel(
            "True Positive Rate"
        )

        plt.title(
            "ROC Curve"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            self.output_path,
            dpi=300
        )

        plt.close()

        print(
            f"Saved ROC curve to:\n"
            f"{self.output_path}"
        )

        return self.output_path


def main():

    evaluator = ROCEvaluator(
        checkpoint_path=f"{config.checkpoint.save_dir}/best_model.pth",
        dataset_dir=config.dataset.test_dir
    )

    evaluator.generate()


if __name__ == "__main__":
    main()