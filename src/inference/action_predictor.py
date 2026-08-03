"""
Action Predictor

Loads trained CNN-LSTM checkpoint
and performs action recognition.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.models.cnn_lstm_baseline import CNNLSTMBaseline
from src.utils.device import get_device
from src.utils.config_loader import ConfigLoader
from src.data.dataset import ActionSequenceDataset


config = ConfigLoader.load("configs/colab_config.yaml")


class ActionPredictor:
    """
    Performs action recognition using a trained CNN-LSTM model.
    """

    def __init__(
        self,
        checkpoint_path,
        sequence_length=16,
        image_size=224
    ):

        self.device = get_device()

        self.sequence_length = sequence_length
        self.image_size = image_size

        # -----------------------------------
        # Load class names automatically
        # -----------------------------------

        dataset = ActionSequenceDataset(
            sequence_root=config.dataset.train_dir,
            image_size=image_size
        )

        self.class_names = dataset.get_class_names()

        self.model = CNNLSTMBaseline(
            num_classes=len(self.class_names),
            pretrained=False
        )

        self._load_checkpoint(
            checkpoint_path
        )

        self.model.eval()

    def _load_checkpoint(
        self,
        checkpoint_path
    ):

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
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

        print(
            f"Loaded checkpoint from: {checkpoint_path}"
        )

    def _preprocess_frame(
        self,
        frame
    ):

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame = cv2.resize(
            frame,
            (
                self.image_size,
                self.image_size
            )
        )

        frame = frame.astype(
            np.float32
        ) / 255.0

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        frame = (
            frame - mean
        ) / std

        frame = np.transpose(
            frame,
            (2, 0, 1)
        )

        return frame

    def _prepare_sequence(
        self,
        frames
    ):

        processed_frames = [

            self._preprocess_frame(frame)

            for frame in frames

        ]

        sequence = np.stack(
            processed_frames,
            axis=0
        )

        tensor = torch.tensor(
            sequence,
            dtype=torch.float32
        )

        tensor = tensor.unsqueeze(0)

        return tensor.to(
            self.device
        )

    @torch.no_grad()
    def predict(
        self,
        frames
    ):

        if len(frames) != self.sequence_length:

            raise ValueError(
                f"Expected {self.sequence_length} frames, "
                f"got {len(frames)}"
            )

        inputs = self._prepare_sequence(
            frames
        )

        logits = self.model(
            inputs
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )

        confidence, pred_idx = (
            probabilities.max(dim=1)
        )

        action = self.class_names[
            pred_idx.item()
        ]

        return {

            "action": action,

            "confidence": float(
                confidence.item()
            )

        }