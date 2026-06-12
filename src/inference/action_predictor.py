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


class ActionPredictor:
    """
    Performs action recognition using a trained CNN-LSTM model.
    """

    def __init__(
        self,
        checkpoint_path,
        class_names,
        sequence_length=16,
        image_size=224
    ):
        self.device = get_device()

        self.class_names = class_names
        self.sequence_length = sequence_length
        self.image_size = image_size

        self.model = CNNLSTMBaseline(
            num_classes=len(class_names),
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
        """
        Load trained checkpoint.
        Supports both:
            - state_dict only
            - training checkpoint dictionary
        """

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

        self.model.to(self.device)

        print(
            f"Loaded checkpoint from: "
            f"{checkpoint_path}"
        )

    def _preprocess_frame(
        self,
        frame
    ):
        """
        Convert OpenCV BGR frame into
        normalized CHW tensor format.
        """

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
        """
        Convert list of frames into:
            (1, T, C, H, W)
        """

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
        """
        Predict action from a sequence of frames.

        Args:
            frames:
                List of frames

        Returns:
            {
                "action": str,
                "confidence": float
            }
        """

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