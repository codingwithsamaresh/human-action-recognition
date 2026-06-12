"""
Frame Buffer Utility

Maintains a fixed-length queue of frames
for real-time action recognition.
"""

from collections import deque
from typing import List
import numpy as np


class FrameBuffer:
    """
    Stores the latest N frames.

    Example:
        buffer = FrameBuffer(maxlen=16)

        buffer.add(frame)

        if buffer.is_full():
            sequence = buffer.get_sequence()
    """

    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)

    def add(self, frame: np.ndarray) -> None:
        """
        Add a frame.

        Args:
            frame: H x W x C image
        """
        self.buffer.append(frame)

    def is_full(self) -> bool:
        """
        Returns:
            True if buffer contains maxlen frames
        """
        return len(self.buffer) == self.maxlen

    def clear(self) -> None:
        """
        Empty buffer.
        """
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def get_sequence(self) -> List[np.ndarray]:
        """
        Returns:
            List of frames in temporal order
        """
        return list(self.buffer)

    def get_numpy(self) -> np.ndarray:
        """
        Returns:
            (T, H, W, C)

        Useful before preprocessing.
        """
        return np.array(self.buffer)

    def get_last_frame(self) -> np.ndarray:
        """
        Returns:
            Most recent frame
        """
        if len(self.buffer) == 0:
            return None

        return self.buffer[-1]