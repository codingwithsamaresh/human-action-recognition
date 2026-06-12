from collections import deque
from typing import List
import numpy as np


class SlidingWindow:
    """
    Maintains a rolling temporal window.

    Example:
        window = SlidingWindow(
            window_size=16,
            stride=4
        )
    """

    def __init__(
        self,
        window_size: int,
        stride: int
    ):
        self.window_size = window_size
        self.stride = stride

        self.buffer = deque(maxlen=window_size)

        self.frames_since_prediction = 0

    def add(self, frame: np.ndarray):
        self.buffer.append(frame)

        if len(self.buffer) == self.window_size:
            self.frames_since_prediction += 1

    def ready(self) -> bool:
        """
        Returns True when a prediction
        should be performed.
        """

        if len(self.buffer) < self.window_size:
            return False

        return self.frames_since_prediction >= self.stride

    def get_window(self) -> List[np.ndarray]:
        """
        Returns current temporal window.
        """

        self.frames_since_prediction = 0

        return list(self.buffer)

    def reset(self):
        self.buffer.clear()
        self.frames_since_prediction = 0

    def __len__(self):
        return len(self.buffer)