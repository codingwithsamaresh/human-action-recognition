import numpy as np

from src.inference.action_predictor import (
    ActionPredictor
)


def test_predictor():

    predictor = ActionPredictor(
        checkpoint_path="weights/checkpoints/best_model.pth",
        class_names=["TestAction"],
        sequence_length=16
    )

    frames = []

    for _ in range(16):

        frame = np.random.randint(
            0,
            255,
            (480, 640, 3),
            dtype=np.uint8
        )

        frames.append(frame)

    result = predictor.predict(
        frames
    )

    print(result)


if __name__ == "__main__":
    test_predictor()