import numpy as np

from src.inference.sliding_window import SlidingWindow


def test_window():

    window = SlidingWindow(
        window_size=16,
        stride=4
    )

    predictions = 0

    for _ in range(25):

        frame = np.random.rand(
            224,
            224,
            3
        )

        window.add(frame)

        if window.ready():

            seq = window.get_window()

            assert len(seq) == 16

            predictions += 1

    print(
        f"Predictions Triggered: {predictions}"
    )


if __name__ == "__main__":
    test_window()