import numpy as np

from src.inference.frame_buffer import FrameBuffer


def test_buffer():

    buffer = FrameBuffer(maxlen=4)

    for i in range(4):
        frame = np.random.rand(224, 224, 3)
        buffer.add(frame)

    assert buffer.is_full()

    seq = buffer.get_sequence()

    assert len(seq) == 4

    print("FrameBuffer Test Passed")


if __name__ == "__main__":
    test_buffer()