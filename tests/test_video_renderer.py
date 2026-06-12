import cv2
import numpy as np

from src.visualization.video_renderer import (
    VideoRenderer
)


def test_video_renderer():

    width = 640
    height = 480
    fps = 30

    output_path = (
        "outputs/videos/test_output.mp4"
    )

    renderer = VideoRenderer(
        output_path=output_path,
        fps=fps,
        frame_width=width,
        frame_height=height
    )

    for i in range(120):

        frame = np.zeros(
            (height, width, 3),
            dtype=np.uint8
        )

        cv2.putText(
            frame,
            f"Frame {i}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        renderer.write(frame)

    renderer.release()

    print(
        f"Saved video:\n{output_path}"
    )


if __name__ == "__main__":
    test_video_renderer()