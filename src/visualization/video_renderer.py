"""
Video Renderer

Handles saving annotated frames
to output video files.
"""

from pathlib import Path

import cv2


class VideoRenderer:

    def __init__(
        self,
        output_path,
        fps,
        frame_width,
        frame_height,
        codec="mp4v"
    ):
        """
        Args:
            output_path (str)
            fps (float)
            frame_width (int)
            frame_height (int)
            codec (str)
        """

        self.output_path = Path(output_path)

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fourcc = cv2.VideoWriter_fourcc(
            *codec
        )

        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            (
                frame_width,
                frame_height
            )
        )

        if not self.writer.isOpened():
            raise RuntimeError(
                f"Could not create video: "
                f"{self.output_path}"
            )

    def write(
        self,
        frame
    ):
        """
        Write one frame.
        """

        self.writer.write(frame)

    def release(self):
        """
        Release video writer.
        """

        if self.writer is not None:
            self.writer.release()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb
    ):
        self.release()