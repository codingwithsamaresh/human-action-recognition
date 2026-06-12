"""
Video Inference

Runs Human Action Recognition
on a saved video file.
"""

from pathlib import Path

import cv2

from src.inference.sliding_window import SlidingWindow
from src.inference.action_predictor import ActionPredictor


def run_video_inference(
    video_path,
    checkpoint_path,
    class_names,
    sequence_length=16,
    stride=4
):
    """
    Run action recognition on a video file.
    """

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    predictor = ActionPredictor(
        checkpoint_path=checkpoint_path,
        class_names=class_names,
        sequence_length=sequence_length
    )

    sliding_window = SlidingWindow(
        window_size=sequence_length,
        stride=stride
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open video."
        )

    current_action = "Waiting..."
    current_confidence = 0.0

    print(
        f"Processing: {video_path}"
    )

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            sliding_window.add(frame)

            if sliding_window.ready():

                frames = (
                    sliding_window.get_window()
                )

                result = predictor.predict(
                    frames
                )

                current_action = result[
                    "action"
                ]

                current_confidence = result[
                    "confidence"
                ]

            cv2.putText(
                frame,
                f"Action: {current_action}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {current_confidence:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "Video Inference",
                frame
            )

            key = (
                cv2.waitKey(30)
                & 0xFF
            )

            if key in [
                ord("q"),
                ord("Q"),
                27
            ]:
                break

    except KeyboardInterrupt:

        print(
            "\nStopping video inference..."
        )

    finally:

        cap.release()

        cv2.destroyAllWindows()


def main():

    run_video_inference(
        video_path="sample.mp4",
        checkpoint_path=
        "weights/checkpoints/best_model.pth",

        class_names=[
            "TestAction"
        ],

        sequence_length=16,
        stride=4
    )


if __name__ == "__main__":
    main()