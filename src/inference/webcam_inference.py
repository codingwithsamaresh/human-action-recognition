"""
Webcam Inference

Real-time Human Action Recognition
using the centralized
InferenceEngine.
"""

import cv2

from src.inference.inference_engine import (
    InferenceEngine
)


def main():

    engine = InferenceEngine(
        checkpoint_path=
        "weights/checkpoints/best_model.pth",

        class_names=[
            "TestAction"
        ],

        sequence_length=16
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open webcam."
        )

    print(
        "Press Q to quit."
    )

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = (
            engine.process_frame(
                frame
            )
        )

        cv2.imshow(
            "Human Action Recognition",
            frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()