"""
Live YOLO Detector Test

Press Q to quit.
"""

import cv2

from src.detection.yolo_detector import (
    YOLODetector
)


def test_detector_live():

    detector = YOLODetector(
        confidence_threshold=0.05
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    print(
        "Starting live detection..."
    )

    print(
        "Press Q to quit."
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        detections = detector.detect(
            frame
        )

        print(
            f"Detections: {len(detections)}",
            end="\r"
        )

        for det in detections:

            x1, y1, x2, y2 = (
                det["bbox"]
            )

            confidence = (
                det["confidence"]
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Person {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "YOLO Live Test",
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
    test_detector_live()