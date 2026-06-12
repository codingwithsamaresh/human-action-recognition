import cv2

from src.detection.yolo_detector import YOLODetector
from src.detection.tracker import CentroidTracker
from src.visualization.overlay import OverlayDrawer


def frame_generator():

    detector = YOLODetector(
        confidence_threshold=0.25
    )

    tracker = CentroidTracker()

    overlay = OverlayDrawer()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    while True:

        success, frame = cap.read()

        if not success:
            break

        detections = detector.detect(
            frame
        )

        tracks = tracker.update(
            detections
        )

        for track in tracks:

            overlay.draw_track(
                frame,
                track["track_id"],
                track["bbox"]
            )

        yield frame

    cap.release()