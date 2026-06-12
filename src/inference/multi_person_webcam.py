"""
Multi-Person Webcam Inference

Pipeline:

Webcam
  ↓
YOLO Detection
  ↓
Tracking
  ↓
ROI Cropping
  ↓
Multi-Person Buffers
  ↓
Action Recognition
  ↓
Alerts
  ↓
Overlay Rendering
"""

import cv2

from src.detection.yolo_detector import YOLODetector
from src.detection.roi_cropper import ROICropper
from src.detection.tracker import CentroidTracker

from src.inference.multi_person_buffer import (
    MultiPersonBuffer
)

from src.inference.action_predictor import (
    ActionPredictor
)

from src.visualization.overlay import (
    OverlayDrawer
)

from src.alerts.alert_manager import (
    AlertManager
)

from src.alerts.sound_alert import (
    SoundAlert
)


def main():

    # ---------------------------------
    # Configuration
    # ---------------------------------

    sequence_length = 16

    checkpoint_path = (
        "weights/checkpoints/best_model.pth"
    )

    class_names = [
        "TestAction"
    ]

    # ---------------------------------
    # Components
    # ---------------------------------

    detector = YOLODetector(
        confidence_threshold=0.25
    )

    tracker = CentroidTracker()

    cropper = ROICropper(
        output_size=224
    )

    buffers = MultiPersonBuffer(
        sequence_length=sequence_length
    )

    predictor = ActionPredictor(
        checkpoint_path=checkpoint_path,
        class_names=class_names,
        sequence_length=sequence_length
    )

    overlay = OverlayDrawer()

    alert_manager = AlertManager()

    sound_alert = SoundAlert()

    # ---------------------------------
    # Store latest prediction
    # per track
    # ---------------------------------

    track_predictions = {}

    # ---------------------------------
    # Webcam
    # ---------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open webcam."
        )

    print("\nPress Q to quit.\n")

    while True:

        success, frame = cap.read()

        if not success:
            break

        # ---------------------------------
        # Detection
        # ---------------------------------

        detections = detector.detect(
            frame
        )

        # ---------------------------------
        # Tracking
        # ---------------------------------

        tracks = tracker.update(
            detections
        )

        # ---------------------------------
        # Process each track
        # ---------------------------------

        for track in tracks:

            track_id = track[
                "track_id"
            ]

            bbox = track[
                "bbox"
            ]

            roi = cropper.crop(
                frame,
                bbox
            )

            if roi is None:
                continue

            # -----------------------------
            # Update buffer
            # -----------------------------

            buffers.update(
                track_id,
                roi
            )

            # -----------------------------
            # Predict action
            # -----------------------------

            if buffers.ready(
                track_id
            ):

                sequence = (
                    buffers.get_sequence(
                        track_id
                    )
                )

                result = predictor.predict(
                    sequence
                )

                action = result[
                    "action"
                ]

                confidence = result[
                    "confidence"
                ]

                track_predictions[
                    track_id
                ] = (
                    action,
                    confidence
                )

                # -------------------------
                # Alert Processing
                # -------------------------

                alert_result = (
                    alert_manager
                    .process_prediction(
                        track_id,
                        action,
                        confidence
                    )
                )

                if alert_result[
                    "alert"
                ]:

                    sound_alert.trigger(
                        alert_result[
                            "severity"
                        ]
                    )

        # ---------------------------------
        # Draw overlays
        # ---------------------------------

        for track in tracks:

            track_id = track[
                "track_id"
            ]

            bbox = track[
                "bbox"
            ]

            action = None
            confidence = None

            if (
                track_id
                in track_predictions
            ):

                (
                    action,
                    confidence
                ) = (
                    track_predictions[
                        track_id
                    ]
                )

            overlay.draw_track_with_action(
                frame,
                track_id,
                bbox,
                action,
                confidence
            )

        # ---------------------------------
        # Display
        # ---------------------------------

        cv2.imshow(
            "Multi-Person HAR",
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