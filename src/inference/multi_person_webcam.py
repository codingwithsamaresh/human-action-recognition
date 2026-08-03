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

from src.inference.multi_person_buffer import MultiPersonBuffer
from src.inference.action_predictor import ActionPredictor

from src.visualization.overlay import OverlayDrawer

from src.alerts.alert_manager import AlertManager
from src.alerts.sound_alert import SoundAlert

from src.utils.config_loader import ConfigLoader



def main():

    # ---------------------------------
    # Load configuration
    # ---------------------------------

    config = ConfigLoader.load(
        "configs/colab_config.yaml"
    )

    

    sequence_length = config.dataset.sequence_length

    checkpoint_path = (
        f"{config.checkpoint.save_dir}/best_model.pth"
    )

    # ---------------------------------
    # Components
    # ---------------------------------

    detector = YOLODetector(
        confidence_threshold=0.25
    )

    tracker = CentroidTracker()

    cropper = ROICropper(
        output_size=config.dataset.image_size
    )

    buffers = MultiPersonBuffer(
        sequence_length=sequence_length
    )

    predictor = ActionPredictor(
        checkpoint_path=checkpoint_path,
        sequence_length=sequence_length,
        image_size=config.dataset.image_size
    )

    overlay = OverlayDrawer()

    alert_manager = AlertManager()

    sound_alert = SoundAlert()

    # ---------------------------------
    # Store latest prediction
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
        # Person Detection
        # ---------------------------------

        detections = detector.detect(frame)

        # ---------------------------------
        # Tracking
        # ---------------------------------

        tracks = tracker.update(detections)

        # ---------------------------------
        # Process each tracked person
        # ---------------------------------

        for track in tracks:

            track_id = track["track_id"]
            bbox = track["bbox"]

            roi = cropper.crop(
                frame,
                bbox
            )

            if roi is None:
                continue

            buffers.update(
                track_id,
                roi
            )

            if buffers.ready(track_id):

                sequence = buffers.get_sequence(
                    track_id
                )

                result = predictor.predict(
                    sequence
                )

                action = result["action"]
                confidence = result["confidence"]

                track_predictions[
                    track_id
                ] = (
                    action,
                    confidence
                )

                alert_result = (
                    alert_manager.process_prediction(
                        track_id,
                        action,
                        confidence
                    )
                )

                if alert_result["alert"]:

                    sound_alert.trigger(
                        alert_result["severity"]
                    )

        # ---------------------------------
        # Draw overlays
        # ---------------------------------

        for track in tracks:

            track_id = track["track_id"]
            bbox = track["bbox"]

            action = None
            confidence = None

            if track_id in track_predictions:

                action, confidence = (
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

        cv2.imshow(
            "Multi-Person Human Action Recognition",
            frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key in (
            ord("q"),
            ord("Q"),
            27
        ):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()