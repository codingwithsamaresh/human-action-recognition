"""
Inference Engine

Centralized real-time inference pipeline.

Pipeline:

Frame
 ↓
YOLO Detection
 ↓
Tracking
 ↓
ROI Cropper
 ↓
MultiPersonBuffer
 ↓
Action Predictor
 ↓
Alert Manager
 ↓
Overlay
 ↓
Annotated Frame
"""

from src.detection.yolo_detector import (
    YOLODetector
)

from src.detection.tracker import (
    CentroidTracker
)

from src.detection.roi_cropper import (
    ROICropper
)

from src.inference.multi_person_buffer import (
    MultiPersonBuffer
)

from src.inference.action_predictor import (
    ActionPredictor
)

from src.alerts.alert_manager import (
    AlertManager
)

from src.alerts.sound_alert import (
    SoundAlert
)

from src.visualization.overlay import (
    OverlayDrawer
)


class InferenceEngine:

    def __init__(
        self,
        checkpoint_path,
        class_names,
        sequence_length=16
    ):

        self.detector = YOLODetector(
            confidence_threshold=0.25
        )

        self.tracker = (
            CentroidTracker()
        )

        self.cropper = (
            ROICropper(
                output_size=224
            )
        )

        self.buffer_manager = (
            MultiPersonBuffer(
                sequence_length=
                sequence_length
            )
        )

        self.predictor = (
            ActionPredictor(
                checkpoint_path=
                checkpoint_path,

                class_names=
                class_names,

                sequence_length=
                sequence_length
            )
        )

        self.alert_manager = (
            AlertManager()
        )

        self.sound_alert = (
            SoundAlert()
        )

        self.overlay = (
            OverlayDrawer()
        )

        self.track_actions = {}

    def process_frame(
        self,
        frame
    ):

        detections = (
            self.detector.detect(
                frame
            )
        )

        tracks = (
            self.tracker.update(
                detections
            )
        )

        current_tracks = set()

        for track in tracks:

            track_id = (
                track["track_id"]
            )

            bbox = (
                track["bbox"]
            )

            current_tracks.add(
                track_id
            )

            roi = (
                self.cropper.crop(
                    frame,
                    bbox
                )
            )

            if roi is not None:

                self.buffer_manager.update(
                    track_id,
                    roi
                )

                if (
                    self.buffer_manager.ready(
                        track_id
                    )
                ):

                    sequence = (
                        self.buffer_manager
                        .get_sequence(
                            track_id
                        )
                    )

                    result = (
                        self.predictor
                        .predict(
                            sequence
                        )
                    )

                    action = (
                        result["action"]
                    )

                    confidence = (
                        result[
                            "confidence"
                        ]
                    )

                    alert_result = (
                        self.alert_manager
                        .process_prediction(
                            track_id=
                            track_id,

                            action=
                            action,

                            confidence=
                            confidence
                        )
                    )

                    self.track_actions[
                        track_id
                    ] = (
                        alert_result
                    )

                    if (
                        alert_result[
                            "alert"
                        ]
                    ):

                        self.sound_alert.trigger(
                            alert_result[
                                "severity"
                            ]
                        )

            self.overlay.draw_track(
                frame,
                track_id,
                bbox
            )

            if (
                track_id
                in self.track_actions
            ):

                info = (
                    self.track_actions[
                        track_id
                    ]
                )

                self.overlay.draw_action(
                    frame,
                    bbox,
                    info["action"],
                    info[
                        "confidence"
                    ]
                )

        stale_tracks = (
            set(
                self.buffer_manager
                .get_active_ids()
            )
            -
            current_tracks
        )

        for track_id in (
            stale_tracks
        ):

            self.buffer_manager.remove(
                track_id
            )

            if (
                track_id
                in self.track_actions
            ):

                del self.track_actions[
                    track_id
                ]

        return frame