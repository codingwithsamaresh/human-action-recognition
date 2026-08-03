"""
YOLO Person Detector

Uses YOLOv8n for human detection.
Returns person bounding boxes only.
"""

from ultralytics import YOLO


class YOLODetector:
    """
    YOLOv8-based person detector.
    """

    def __init__(
        self,
        model_path="yolov8n.pt",
        confidence_threshold=0.25
    ):
        self.model = YOLO(model_path)

        self.confidence_threshold = (
            confidence_threshold
        )

        # COCO class ID for person
        self.person_class_id = 0

    def detect(
        self,
        frame
    ):
        """
        Detect persons in an image.

        Parameters
        ----------
        frame : numpy.ndarray
            Input BGR frame.

        Returns
        -------
        list
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float
                }
            ]
        """

        results = self.model(
            frame,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                cls_id = int(
                    box.cls.item()
                )

                if (
                    cls_id
                    != self.person_class_id
                ):
                    continue

                confidence = float(
                    box.conf.item()
                )

                if (
                    confidence
                    < self.confidence_threshold
                ):
                    continue

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                    .tolist()
                )

                detections.append(
                    {
                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2
                        ],
                        "confidence": confidence
                    }
                )

        return detections