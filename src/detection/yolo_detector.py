"""
YOLO Person Detector

Uses YOLOv8n for human detection.
Returns person bounding boxes only.
"""

from ultralytics import YOLO


class YOLODetector:

    def __init__(
        self,
        model_path="yolov8n.pt",
        confidence_threshold=0.05
    ):

        self.model = YOLO(model_path)

        self.confidence_threshold = (
            confidence_threshold
        )

        # COCO person class
        self.person_class_id = 0

    def detect(
        self,
        frame
    ):
        """
        Args:
            frame (numpy.ndarray)

        Returns:
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

        # -------------------------
        # DEBUG INFO
        # -------------------------

        try:

            print(
                "\nClasses:",
                results[0].names
            )

            if results[0].boxes is not None:

                print(
                    "Raw boxes:",
                    len(results[0].boxes)
                )

        except Exception as e:

            print(
                "Debug error:",
                e
            )

        # -------------------------

        detections = []

        for result in results:

            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                cls_id = int(
                    box.cls.item()
                )

                confidence = float(
                    box.conf.item()
                )

                print(
                    f"class={cls_id} "
                    f"conf={confidence:.3f}"
                )

                # Keep only persons

                if (
                    cls_id
                    != self.person_class_id
                ):
                    continue

                # Confidence filter

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
                        "confidence":
                        confidence
                    }
                )

        return detections