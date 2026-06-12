"""
ROI Cropper

Extracts human regions from
YOLO detections.
"""

import cv2
import numpy as np


class ROICropper:

    def __init__(
        self,
        output_size=224,
        padding=10
    ):

        self.output_size = output_size
        self.padding = padding

    def crop(
        self,
        frame,
        bbox
    ):
        """
        Args:
            frame : BGR image
            bbox  : [x1,y1,x2,y2]

        Returns:
            cropped human image
        """

        h, w = frame.shape[:2]

        x1, y1, x2, y2 = bbox

        # Add padding

        x1 = max(
            0,
            x1 - self.padding
        )

        y1 = max(
            0,
            y1 - self.padding
        )

        x2 = min(
            w,
            x2 + self.padding
        )

        y2 = min(
            h,
            y2 + self.padding
        )

        roi = frame[
            y1:y2,
            x1:x2
        ]

        if roi.size == 0:
            return None

        roi = cv2.resize(
            roi,
            (
                self.output_size,
                self.output_size
            )
        )

        return roi

    def crop_all(
        self,
        frame,
        detections
    ):
        """
        Crop all detected persons.

        Returns:
            list of ROIs
        """

        rois = []

        for det in detections:

            roi = self.crop(
                frame,
                det["bbox"]
            )

            if roi is not None:
                rois.append(roi)

        return rois