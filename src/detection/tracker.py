"""
Centroid-Based Multi-Object Tracker

Maintains stable IDs for detected persons
across consecutive frames.

Output format:

[
    {
        "track_id": 0,
        "bbox": [x1, y1, x2, y2]
    },
    ...
]
"""

from collections import OrderedDict

import numpy as np


class CentroidTracker:

    def __init__(
        self,
        max_disappeared=30,
        max_distance=100
    ):
        """
        Args:
            max_disappeared:
                Frames before removing track

            max_distance:
                Maximum centroid distance
                allowed for matching
        """

        self.next_track_id = 0

        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        self.max_disappeared = (
            max_disappeared
        )

        self.max_distance = (
            max_distance
        )

    def register(
        self,
        bbox
    ):
        """
        Register new object.
        """

        self.objects[
            self.next_track_id
        ] = bbox

        self.disappeared[
            self.next_track_id
        ] = 0

        self.next_track_id += 1

    def deregister(
        self,
        track_id
    ):
        """
        Remove object.
        """

        del self.objects[
            track_id
        ]

        del self.disappeared[
            track_id
        ]

    @staticmethod
    def compute_centroid(
        bbox
    ):
        """
        bbox:
            [x1,y1,x2,y2]
        """

        x1, y1, x2, y2 = bbox

        cx = int(
            (x1 + x2) / 2
        )

        cy = int(
            (y1 + y2) / 2
        )

        return np.array(
            [cx, cy]
        )

    def update(
        self,
        detections
    ):
        """
        Args:
            detections:
                [
                    {
                        "bbox":[...],
                        "confidence":...
                    }
                ]

        Returns:
            [
                {
                    "track_id": id,
                    "bbox": [...]
                }
            ]
        """

        # ---------------------------------
        # No detections
        # ---------------------------------

        if len(detections) == 0:

            for track_id in list(
                self.disappeared.keys()
            ):

                self.disappeared[
                    track_id
                ] += 1

                if (
                    self.disappeared[
                        track_id
                    ]
                    >
                    self.max_disappeared
                ):
                    self.deregister(
                        track_id
                    )

            return self.get_tracks()

        # ---------------------------------
        # Current frame detections
        # ---------------------------------

        input_boxes = [
            d["bbox"]
            for d in detections
        ]

        input_centroids = np.array([
            self.compute_centroid(b)
            for b in input_boxes
        ])

        # ---------------------------------
        # No existing tracks
        # ---------------------------------

        if len(self.objects) == 0:

            for bbox in input_boxes:
                self.register(
                    bbox
                )

            return self.get_tracks()

        # ---------------------------------
        # Existing tracks
        # ---------------------------------

        object_ids = list(
            self.objects.keys()
        )

        object_boxes = list(
            self.objects.values()
        )

        object_centroids = np.array([
            self.compute_centroid(b)
            for b in object_boxes
        ])

        # Distance matrix
        D = np.linalg.norm(
            object_centroids[:, None]
            -
            input_centroids[None, :],
            axis=2
        )

        rows = D.min(
            axis=1
        ).argsort()

        cols = D.argmin(
            axis=1
        )[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(
            rows,
            cols
        ):

            if row in used_rows:
                continue

            if col in used_cols:
                continue

            if (
                D[row, col]
                >
                self.max_distance
            ):
                continue

            track_id = object_ids[row]

            self.objects[
                track_id
            ] = input_boxes[col]

            self.disappeared[
                track_id
            ] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(
            range(
                D.shape[0]
            )
        ) - used_rows

        unused_cols = set(
            range(
                D.shape[1]
            )
        ) - used_cols

        # Existing objects disappeared

        for row in unused_rows:

            track_id = object_ids[row]

            self.disappeared[
                track_id
            ] += 1

            if (
                self.disappeared[
                    track_id
                ]
                >
                self.max_disappeared
            ):
                self.deregister(
                    track_id
                )

        # New objects appeared

        for col in unused_cols:

            self.register(
                input_boxes[col]
            )

        return self.get_tracks()

    def get_tracks(
        self
    ):
        """
        Returns:
            [
                {
                    "track_id": id,
                    "bbox": [...]
                }
            ]
        """

        tracks = []

        for track_id, bbox in (
            self.objects.items()
        ):

            tracks.append(
                {
                    "track_id": track_id,
                    "bbox": bbox
                }
            )

        return tracks