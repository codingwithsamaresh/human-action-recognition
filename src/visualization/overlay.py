"""
Overlay Utilities

Draws:
- Bounding Boxes
- Track IDs
- Action Labels
- Confidence Scores
- FPS
"""

import cv2


class OverlayDrawer:

    def __init__(
        self,
        box_thickness=2,
        font_scale=0.6,
        font_thickness=2
    ):
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness

    def draw_track(
        self,
        frame,
        track_id,
        bbox,
        color=(0, 255, 0)
    ):
        """
        Draw track ID and bbox.
        """

        x1, y1, x2, y2 = bbox

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            self.box_thickness
        )

        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            color,
            self.font_thickness
        )

        return frame

    def draw_action(
        self,
        frame,
        bbox,
        action,
        confidence,
        color=(0, 255, 255)
    ):
        """
        Draw action label.
        """

        x1, y1, x2, y2 = bbox

        label = (
            f"{action} "
            f"{confidence:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            color,
            self.font_thickness
        )

        return frame

    def draw_track_with_action(
        self,
        frame,
        track_id,
        bbox,
        action=None,
        confidence=None
    ):
        """
        Draw full information.
        """

        self.draw_track(
            frame,
            track_id,
            bbox
        )

        if action is not None:

            conf = (
                confidence
                if confidence is not None
                else 0.0
            )

            self.draw_action(
                frame,
                bbox,
                action,
                conf
            )

        return frame

    def draw_tracks(
        self,
        frame,
        tracks
    ):
        """
        tracks:
        [
            {
                "track_id": int,
                "bbox": [...]
            }
        ]
        """

        for track in tracks:

            self.draw_track(
                frame,
                track["track_id"],
                track["bbox"]
            )

        return frame

    def draw_fps(
        self,
        frame,
        fps
    ):
        """
        Draw FPS.
        """

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        return frame