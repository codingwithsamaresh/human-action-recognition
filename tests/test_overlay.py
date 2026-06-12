import cv2
import numpy as np

from src.visualization.overlay import (
    OverlayDrawer
)


def test_overlay():

    frame = np.zeros(
        (720, 1280, 3),
        dtype=np.uint8
    )

    overlay = OverlayDrawer()

    overlay.draw_track_with_action(
        frame=frame,
        track_id=0,
        bbox=[
            200,
            150,
            500,
            600
        ],
        action="Walking",
        confidence=0.98
    )

    overlay.draw_track_with_action(
        frame=frame,
        track_id=1,
        bbox=[
            700,
            180,
            1000,
            620
        ],
        action="Running",
        confidence=0.94
    )

    overlay.draw_fps(
        frame,
        28.5
    )

    cv2.imshow(
        "Overlay Test",
        frame
    )

    print(
        "Press any key..."
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_overlay()