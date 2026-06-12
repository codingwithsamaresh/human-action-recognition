"""
Tracker Unit Test
"""

from src.detection.tracker import (
    CentroidTracker
)


def test_tracker():

    tracker = CentroidTracker(
        max_disappeared=5,
        max_distance=100
    )

    # -------------------------
    # Frame 1
    # -------------------------

    detections = [
        {
            "bbox": [
                100,
                100,
                200,
                300
            ],
            "confidence": 0.9
        }
    ]

    tracks = tracker.update(
        detections
    )

    print(
        "Frame 1:",
        tracks
    )

    assert len(tracks) == 1
    assert tracks[0]["track_id"] == 0

    # -------------------------
    # Frame 2
    # Same person moved
    # -------------------------

    detections = [
        {
            "bbox": [
                110,
                105,
                210,
                305
            ],
            "confidence": 0.95
        }
    ]

    tracks = tracker.update(
        detections
    )

    print(
        "Frame 2:",
        tracks
    )

    assert len(tracks) == 1
    assert tracks[0]["track_id"] == 0

    # -------------------------
    # Frame 3
    # New person appears
    # -------------------------

    detections = [
        {
            "bbox": [
                120,
                110,
                220,
                310
            ],
            "confidence": 0.9
        },
        {
            "bbox": [
                500,
                100,
                600,
                300
            ],
            "confidence": 0.8
        }
    ]

    tracks = tracker.update(
        detections
    )

    print(
        "Frame 3:",
        tracks
    )

    assert len(tracks) == 2

    track_ids = sorted(
        [
            t["track_id"]
            for t in tracks
        ]
    )

    assert track_ids == [0, 1]

    print(
        "\nTracker Test Passed"
    )


if __name__ == "__main__":
    test_tracker()