import cv2

from src.detection.yolo_detector import YOLODetector
from src.detection.roi_cropper import ROICropper


def test_roi_cropper():

    detector = YOLODetector(
        confidence_threshold=0.25
    )

    cropper = ROICropper()

    cap = cv2.VideoCapture(0)

    print("Press SPACE to capture.")
    print("Press Q to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow(
            "Webcam",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            break

        if key in [ord("q"), ord("Q")]:
            cap.release()
            cv2.destroyAllWindows()
            return

    detections = detector.detect(
        frame
    )

    rois = cropper.crop_all(
        frame,
        detections
    )

    print(
        f"Detections: {len(detections)}"
    )

    print(
        f"ROIs: {len(rois)}"
    )

    for i, roi in enumerate(rois):

        cv2.imshow(
            f"ROI {i}",
            roi
        )

    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_roi_cropper()