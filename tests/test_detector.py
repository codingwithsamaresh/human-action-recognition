import cv2

from src.detection.yolo_detector import YOLODetector


def test_detector():

    detector = YOLODetector(
        confidence_threshold=0.25
    )

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        detections = detector.detect(frame)

        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            conf = det["confidence"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Person {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "YOLO Detector Test",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_detector()