"""
Webcam Inference

Real-time Human Action Recognition
using the centralized
InferenceEngine.
"""

import cv2

from src.inference.inference_engine import (
    InferenceEngine
)

from src.utils.config_loader import (
    ConfigLoader
)




def main():

    # ---------------------------------
    # Load configuration
    # ---------------------------------

    config = ConfigLoader.load(
        "configs/colab_config.yaml"
    )

    # ---------------------------------
    # Load dataset only to obtain
    # class names
    # ---------------------------------

    

    # ---------------------------------
    # Create inference engine
    # ---------------------------------

    engine = InferenceEngine(
        checkpoint_path=f"{config.checkpoint.save_dir}/best_model.pth",
        sequence_length=config.dataset.sequence_length
    )

    # ---------------------------------
    # Webcam
    # ---------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open webcam."
        )

    print(
        "\nPress Q to quit.\n"
    )

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame = engine.process_frame(
                frame
            )

            cv2.imshow(
                "Human Action Recognition",
                frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
                27
            ):
                break

    except KeyboardInterrupt:

        print(
            "\nStopping webcam inference..."
        )

    finally:

        cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()