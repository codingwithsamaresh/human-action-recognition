"""
Extract frames from videos and save them as images.

Input:
    data/raw/UCF101/

Output:
    data/processed/frames/
"""

from pathlib import Path
import cv2
from tqdm import tqdm

from src.utils.logger import get_logger



logger = get_logger("frame_extractor")


class FrameExtractor:
    """
    Extract frames from videos.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        sample_rate: int = 1
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def extract_video_frames(self, video_path: Path):
        """
        Extract frames from a single video.
        """

        class_name = video_path.parent.name
        video_name = video_path.stem

        save_dir = (
            self.output_dir /
            class_name /
            video_name
        )

        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(
                f"Could not open {video_path}"
            )
            return

        frame_count = 0
        saved_count = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            if frame_count % self.sample_rate == 0:

                frame_file = (
                    save_dir /
                    f"frame_{saved_count:06d}.jpg"
                )

                cv2.imwrite(
                    str(frame_file),
                    frame
                )

                saved_count += 1

            frame_count += 1

        cap.release()

        logger.info(
            f"{video_name}: "
            f"{saved_count} frames saved"
        )

    def run(self):
        """
        Process all videos.
        """

        video_extensions = (
            ".avi",
            ".mp4",
            ".mov",
            ".mkv"
        )

        video_files = []

        for ext in video_extensions:
            video_files.extend(
                self.input_dir.rglob(f"*{ext}")
            )

        logger.info(
            f"Found {len(video_files)} videos"
        )

        for video_path in tqdm(
            video_files,
            desc="Extracting Frames"
        ):
            self.extract_video_frames(video_path)

        logger.info(
            "Frame extraction complete."
        )


from src.utils.config_loader import ConfigLoader

if __name__ == "__main__":

    config = ConfigLoader.load(
        "configs/colab_config.yaml"
    )

    extractor = FrameExtractor(
        input_dir=config.dataset.raw_dir,
        output_dir=config.dataset.processed_frames_dir,
        sample_rate=5
    )

    extractor.run()