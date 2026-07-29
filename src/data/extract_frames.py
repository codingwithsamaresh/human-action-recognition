"""
Extract frames from videos and save them as images.

Input:
    UCF101 videos

Output:
    processed/frames/
        ├── ApplyEyeMakeup/
        │     ├── v_ApplyEyeMakeup_g01_c01/
        │     ├── ...
        ├── ApplyLipstick/
        └── ...
"""

from pathlib import Path
import cv2
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader


logger = get_logger("frame_extractor")


class FrameExtractor:

    def __init__(
        self,
        input_dir,
        output_dir,
        image_size=224,
        sample_rate=3
    ):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.image_size = image_size
        self.sample_rate = sample_rate

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.total_videos = 0
        self.processed_videos = 0
        self.skipped_videos = 0
        self.failed_videos = 0
        self.total_frames = 0

    def extract_video_frames(
        self,
        video_path
    ):

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

        # -------------------------
        # Resume Support
        # -------------------------

        existing_frames = list(
            save_dir.glob("*.jpg")
        )

        if len(existing_frames) > 0:

            self.skipped_videos += 1

            logger.info(
                f"Skipping {class_name}/{video_name}"
            )

            return

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            self.failed_videos += 1

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

                frame = cv2.resize(
                    frame,
                    (
                        self.image_size,
                        self.image_size
                    ),
                    interpolation=cv2.INTER_AREA
                )

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

        self.processed_videos += 1
        self.total_frames += saved_count

    def run(self):

        video_extensions = (
            ".avi",
            ".mp4",
            ".mov",
            ".mkv"
        )

        video_files = []

        for ext in video_extensions:

            video_files.extend(
                self.input_dir.rglob(
                    f"*{ext}"
                )
            )

        video_files = sorted(
            video_files
        )

        self.total_videos = len(
            video_files
        )

        logger.info(
            f"Found {self.total_videos} videos."
        )

        for video_path in tqdm(
            video_files,
            desc="Extracting Frames"
        ):

            self.extract_video_frames(
                video_path
            )

        logger.info("-" * 60)

        logger.info(
            f"Total Videos     : {self.total_videos}"
        )

        logger.info(
            f"Processed Videos : {self.processed_videos}"
        )

        logger.info(
            f"Skipped Videos   : {self.skipped_videos}"
        )

        logger.info(
            f"Failed Videos    : {self.failed_videos}"
        )

        logger.info(
            f"Frames Saved     : {self.total_frames}"
        )

        logger.info(
            "Frame extraction completed."
        )


if __name__ == "__main__":

    config = ConfigLoader.load(
        "configs/colab_config.yaml"
    )

    extractor = FrameExtractor(
        input_dir=config.dataset.raw_dir,
        output_dir=config.dataset.processed_frames_dir,
        image_size=config.dataset.image_size,
        sample_rate=config.dataset.sample_rate
    )

    extractor.run()