"""
Generate fixed-length frame sequences for Human Action Recognition.

Input:
processed/
└── frames/
    ├── Basketball/
    │   ├── v_Basketball_g01_c01/
    │   ├── v_Basketball_g01_c02/
    │   └── ...
    └── ...

Output:
processed/
└── sequences/
    ├── Basketball/
    │   ├── v_Basketball_g01_c01/
    │   │   ├── sequence_000000.txt
    │   │   ├── sequence_000004.txt
    │   │   └── ...
    │   └── ...
"""

from pathlib import Path

from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader

logger = get_logger("sequence_generator")


class SequenceGenerator:

    def __init__(
        self,
        input_dir,
        output_dir,
        sequence_length=8,
        stride=4
    ):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.sequence_length = sequence_length
        self.stride = stride

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.total_videos = 0
        self.processed_videos = 0
        self.skipped_videos = 0
        self.failed_videos = 0
        self.total_sequences = 0

    def process_video_folder(
        self,
        class_name,
        video_folder
    ):

        frame_files = sorted(
            video_folder.glob("*.jpg")
        )

        total_frames = len(frame_files)

        if total_frames < self.sequence_length:

            self.failed_videos += 1

            logger.warning(
                f"{class_name}/{video_folder.name} "
                f"skipped "
                f"({total_frames} frames)"
            )

            return

        save_dir = (
            self.output_dir /
            class_name /
            video_folder.name
        )

        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # -------------------------
        # Resume Support
        # -------------------------

        existing_sequences = list(
            save_dir.glob("*.txt")
        )

        if existing_sequences:

            self.skipped_videos += 1

            logger.info(
                f"Skipping "
                f"{class_name}/{video_folder.name}"
            )

            return

        sequence_count = 0

        for start_idx in range(
            0,
            total_frames - self.sequence_length + 1,
            self.stride
        ):

            sequence_frames = frame_files[
                start_idx:
                start_idx + self.sequence_length
            ]

            sequence_file = (
                save_dir /
                f"sequence_{start_idx:06d}.txt"
            )

            with open(
                sequence_file,
                "w",
                encoding="utf-8"
            ) as f:

                for frame in sequence_frames:

                    f.write(
                        str(frame) + "\n"
                    )

            sequence_count += 1

        self.processed_videos += 1
        self.total_sequences += sequence_count

        logger.info(
            f"{class_name}/{video_folder.name}: "
            f"{sequence_count} sequences"
        )

    def run(self):

        classes = sorted([
            x
            for x in self.input_dir.iterdir()
            if x.is_dir()
        ])

        logger.info(
            f"Found {len(classes)} classes."
        )

        for class_dir in classes:

            class_name = class_dir.name

            video_folders = sorted([
                x
                for x in class_dir.iterdir()
                if x.is_dir()
            ])

            self.total_videos += len(
                video_folders
            )

            for video_folder in video_folders:

                self.process_video_folder(
                    class_name,
                    video_folder
                )

        logger.info("-" * 60)

        logger.info(
            f"Videos Found      : "
            f"{self.total_videos}"
        )

        logger.info(
            f"Processed Videos  : "
            f"{self.processed_videos}"
        )

        logger.info(
            f"Skipped Videos    : "
            f"{self.skipped_videos}"
        )

        logger.info(
            f"Failed Videos     : "
            f"{self.failed_videos}"
        )

        logger.info(
            f"Sequences Created : "
            f"{self.total_sequences}"
        )

        logger.info(
            "Sequence generation complete."
        )


if __name__ == "__main__":

    config = ConfigLoader.load(
        "configs/colab_config.yaml"
    )

    generator = SequenceGenerator(
        input_dir=config.dataset.processed_frames_dir,
        output_dir=config.dataset.processed_sequences_dir,
        sequence_length=config.dataset.sequence_length,
        stride=config.dataset.sequence_stride
    )

    generator.run()