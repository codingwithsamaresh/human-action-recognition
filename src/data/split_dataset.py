"""
Split generated sequences into train, validation, and test sets.

Input:
processed/
└── sequences/
    ├── Basketball/
    │   ├── v_Basketball_g01_c01/
    │   │   ├── sequence_000000.txt
    │   │   └── ...
    │   ├── v_Basketball_g01_c02/
    │   └── ...
    └── ...

Output:
train/
val/
test/
"""

from pathlib import Path
import shutil
import random

from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader

logger = get_logger("dataset_splitter")


class DatasetSplitter:

    def __init__(
        self,
        source_dir,
        train_dir,
        val_dir,
        test_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    ):

        self.source_dir = Path(source_dir)

        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.test_dir = Path(test_dir)

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        random.seed(seed)

        total = (
            train_ratio +
            val_ratio +
            test_ratio
        )

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Train/Val/Test ratios must sum to 1."
            )

        self.total_sequences = 0

    def copy_sequence(
        self,
        src_file,
        dst_file
    ):

        dst_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            src_file,
            dst_file
        )

        self.total_sequences += 1

    def copy_video(
        self,
        video_dir,
        destination_root,
        class_name
    ):

        sequence_files = sorted(
            video_dir.glob("*.txt")
        )

        for seq_file in sequence_files:

            destination = (
                destination_root /
                class_name /
                video_dir.name /
                seq_file.name
            )

            self.copy_sequence(
                seq_file,
                destination
            )

        return len(sequence_files)

    def split_class(
        self,
        class_dir
    ):

        class_name = class_dir.name

        video_dirs = sorted([
            x
            for x in class_dir.iterdir()
            if x.is_dir()
        ])

        if len(video_dirs) == 0:

            logger.warning(
                f"No videos found for {class_name}"
            )

            return

        random.shuffle(video_dirs)

        n = len(video_dirs)

        train_end = int(
            n * self.train_ratio
        )

        val_end = (
            train_end +
            int(n * self.val_ratio)
        )

        train_videos = video_dirs[:train_end]
        val_videos = video_dirs[train_end:val_end]
        test_videos = video_dirs[val_end:]

        train_sequences = 0
        val_sequences = 0
        test_sequences = 0

        for video_dir in train_videos:

            train_sequences += self.copy_video(
                video_dir,
                self.train_dir,
                class_name
            )

        for video_dir in val_videos:

            val_sequences += self.copy_video(
                video_dir,
                self.val_dir,
                class_name
            )

        for video_dir in test_videos:

            test_sequences += self.copy_video(
                video_dir,
                self.test_dir,
                class_name
            )

        logger.info(
            f"{class_name}: "
            f"videos="
            f"{len(train_videos)}/"
            f"{len(val_videos)}/"
            f"{len(test_videos)} | "
            f"sequences="
            f"{train_sequences}/"
            f"{val_sequences}/"
            f"{test_sequences}"
        )

    def run(self):

        if not self.source_dir.exists():

            raise FileNotFoundError(
                f"Source directory not found: "
                f"{self.source_dir}"
            )

        for directory in [
            self.train_dir,
            self.val_dir,
            self.test_dir
        ]:

            if directory.exists():

                shutil.rmtree(directory)

            directory.mkdir(
                parents=True,
                exist_ok=True
            )

        class_dirs = sorted([
            x
            for x in self.source_dir.iterdir()
            if x.is_dir()
        ])

        logger.info(
            f"Found {len(class_dirs)} classes."
        )

        for class_dir in class_dirs:

            self.split_class(
                class_dir
            )

        logger.info("-" * 60)

        logger.info(
            f"Total sequences copied: "
            f"{self.total_sequences}"
        )

        logger.info(
            "Dataset splitting complete."
        )


if __name__ == "__main__":

    config = ConfigLoader.load(
        "configs/colab_config.yaml"
    )

    splitter = DatasetSplitter(
        source_dir=config.dataset.processed_sequences_dir,

        train_dir=config.dataset.train_dir,

        val_dir=config.dataset.val_dir,

        test_dir=config.dataset.test_dir,

        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,

        seed=config.seed
    )

    splitter.run()