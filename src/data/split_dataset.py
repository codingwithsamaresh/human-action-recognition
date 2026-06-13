from pathlib import Path
import shutil
import random

from src.utils.logger import get_logger

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
            train_ratio
            + val_ratio
            + test_ratio
        )

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Train/Val/Test ratios "
                "must sum to 1.0"
            )

    def copy_file(
        self,
        src,
        dst
    ):
        dst.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            src,
            dst
        )

    def split_class(
        self,
        class_dir
    ):

        class_name = class_dir.name

        sequence_files = sorted(
            class_dir.glob("*.txt")
        )

        if len(sequence_files) == 0:

            logger.warning(
                f"No sequence files found in "
                f"{class_dir}"
            )

            return

        random.shuffle(
            sequence_files
        )

        n = len(sequence_files)

        train_end = int(
            n * self.train_ratio
        )

        val_end = (
            train_end
            +
            int(n * self.val_ratio)
        )

        train_sequences = (
            sequence_files[:train_end]
        )

        val_sequences = (
            sequence_files[
                train_end:val_end
            ]
        )

        test_sequences = (
            sequence_files[val_end:]
        )

        for seq_file in train_sequences:

            self.copy_file(
                seq_file,
                self.train_dir
                / class_name
                / seq_file.name
            )

        for seq_file in val_sequences:

            self.copy_file(
                seq_file,
                self.val_dir
                / class_name
                / seq_file.name
            )

        for seq_file in test_sequences:

            self.copy_file(
                seq_file,
                self.test_dir
                / class_name
                / seq_file.name
            )

        logger.info(
            f"{class_name}: "
            f"train={len(train_sequences)} "
            f"val={len(val_sequences)} "
            f"test={len(test_sequences)}"
        )

    def run(self):

        if not self.source_dir.exists():

            raise FileNotFoundError(
                f"Source directory not found: "
                f"{self.source_dir}"
            )

        classes = [
            x
            for x in self.source_dir.iterdir()
            if x.is_dir()
        ]

        logger.info(
            f"Found {len(classes)} classes"
        )

        for class_dir in classes:

            self.split_class(
                class_dir
            )

        logger.info(
            "Dataset split complete."
        )


if __name__ == "__main__":

    splitter = DatasetSplitter(
        source_dir=
        "data/processed/sequences",

        train_dir=
        "data/train",

        val_dir=
        "data/val",

        test_dir=
        "data/test"
    )

    splitter.run()