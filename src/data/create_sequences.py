from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger("sequence_generator")


class SequenceGenerator:

    def __init__(
        self,
        input_dir,
        output_dir,
        sequence_length=16,
        stride=8
    ):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.sequence_length = sequence_length
        self.stride = stride

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

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

            logger.warning(
                f"{video_folder.name} skipped "
                f"(only {total_frames} frames)"
            )
            return

        save_dir = (
            self.output_dir /
            class_name
        )

        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

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
                f"sequence_{sequence_count:06d}.txt"
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

        logger.info(
            f"{video_folder.name}: "
            f"{sequence_count} sequences created"
        )

    def run(self):

        classes = [
            x for x in self.input_dir.iterdir()
            if x.is_dir()
        ]

        logger.info(
            f"Found {len(classes)} classes"
        )

        for class_dir in classes:

            class_name = class_dir.name

            video_folders = [
                x for x in class_dir.iterdir()
                if x.is_dir()
            ]

            for video_folder in video_folders:

                self.process_video_folder(
                    class_name,
                    video_folder
                )

        logger.info(
            "Sequence generation complete."
        )


from src.utils.config_loader import ConfigLoader

if __name__ == "__main__":

    config = ConfigLoader.load(
        "configs/colab_config.yaml"
    )

    generator = SequenceGenerator(
        input_dir=config.dataset.processed_frames_dir,
        output_dir=config.dataset.processed_sequences_dir,
        sequence_length=config.dataset.sequence_length,
        stride=8
    )

    generator.run()