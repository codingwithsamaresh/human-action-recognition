from pathlib import Path

import torch
from torch.utils.data import Dataset

from PIL import Image
from torchvision import transforms


class ActionSequenceDataset(Dataset):
    """
    Dataset for Human Action Recognition.

    Expected Directory Structure
    ----------------------------

    train/
    ├── Class1/
    │   ├── v_Class1_g01_c01/
    │   │   ├── sequence_000000.txt
    │   │   ├── sequence_000004.txt
    │   │   └── ...
    │   └── ...
    │
    ├── Class2/
    └── ...

    Each sequence text file contains the absolute
    path of every frame belonging to that sequence.
    """

    def __init__(
        self,
        sequence_root,
        image_size=224,
        transform=None
    ):
        self.sequence_root = Path(sequence_root)

        # -------------------------------------------------
        # Validate directory
        # -------------------------------------------------

        if not self.sequence_root.exists():
            raise FileNotFoundError(
                f"Sequence directory not found: "
                f"{self.sequence_root}"
            )

        if not self.sequence_root.is_dir():
            raise NotADirectoryError(
                f"Sequence root is not a directory: "
                f"{self.sequence_root}"
            )

        # -------------------------------------------------
        # Image Transform
        # -------------------------------------------------

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size)
                ),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        # -------------------------------------------------
        # Discover Classes
        # -------------------------------------------------

        class_names = sorted([
            directory.name
            for directory in self.sequence_root.iterdir()
            if directory.is_dir()
        ])

        if len(class_names) == 0:
            raise RuntimeError(
                f"No class folders found in "
                f"{self.sequence_root}"
            )

        # Class -> integer label
        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(class_names)
        }

        # Integer label -> class
        self.idx_to_class = {
            idx: class_name
            for class_name, idx
            in self.class_to_idx.items()
        }

        # -------------------------------------------------
        # Discover Sequence Files
        # -------------------------------------------------

        self.samples = []

        for class_name in class_names:

            class_dir = (
                self.sequence_root /
                class_name
            )

            # rglob is required because the structure is:
            #
            # class/
            #     video/
            #         sequence.txt

            sequence_files = sorted(
                class_dir.rglob("*.txt")
            )

            if len(sequence_files) == 0:
                print(
                    f"Warning: No sequence files found "
                    f"for class '{class_name}'"
                )
                continue

            for sequence_file in sequence_files:

                self.samples.append(
                    (
                        sequence_file,
                        self.class_to_idx[class_name]
                    )
                )

        # -------------------------------------------------
        # Validate Samples
        # -------------------------------------------------

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No sequence files found in "
                f"{self.sequence_root}"
            )

        print(
            f"Loaded {len(self.samples)} sequences "
            f"from {len(self.class_to_idx)} classes."
        )

    # =====================================================
    # Dataset Length
    # =====================================================

    def __len__(self):
        return len(self.samples)

    # =====================================================
    # Get Item
    # =====================================================

    def __getitem__(self, idx):

        sequence_file, label = self.samples[idx]

        # -------------------------------------------------
        # Read frame paths
        # -------------------------------------------------

        with open(
            sequence_file,
            "r",
            encoding="utf-8"
        ) as file:

            frame_paths = [
                line.strip()
                for line in file.readlines()
                if line.strip()
            ]

        if len(frame_paths) == 0:
            raise RuntimeError(
                f"Empty sequence file: "
                f"{sequence_file}"
            )

        # -------------------------------------------------
        # Load Frames
        # -------------------------------------------------

        frames = []

        for frame_path in frame_paths:

            frame_path = Path(frame_path)

            if not frame_path.exists():
                raise FileNotFoundError(
                    f"Frame not found:\n"
                    f"{frame_path}\n"
                    f"Referenced by:\n"
                    f"{sequence_file}"
                )

            try:
                image = Image.open(
                    frame_path
                ).convert("RGB")

            except Exception as error:
                raise RuntimeError(
                    f"Could not load image:\n"
                    f"{frame_path}"
                ) from error

            image = self.transform(image)

            frames.append(image)

        # -------------------------------------------------
        # Stack Frames
        # -------------------------------------------------

        # Shape:
        #
        # (T, C, H, W)
        #
        # T = sequence length
        # C = 3
        # H = image height
        # W = image width

        frames = torch.stack(frames)

        # -------------------------------------------------
        # Label
        # -------------------------------------------------

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return frames, label

    # =====================================================
    # Dataset Information
    # =====================================================

    def get_num_classes(self):
        return len(self.class_to_idx)

    def get_class_names(self):
        return list(self.class_to_idx.keys())

    def get_class_to_idx(self):
        return self.class_to_idx

    def get_idx_to_class(self):
        return self.idx_to_class