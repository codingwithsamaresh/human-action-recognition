from pathlib import Path

import torch
from torch.utils.data import Dataset

from PIL import Image
from torchvision import transforms


class ActionSequenceDataset(Dataset):
    """
    Dataset for Human Action Recognition.

    Expected Structure:
    data/processed/sequences/
    ├── Class1/
    │   ├── sequence_000000.txt
    │   ├── sequence_000001.txt
    │   └── ...
    ├── Class2/
    │   ├── sequence_000000.txt
    │   └── ...
    """

    def __init__(
        self,
        sequence_root,
        image_size=224,
        transform=None
    ):
        """
        Args:
            sequence_root (str): Root directory containing sequence txt files.
            image_size (int): Resize dimension for default transform.
            transform: torchvision transform pipeline.
        """

        self.sequence_root = Path(sequence_root)

        if not self.sequence_root.exists():
            raise FileNotFoundError(
                f"Sequence directory not found: {self.sequence_root}"
            )

        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        self.samples = []

        # Discover classes
        class_names = sorted([
            d.name
            for d in self.sequence_root.iterdir()
            if d.is_dir()
        ])

        if len(class_names) == 0:
            raise RuntimeError(
                f"No class folders found in {self.sequence_root}"
            )

        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(class_names)
        }

        self.idx_to_class = {
            idx: class_name
            for class_name, idx in self.class_to_idx.items()
        }

        # Gather all sequence files
        for class_name in class_names:

            class_dir = self.sequence_root / class_name

            sequence_files = sorted(
                class_dir.glob("*.txt")
            )

            for sequence_file in sequence_files:

                self.samples.append(
                    (
                        sequence_file,
                        self.class_to_idx[class_name]
                    )
                )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No sequence files found in {self.sequence_root}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sequence_file, label = self.samples[idx]

        # Read frame paths
        with open(
            sequence_file,
            "r",
            encoding="utf-8"
        ) as f:

            frame_paths = [
                line.strip()
                for line in f.readlines()
                if line.strip()
            ]

        frames = []

        for frame_path in frame_paths:

            frame_path = Path(frame_path)

            if not frame_path.exists():
                raise FileNotFoundError(
                    f"Frame not found: {frame_path}"
                )

            image = Image.open(
                frame_path
            ).convert("RGB")

            image = self.transform(image)

            frames.append(image)

        # Shape:
        # (T, C, H, W)
        frames = torch.stack(frames)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return frames, label

    def get_num_classes(self):
        return len(self.class_to_idx)

    def get_class_names(self):
        return list(self.class_to_idx.keys())