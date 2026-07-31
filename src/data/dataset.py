from pathlib import Path

import torch
from torch.utils.data import Dataset

from PIL import Image
from torchvision import transforms

class ActionSequenceDataset(Dataset):
"""
Dataset for Human Action Recognition.

```
Expected structure:

sequence_root/
├── Class1/
│   ├── video_1/
│   │   ├── sequence_000000.txt
│   │   └── ...
│   └── video_2/
│
├── Class2/
└── ...

Each sequence text file contains the absolute paths
of the frames belonging to that sequence.
"""

def __init__(
    self,
    sequence_root,
    image_size=224,
    transform=None
):

    self.sequence_root = Path(sequence_root)

    if not self.sequence_root.exists():
        raise FileNotFoundError(
            f"Sequence directory not found: "
            f"{self.sequence_root}"
        )

    # ==================================================
    # Transform
    # ==================================================

    if transform is None:

        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor(),
        ])

    else:

        self.transform = transform

    # ==================================================
    # Discover Classes
    # ==================================================

    class_names = sorted([
        directory.name
        for directory in self.sequence_root.iterdir()
        if directory.is_dir()
    ])

    if not class_names:

        raise RuntimeError(
            f"No class folders found in "
            f"{self.sequence_root}"
        )

    self.class_to_idx = {
        class_name: index
        for index, class_name
        in enumerate(class_names)
    }

    self.idx_to_class = {
        index: class_name
        for class_name, index
        in self.class_to_idx.items()
    }

    # ==================================================
    # Discover Sequences
    # ==================================================

    self.samples = []

    for class_name in class_names:

        class_dir = (
            self.sequence_root /
            class_name
        )

        sequence_files = sorted(
            class_dir.rglob("*.txt")
        )

        if not sequence_files:

            print(
                f"Warning: No sequences found "
                f"for class '{class_name}'"
            )

            continue

        label = self.class_to_idx[
            class_name
        ]

        for sequence_file in sequence_files:

            self.samples.append(
                (
                    sequence_file,
                    label
                )
            )

    if not self.samples:

        raise RuntimeError(
            f"No sequence files found in "
            f"{self.sequence_root}"
        )

    # ==================================================
    # Preload Frame Paths
    # ==================================================
    #
    # The old implementation opened every .txt file
    # inside __getitem__().
    #
    # With more than 100k sequences, this creates a
    # significant amount of unnecessary filesystem I/O.
    #
    # We therefore read all sequence metadata once.
    # ==================================================

    print(
        f"Indexing frame paths for "
        f"{len(self.samples)} sequences..."
    )

    indexed_samples = []

    for sequence_file, label in self.samples:

        with open(
            sequence_file,
            "r",
            encoding="utf-8"
        ) as file:

            frame_paths = [
                line.strip()
                for line in file
                if line.strip()
            ]

        if not frame_paths:

            continue

        indexed_samples.append(
            (
                frame_paths,
                label
            )
        )

    self.samples = indexed_samples

    if not self.samples:

        raise RuntimeError(
            f"No valid sequence samples found in "
            f"{self.sequence_root}"
        )

    print(
        f"Loaded "
        f"{len(self.samples)} sequences "
        f"from "
        f"{len(self.class_to_idx)} classes."
    )

# ==================================================
# Dataset Interface
# ==================================================

def __len__(self):

    return len(self.samples)

def __getitem__(self, index):

    frame_paths, label = (
        self.samples[index]
    )

    frames = []

    for frame_path in frame_paths:

        frame_path = Path(
            frame_path
        )

        if not frame_path.exists():

            raise FileNotFoundError(
                f"Frame not found:\n"
                f"{frame_path}"
            )

        with Image.open(
            frame_path
        ) as image:

            image = image.convert(
                "RGB"
            )

            image = self.transform(
                image
            )

        frames.append(
            image
        )

    frames = torch.stack(
        frames,
        dim=0
    )

    label = torch.tensor(
        label,
        dtype=torch.long
    )

    return (
        frames,
        label
    )

# ==================================================
# Metadata
# ==================================================

def get_num_classes(self):

    return len(
        self.class_to_idx
    )

def get_class_names(self):

    return list(
        self.class_to_idx.keys()
    )

def get_class_indices(self):

    class_indices = {
        class_index: []
        for class_index
        in range(
            self.get_num_classes()
        )
    }

    for sample_index, (
        _,
        label
    ) in enumerate(
        self.samples
    ):

        class_indices[
            label
        ].append(
            sample_index
        )

    return class_indices
```
