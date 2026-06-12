from src.data.dataset import ActionSequenceDataset

dataset = ActionSequenceDataset(
    "data/processed/sequences"
)

print("=" * 50)

print("Dataset Size:", len(dataset))

frames, label = dataset[0]

print("Frames Shape:", frames.shape)

print("Label:", label)

print("=" * 50)