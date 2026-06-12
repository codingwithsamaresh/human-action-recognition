from PIL import Image

from src.data.augmentations import (
    get_train_transforms
)

image = Image.open(
    "data/processed/frames/TestAction/videoplayback/frame_000000.jpg"
).convert("RGB")

transform = get_train_transforms()

tensor = transform(image)

print("=" * 50)
print("Tensor Shape:", tensor.shape)
print("Tensor Type :", tensor.dtype)
print("=" * 50)