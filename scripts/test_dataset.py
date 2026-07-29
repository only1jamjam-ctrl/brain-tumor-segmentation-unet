import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.dataset import BrainTumorDataset
from utils.transforms import train_transform

dataset = BrainTumorDataset(
    image_dir="data/train/images",
    mask_dir="data/train/masks",
    transform=train_transform,
)

print("=" * 50)
print("DATASET TEST")
print("=" * 50)

print(f"Training samples : {len(dataset)}")

image, mask = dataset[0]

print(f"Image shape : {image.shape}")
print(f"Mask shape  : {mask.shape}")

print(f"Image dtype : {image.dtype}")
print(f"Mask dtype  : {mask.dtype}")