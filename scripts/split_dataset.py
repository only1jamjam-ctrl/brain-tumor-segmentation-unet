from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# Paths
images_dir = Path("dataset/images")
masks_dir = Path("dataset/masks")
output_dir = Path("data")

# Create folders
for split in ["train", "val", "test"]:
    (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)

# Get image files
image_files = sorted(images_dir.glob("*"))

# Split dataset
train_files, temp_files = train_test_split(
    image_files,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

val_files, test_files = train_test_split(
    temp_files,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

splits = {
    "train": train_files,
    "val": val_files,
    "test": test_files
}

# Copy files
for split_name, files in splits.items():
    for image_path in files:
        mask_path = masks_dir / image_path.name

        shutil.copy2(
            image_path,
            output_dir / split_name / "images" / image_path.name
        )

        shutil.copy2(
            mask_path,
            output_dir / split_name / "masks" / mask_path.name
        )

print("=" * 50)
print("DATASET SPLIT COMPLETED")
print("=" * 50)

for split_name, files in splits.items():
    print(f"{split_name.upper():<10}: {len(files)} images")

print("=" * 50)