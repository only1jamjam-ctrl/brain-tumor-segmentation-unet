from pathlib import Path
from PIL import Image

images_dir = Path("dataset/images")
masks_dir = Path("dataset/masks")

image_files = sorted(images_dir.glob("*"))
mask_files = sorted(masks_dir.glob("*"))

print("=" * 50)
print("BRAIN TUMOR DATASET VERIFICATION")
print("=" * 50)

print(f"\nImages found : {len(image_files)}")
print(f"Masks found  : {len(mask_files)}")

image_names = {f.name for f in image_files}
mask_names = {f.name for f in mask_files}

missing_masks = image_names - mask_names
missing_images = mask_names - image_names

print(f"\nMissing masks : {len(missing_masks)}")
print(f"Missing images: {len(missing_images)}")

if image_files:
    img = Image.open(image_files[0])
    print(f"\nSample image size : {img.size}")
    print(f"Image mode        : {img.mode}")

if mask_files:
    mask = Image.open(mask_files[0])
    print(f"Sample mask size  : {mask.size}")
    print(f"Mask mode         : {mask.mode}")

print("\nDataset verification completed.")