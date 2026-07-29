import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.config import IMAGE_SIZE

IMAGE_SIZE = 256

train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),

    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),

    A.Normalize(mean=(0.0,), std=(1.0,)),

    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),

    A.Normalize(mean=(0.0,), std=(1.0,)),

    ToTensorV2(),
])