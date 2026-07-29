from pathlib import Path
import torch

# Dataset
TRAIN_IMAGES = "data/train/images"
TRAIN_MASKS = "data/train/masks"

VAL_IMAGES = "data/val/images"
VAL_MASKS = "data/val/masks"

TEST_IMAGES = "data/test/images"
TEST_MASKS = "data/test/masks"

# Training
BATCH_SIZE = 2
EPOCHS = 50
LEARNING_RATE = 1e-4
IMAGE_SIZE = 256

# Early stopping
PATIENCE = 10

# Random seed
SEED = 42

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output folders
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)