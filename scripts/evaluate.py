import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from utils.config import *
from utils.dataset import BrainTumorDataset
from utils.transforms import val_transform
from utils.losses import BCEDiceLoss
from utils.metrics import dice_score, iou_score
from models.unet import get_model

# ----------------------------
# Dataset
# ----------------------------

test_dataset = BrainTumorDataset(
    TEST_IMAGES,
    TEST_MASKS,
    transform=val_transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
)

# ----------------------------
# Model
# ----------------------------

model = get_model().to(DEVICE)

checkpoint = torch.load(
    MODEL_DIR / "best_model.pth",
    map_location=DEVICE,
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

criterion = BCEDiceLoss()

total_loss = 0
total_dice = 0
total_iou = 0

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(DEVICE)
        masks = masks.unsqueeze(1).to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs, masks)

        total_loss += loss.item()
        total_dice += dice_score(outputs, masks).item()
        total_iou += iou_score(outputs, masks).item()

print("\n========== TEST RESULTS ==========")
print(f"Test Loss : {total_loss / len(test_loader):.4f}")
print(f"Dice Score: {total_dice / len(test_loader):.4f}")
print(f"IoU Score : {total_iou / len(test_loader):.4f}")