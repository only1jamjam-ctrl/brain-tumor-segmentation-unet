import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from utils.losses import BCEDiceLoss
from utils.metrics import dice_score, iou_score

pred = torch.randn(2, 1, 512, 512)
target = torch.randint(0, 2, (2, 1, 512, 512)).float()

criterion = BCEDiceLoss()

loss = criterion(pred, target)

dice = dice_score(pred, target)

iou = iou_score(pred, target)

print("=" * 50)
print("LOSS & METRICS TEST")
print("=" * 50)
print(f"Loss : {loss:.4f}")
print(f"Dice : {dice:.4f}")
print(f"IoU  : {iou:.4f}")