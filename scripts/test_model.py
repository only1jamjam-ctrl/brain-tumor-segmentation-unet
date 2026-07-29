import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from models.unet import get_model

model = get_model()

x = torch.randn(1, 1, 512, 512)

with torch.no_grad():
    y = model(x)

print("=" * 50)
print("MODEL TEST")
print("=" * 50)
print(f"Input shape  : {x.shape}")
print(f"Output shape : {y.shape}")
print(f"Parameters   : {sum(p.numel() for p in model.parameters()):,}")