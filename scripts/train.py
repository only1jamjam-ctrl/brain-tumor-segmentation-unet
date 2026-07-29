import sys
from pathlib import Path
import random
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.config import *
from utils.dataset import BrainTumorDataset
from utils.transforms import train_transform, val_transform
from utils.losses import BCEDiceLoss
from utils.history import TrainingHistory
from utils.trainer import Trainer

from models.unet import get_model


# ======================================================
# Reproducibility
# ======================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ======================================================
# Datasets
# ======================================================

train_dataset = BrainTumorDataset(
    TRAIN_IMAGES,
    TRAIN_MASKS,
    transform=train_transform,
)

val_dataset = BrainTumorDataset(
    VAL_IMAGES,
    VAL_MASKS,
    transform=val_transform,
)

# ======================================================
# DataLoaders
# ======================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

# ======================================================
# Model
# ======================================================

model = get_model().to(DEVICE)

criterion = BCEDiceLoss()

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
)

history = TrainingHistory()

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=DEVICE,
    history=history,
    model_dir=MODEL_DIR,
    report_dir=REPORT_DIR,
    patience=PATIENCE,
)

trainer.fit(EPOCHS)