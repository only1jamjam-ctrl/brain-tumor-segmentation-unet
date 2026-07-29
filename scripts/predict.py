import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.config import *
from utils.dataset import BrainTumorDataset
from utils.transforms import val_transform
from models.unet import get_model


# --------------------------------------------------
# Dataset
# --------------------------------------------------

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


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = get_model().to(DEVICE)

checkpoint = torch.load(
    MODEL_DIR / "best_model.pth",
    map_location=DEVICE,
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


# --------------------------------------------------
# Output Folder
# --------------------------------------------------

OUTPUT_DIR = Path("outputs/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Predict
# --------------------------------------------------

with torch.no_grad():

    for i, (images, masks) in enumerate(test_loader):

        if i >= 10:
            break

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)
        outputs = torch.sigmoid(outputs)
        outputs = (outputs > 0.5).float()

        image = images.squeeze().cpu().numpy()
        mask = masks.squeeze().cpu().numpy()
        prediction = outputs.squeeze().cpu().numpy()

        plt.figure(figsize=(20, 5))

        plt.suptitle(
            "Brain Tumor Segmentation Results",
            fontsize=18,
            fontweight="bold",
        )

        # -----------------------------
        # Original MRI
        # -----------------------------

        plt.subplot(1, 4, 1)

        plt.imshow(image, cmap="gray")

        plt.title(
            "Original MRI",
            fontsize=14,
            fontweight="bold",
        )

        plt.axis("off")

        # -----------------------------
        # Ground Truth
        # -----------------------------

        plt.subplot(1, 4, 2)

        plt.imshow(mask, cmap="gray")

        plt.title(
            "Ground Truth Mask",
            fontsize=14,
            fontweight="bold",
        )

        plt.axis("off")

        # -----------------------------
        # Prediction
        # -----------------------------

        plt.subplot(1, 4, 3)

        plt.imshow(prediction, cmap="gray")

        plt.title(
            "Predicted Mask",
            fontsize=14,
            fontweight="bold",
        )

        plt.axis("off")

        # -----------------------------
        # Overlay
        # -----------------------------

        plt.subplot(1, 4, 4)

        # MRI
        plt.imshow(image, cmap="gray")

        # Transparent tumour
        plt.imshow(
            prediction,
            cmap="Reds",
            alpha=0.75,
            vmin=0,
            vmax=1,
        )

        # Tumour outline
        plt.contour(
            prediction,
            levels=[0.5],
            colors="red",
            linewidths=2,
        )

        plt.title(
            "Tumour Overlay",
            fontsize=14,
            fontweight="bold",
        )

        plt.axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])

        plt.savefig(
            OUTPUT_DIR / f"prediction_{i+1}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

print(f"\nSaved prediction images to: {OUTPUT_DIR}")