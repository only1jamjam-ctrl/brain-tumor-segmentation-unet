import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

REPORT_DIR = Path("reports")

history = pd.read_csv(REPORT_DIR / "history.csv")

# -----------------------------
# Loss Curve
# -----------------------------
plt.figure(figsize=(8,5))

plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(REPORT_DIR / "loss_curve.png")

plt.close()

# -----------------------------
# Dice Curve
# -----------------------------
plt.figure(figsize=(8,5))

plt.plot(history["epoch"], history["dice"])

plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Validation Dice Score")

plt.grid(True)

plt.tight_layout()

plt.savefig(REPORT_DIR / "dice_curve.png")

plt.close()

print("Training curves saved successfully!")