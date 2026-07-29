import time
import torch
from tqdm import tqdm

from utils.metrics import dice_score
from utils.checkpoint import save_checkpoint


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        history,
        model_dir,
        report_dir,
        patience,
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device

        self.history = history

        self.model_dir = model_dir
        self.report_dir = report_dir

        self.patience = patience

        self.best_dice = 0.0
        self.no_improvement = 0

    # ==========================================================
    # Training
    # ==========================================================

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0

        progress = tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
        )

        for images, masks in progress:

            images = images.to(self.device)

            masks = masks.unsqueeze(1).to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, masks)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        return running_loss / len(self.train_loader)

    # ==========================================================
    # Validation
    # ==========================================================

    def validate(self):

        self.model.eval()

        running_loss = 0.0
        running_dice = 0.0

        with torch.no_grad():

            progress = tqdm(
                self.val_loader,
                desc="Validation",
                leave=False,
            )

            for images, masks in progress:

                images = images.to(self.device)

                masks = masks.unsqueeze(1).to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, masks)

                running_loss += loss.item()

                running_dice += dice_score(
                    outputs,
                    masks,
                ).item()

                progress.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

        return (
            running_loss / len(self.val_loader),
            running_dice / len(self.val_loader),
        )

            # ==========================================================
    # Fit
    # ==========================================================

    def fit(self, epochs):

        for epoch in range(epochs):

            start_time = time.time()

            train_loss = self.train_one_epoch()

            val_loss, val_dice = self.validate()

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            self.history.update(
                epoch + 1,
                train_loss,
                val_loss,
                val_dice,
            )

            elapsed = (time.time() - start_time) / 60

            print("\n" + "=" * 60)
            print(f"Epoch {epoch+1}/{epochs}")
            print("=" * 60)
            print(f"Train Loss : {train_loss:.4f}")
            print(f"Val Loss   : {val_loss:.4f}")
            print(f"Dice Score : {val_dice:.4f}")
            print(f"Best Dice  : {self.best_dice:.4f}")
            print(f"Time       : {elapsed:.2f} minutes")

            # Save latest checkpoint
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                self.best_dice,
                self.model_dir / "last_model.pth",
            )

            # Save best model
            if val_dice > self.best_dice:

                self.best_dice = val_dice
                self.no_improvement = 0

                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.best_dice,
                    self.model_dir / "best_model.pth",
                )

                print("✓ Best model updated!")

            else:

                self.no_improvement += 1

                print(
                    f"No improvement ({self.no_improvement}/{self.patience})"
                )

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:

                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.best_dice,
                    self.model_dir / f"checkpoint_epoch_{epoch+1}.pth",
                )

            # Early stopping
            if self.no_improvement >= self.patience:

                print("\nEarly stopping triggered.")
                break

        self.history.save(
            self.report_dir / "history.csv"
        )

        print("\nTraining completed successfully!")
        print(f"Best Dice Score: {self.best_dice:.4f}")