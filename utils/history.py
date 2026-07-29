import pandas as pd


class TrainingHistory:

    def __init__(self):

        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "dice": [],
        }

    def update(
        self,
        epoch,
        train_loss,
        val_loss,
        dice,
    ):

        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["dice"].append(dice)

    def save(self, filename):

        df = pd.DataFrame(self.history)

        df.to_csv(filename, index=False)