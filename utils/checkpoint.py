import torch


def save_checkpoint(model,
                    optimizer,
                    epoch,
                    best_dice,
                    filename):

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_dice": best_dice,
    }, filename)


def load_checkpoint(filename,
                    model,
                    optimizer):

    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return (
        checkpoint["epoch"],
        checkpoint["best_dice"],
    )