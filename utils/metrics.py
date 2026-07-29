import torch


def dice_score(preds, targets, smooth=1.0):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()

    return (
        2.0 * intersection + smooth
    ) / (
        preds.sum() + targets.sum() + smooth
    )


def iou_score(preds, targets, smooth=1.0):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()

    union = preds.sum() + targets.sum() - intersection

    return (
        intersection + smooth
    ) / (
        union + smooth
    )