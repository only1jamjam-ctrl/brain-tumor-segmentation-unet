import segmentation_models_pytorch as smp


def get_model():
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation=None,
    )

    return model