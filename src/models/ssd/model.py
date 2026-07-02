import torch
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


def build_ssd(num_classes, pretrained=True):
    if pretrained:
        try:
            return ssdlite320_mobilenet_v3_large(
                weights=None,
                weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
                num_classes=num_classes,
            )
        except Exception as error:
            print(f"Could not load pretrained SSD backbone weights: {error}")
            print("Falling back to randomly initialized SSD.")

    return ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
    )


def save_checkpoint(path, model, optimizer, epoch, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_model_from_checkpoint(checkpoint_path, num_classes, device, pretrained=False):
    model = build_ssd(num_classes=num_classes, pretrained=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint
