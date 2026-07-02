from functools import partial

import torch
from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


def build_retinanet(num_classes, pretrained=True):
    weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None

    try:
        if weights is None:
            model = retinanet_resnet50_fpn_v2(
                weights=None,
                weights_backbone=None,
                num_classes=num_classes,
            )
        else:
            model = retinanet_resnet50_fpn_v2(weights=weights)
    except Exception as error:
        if not pretrained:
            raise

        print(f"Could not load pretrained RetinaNet weights: {error}")
        print("Falling back to randomly initialized RetinaNet.")
        model = retinanet_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
        )

    head = model.head.classification_head
    in_channels = head.conv[0][0].in_channels
    num_anchors = head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32),
    )

    return model


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
    model = build_retinanet(num_classes=num_classes, pretrained=pretrained)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint
