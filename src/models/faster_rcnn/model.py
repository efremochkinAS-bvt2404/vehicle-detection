import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn(num_classes, pretrained=True):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None

    try:
        if weights is None:
            model = fasterrcnn_resnet50_fpn_v2(
                weights=None,
                weights_backbone=None,
            )
        else:
            model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    except Exception as error:
        if not pretrained:
            raise

        print(f"Could not load pretrained Faster R-CNN weights: {error}")
        print("Falling back to randomly initialized Faster R-CNN.")
        model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

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
    model = build_faster_rcnn(num_classes=num_classes, pretrained=pretrained)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint
