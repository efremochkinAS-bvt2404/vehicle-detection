import random

import torch
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as F


class DetectionCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)

        return image, target


class DetectionRandomHorizontalFlip:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, image, target):
        if random.random() >= self.probability:
            return image, target

        width = image.width
        image = F.hflip(image)

        boxes = target["boxes"].clone()
        xmin = boxes[:, 0].clone()
        xmax = boxes[:, 2].clone()
        boxes[:, 0] = width - xmax
        boxes[:, 2] = width - xmin
        target["boxes"] = boxes

        return image, target


class DetectionColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02):
        self.transform = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, image, target):
        return self.transform(image), target


class DetectionToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class DetectionSanitizeBoxes:
    def __call__(self, image, target):
        boxes = target["boxes"]

        if boxes.numel() == 0:
            return image, target

        _, height, width = image.shape
        boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height)

        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

        for key in ("boxes", "labels", "area", "iscrowd"):
            target[key] = target[key][keep]

        target["area"] = (
            (target["boxes"][:, 2] - target["boxes"][:, 0])
            * (target["boxes"][:, 3] - target["boxes"][:, 1])
        ).to(torch.float32)

        return image, target


def get_detection_transforms(augmentation_config=None, train=False):
    augmentation_config = augmentation_config or {}
    transforms = []

    if train and augmentation_config.get("horizontal_flip", False):
        transforms.append(DetectionRandomHorizontalFlip(probability=0.5))

    if train and augmentation_config.get("color_jitter", False):
        transforms.append(DetectionColorJitter())

    transforms.extend([
        DetectionToTensor(),
        DetectionSanitizeBoxes(),
    ])

    return DetectionCompose(transforms)
