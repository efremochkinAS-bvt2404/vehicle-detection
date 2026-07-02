from pathlib import Path

import torch
import torch.nn.functional as torch_f
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as F

from src.dataset.loaders.manifest import load_manifest


class DetrDetectionDataset(Dataset):
    def __init__(
        self,
        manifest_file,
        images_dir,
        image_processor,
        augmentation_config=None,
        train=False,
    ):
        self.manifest_file = Path(manifest_file)
        self.images_dir = Path(images_dir)
        self.image_processor = image_processor
        self.augmentation_config = augmentation_config or {}
        self.train = train
        self.manifest = load_manifest(self.manifest_file)
        self.images = self.manifest.images
        self.color_jitter = ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02,
        )

    def __len__(self):
        return len(self.images)

    def _augment(self, image, boxes):
        if not self.train:
            return image, boxes

        if self.augmentation_config.get("horizontal_flip", False):
            if torch.rand(1).item() < 0.5:
                width = image.width
                image = F.hflip(image)
                flipped = boxes.clone()
                xmin = flipped[:, 0].clone()
                xmax = flipped[:, 2].clone()
                flipped[:, 0] = width - xmax
                flipped[:, 2] = width - xmin
                boxes = flipped

        if self.augmentation_config.get("color_jitter", False):
            image = self.color_jitter(image)

        return image, boxes

    def __getitem__(self, index):
        annotation = self.images[index]
        image_path = self.images_dir / annotation.file_name

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

        boxes = []
        class_labels = []
        areas = []

        for obj in annotation.objects:
            xmin, ymin, xmax, ymax = obj.bbox.to_list()
            boxes.append([xmin, ymin, xmax, ymax])
            class_labels.append(obj.class_id)
            areas.append((xmax - xmin) * (ymax - ymin))

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        image, boxes = self._augment(image, boxes)

        coco_annotations = []

        for box, class_label, area in zip(boxes.tolist(), class_labels, areas):
            xmin, ymin, xmax, ymax = box
            width = max(0.0, xmax - xmin)
            height = max(0.0, ymax - ymin)

            if width <= 0 or height <= 0:
                continue

            coco_annotations.append(
                {
                    "image_id": index,
                    "category_id": class_label,
                    "bbox": [xmin, ymin, width, height],
                    "area": area,
                    "iscrowd": 0,
                }
            )

        encoding = self.image_processor(
            images=image,
            annotations={
                "image_id": index,
                "annotations": coco_annotations,
            },
            return_tensors="pt",
        )

        labels = encoding["labels"][0]
        metric_target = {
            "boxes": boxes,
            "labels": torch.as_tensor(
                [label + 1 for label in class_labels],
                dtype=torch.int64,
            ),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "pixel_mask": encoding["pixel_mask"].squeeze(0),
            "labels": labels,
            "metric_target": metric_target,
            "file_name": annotation.file_name,
        }


def detr_collate_fn(batch):
    max_height = max(item["pixel_values"].shape[1] for item in batch)
    max_width = max(item["pixel_values"].shape[2] for item in batch)

    padded_pixel_values = []
    padded_pixel_masks = []

    for item in batch:
        pixel_values = item["pixel_values"]
        pixel_mask = item["pixel_mask"]
        height = pixel_values.shape[1]
        width = pixel_values.shape[2]
        padding = (0, max_width - width, 0, max_height - height)

        padded_pixel_values.append(torch_f.pad(pixel_values, padding, value=0))
        padded_pixel_masks.append(torch_f.pad(pixel_mask, padding, value=0))

    pixel_values = torch.stack(padded_pixel_values)
    pixel_mask = torch.stack(padded_pixel_masks)

    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "labels": [item["labels"] for item in batch],
        "metric_targets": [item["metric_target"] for item in batch],
        "file_names": [item["file_name"] for item in batch],
    }
