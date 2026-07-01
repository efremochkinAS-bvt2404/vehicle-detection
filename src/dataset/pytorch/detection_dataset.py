from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.dataset.loaders.manifest import load_manifest


class DetectionDataset(Dataset):
    def __init__(self, manifest_file, images_dir, transforms=None):
        self.manifest_file = Path(manifest_file)
        self.images_dir = Path(images_dir)
        self.transforms = transforms

        self.manifest = load_manifest(self.manifest_file)
        self.images = self.manifest.images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        annotation = self.images[index]

        image_path = self.images_dir / annotation.file_name

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for obj in annotation.objects:
            xmin, ymin, xmax, ymax = obj.bbox.to_list()

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj.class_id)
            areas.append((xmax - xmin) * (ymax - ymin))
            iscrowd.append(0)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        image_id = torch.tensor(
            [int(Path(annotation.file_name).stem)],
            dtype=torch.int64,
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target