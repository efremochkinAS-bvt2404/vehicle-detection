from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def to_list(self):
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionObject:
    class_id: int
    class_name: str
    bbox: BBox


@dataclass
class ImageAnnotation:
    file_name: str
    width: int
    height: int
    objects: list[DetectionObject]


@dataclass
class DetectionManifest:
    dataset: str
    version: str
    split: str
    images_count: int
    objects_count: int
    class_mapping: dict[str, int]
    classes: list[str]
    images: list[ImageAnnotation]


def load_manifest(manifest_path: str | Path) -> DetectionManifest:
    manifest_path = Path(manifest_path)

    with manifest_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    images = []

    for item in data["images"]:
        objects = []

        for obj in item["objects"]:
            bbox_data = obj["bbox"]

            objects.append(
                DetectionObject(
                    class_id=obj["class_id"],
                    class_name=obj["class_name"],
                    bbox=BBox(
                        xmin=bbox_data["xmin"],
                        ymin=bbox_data["ymin"],
                        xmax=bbox_data["xmax"],
                        ymax=bbox_data["ymax"],
                    ),
                )
            )

        images.append(
            ImageAnnotation(
                file_name=item["file_name"],
                width=item["width"],
                height=item["height"],
                objects=objects,
            )
        )

    return DetectionManifest(
        dataset=data["dataset"],
        version=data["version"],
        split=data["split"],
        images_count=data["images_count"],
        objects_count=data["objects_count"],
        class_mapping=data["class_mapping"],
        classes=data["classes"],
        images=images,
    )