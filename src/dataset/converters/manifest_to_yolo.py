from pathlib import Path
import shutil
import yaml

from src.dataset.loaders.manifest import load_manifest
from src.utils.paths import (
    FILTERED_IMAGES_DIR,
    YOLO_DATA_DIR,
    YOLO_IMAGES_DIR,
    YOLO_LABELS_DIR,
    YOLO_DATA_YAML_PATH,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
    TEST_MANIFEST_PATH,
)


SPLITS = ["train", "val", "test"]

MANIFEST_PATHS = {
    "train": TRAIN_MANIFEST_PATH,
    "val": VAL_MANIFEST_PATH,
    "test": TEST_MANIFEST_PATH,
}


def prepare_dirs():
    if YOLO_DATA_DIR.exists():
        shutil.rmtree(YOLO_DATA_DIR)

    for split in SPLITS:
        (YOLO_IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (YOLO_LABELS_DIR / split).mkdir(parents=True, exist_ok=True)


def convert_bbox_to_yolo(bbox, image_width, image_height):
    x_center = ((bbox.xmin + bbox.xmax) / 2) / image_width
    y_center = ((bbox.ymin + bbox.ymax) / 2) / image_height
    width = (bbox.xmax - bbox.xmin) / image_width
    height = (bbox.ymax - bbox.ymin) / image_height

    return x_center, y_center, width, height


def convert_split(split, verbose=False):
    manifest_path = MANIFEST_PATHS[split]

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    manifest = load_manifest(manifest_path)

    converted_images = 0
    converted_labels = 0
    converted_objects = 0
    missing_images = 0

    for image_annotation in manifest.images:
        image_name = image_annotation.file_name

        source_image = FILTERED_IMAGES_DIR / image_name
        target_image = YOLO_IMAGES_DIR / split / image_name

        if not source_image.exists():
            print(f"  Missing image: {source_image}")
            missing_images += 1
            continue

        shutil.copy(source_image, target_image)

        label_file = YOLO_LABELS_DIR / split / f"{Path(image_name).stem}.txt"

        lines = []

        for obj in image_annotation.objects:
            x_center, y_center, width, height = convert_bbox_to_yolo(
                bbox=obj.bbox,
                image_width=image_annotation.width,
                image_height=image_annotation.height,
            )

            lines.append(
                f"{obj.class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}\n"
            )

            converted_objects += 1

        with label_file.open("w", encoding="utf-8") as file:
            file.writelines(lines)

        converted_images += 1
        converted_labels += 1

    split_summary = {
        "split": split,
        "images": converted_images,
        "labels": converted_labels,
        "objects": converted_objects,
        "missing_images": missing_images,
    }

    if verbose:
        print_split_summary(split_summary)

    return split_summary


def create_data_yaml(class_names):
    data_yaml = {
        "path": str(YOLO_DATA_DIR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }

    YOLO_DATA_YAML_PATH.parent.mkdir(parents=True, exist_ok=True)

    with YOLO_DATA_YAML_PATH.open("w", encoding="utf-8") as file:
        yaml.dump(data_yaml, file, allow_unicode=True, sort_keys=False)


def print_split_summary(summary):
    print(f"{summary['split']}:")
    print(f"  Converted images: {summary['images']}")
    print(f"  Label files: {summary['labels']}")
    print(f"  Objects: {summary['objects']}")
    print(f"  Missing images: {summary['missing_images']}")
    print()


def print_final_summary(summaries):
    total_images = sum(item["images"] for item in summaries)
    total_labels = sum(item["labels"] for item in summaries)
    total_objects = sum(item["objects"] for item in summaries)
    total_missing_images = sum(item["missing_images"] for item in summaries)

    print("Manifest to YOLO conversion completed")
    print()
    print("Summary:")
    print(f"  Total images: {total_images}")
    print(f"  Total label files: {total_labels}")
    print(f"  Total objects: {total_objects}")
    print(f"  Missing images: {total_missing_images}")
    print()
    print("Generated files:")
    print(f"  YOLO dataset directory: {YOLO_DATA_DIR}")
    print(f"  YOLO config: {YOLO_DATA_YAML_PATH}")


def convert_manifest_to_yolo(verbose=False):
    prepare_dirs()

    summaries = []

    first_manifest = load_manifest(TRAIN_MANIFEST_PATH)
    class_names = first_manifest.classes

    for split in SPLITS:
        summary = convert_split(split, verbose)
        summaries.append(summary)

    create_data_yaml(class_names)

    if verbose:
        print_final_summary(summaries)

    return summaries


def main():
    convert_manifest_to_yolo(verbose=True)


if __name__ == "__main__":
    main()