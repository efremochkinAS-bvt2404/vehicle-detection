from collections import Counter
from datetime import datetime
import json
import random

from PIL import Image

from configs.loader import KEEP_CLASSES, SPLIT_RATIOS, RANDOM_SEED
from src.utils.paths import (
    FILTERED_IMAGES_DIR,
    FILTERED_LABELS_DIR,
    MANIFESTS_DIR,
    SPLITS_DIR,
    METADATA_DIR,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
    TEST_MANIFEST_PATH,
    TRAIN_SPLIT_PATH,
    VAL_SPLIT_PATH,
    TEST_SPLIT_PATH,
    SPLIT_CONFIG_PATH,
)


DATASET_NAME = "KITTI"
DATASET_VERSION = "filtered_transport"


CLASS_TO_ID = {
    class_name: index
    for index, class_name in enumerate(sorted(KEEP_CLASSES))
}


def parse_kitti_label(label_file):
    objects = []

    with label_file.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) < 15:
                continue

            class_name = parts[0]

            if class_name not in CLASS_TO_ID:
                continue

            xmin = float(parts[4])
            ymin = float(parts[5])
            xmax = float(parts[6])
            ymax = float(parts[7])

            objects.append({
                "class_id": CLASS_TO_ID[class_name],
                "class_name": class_name,
                "bbox": {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                }
            })

    return objects


def create_manifest_item(image_file, label_file):
    with Image.open(image_file) as image:
        width, height = image.size

    return {
        "file_name": image_file.name,
        "width": width,
        "height": height,
        "objects": parse_kitti_label(label_file)
    }


def build_items():
    if not FILTERED_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Filtered images directory not found: {FILTERED_IMAGES_DIR}")

    if not FILTERED_LABELS_DIR.exists():
        raise FileNotFoundError(f"Filtered labels directory not found: {FILTERED_LABELS_DIR}")

    label_files = sorted(FILTERED_LABELS_DIR.glob("*.txt"))

    if not label_files:
        raise FileNotFoundError(f"No filtered labels found: {FILTERED_LABELS_DIR}")

    items = []

    for label_file in label_files:
        image_file = FILTERED_IMAGES_DIR / f"{label_file.stem}.png"

        if not image_file.exists():
            continue

        item = create_manifest_item(image_file, label_file)

        if item["objects"]:
            items.append(item)

    if not items:
        raise RuntimeError("No valid manifest items were created")

    return items


def split_items(items):
    train_ratio = SPLIT_RATIOS["train"]
    val_ratio = SPLIT_RATIOS["val"]
    test_ratio = SPLIT_RATIOS["test"]

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    rng = random.Random(RANDOM_SEED)

    items = items.copy()
    rng.shuffle(items)

    total = len(items)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return (
        items[:train_end],
        items[train_end:val_end],
        items[val_end:]
    )


def count_objects(items):
    counter = Counter()

    for item in items:
        for obj in item["objects"]:
            counter[obj["class_name"]] += 1

    return counter


def build_split_stats(items):
    counter = count_objects(items)

    return {
        "images": len(items),
        "objects": sum(counter.values()),
        "classes": {
            class_name: counter[class_name]
            for class_name in CLASS_TO_ID
        }
    }


def build_manifest(split_name, items):
    stats = build_split_stats(items)

    return {
        "dataset": DATASET_NAME,
        "version": DATASET_VERSION,
        "split": split_name,
        "images_count": stats["images"],
        "objects_count": stats["objects"],
        "class_mapping": CLASS_TO_ID,
        "classes": list(CLASS_TO_ID.keys()),
        "images": items
    }


def save_json(data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def save_split_list(items, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(item["file_name"] + "\n")


def save_split_config(train_items, val_items, test_items):
    split_config = {
        "dataset": DATASET_NAME,
        "version": DATASET_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "random_seed": RANDOM_SEED,
        "ratios": dict(SPLIT_RATIOS),
        "class_mapping": CLASS_TO_ID,
        "splits": {
            "train": build_split_stats(train_items),
            "val": build_split_stats(val_items),
            "test": build_split_stats(test_items)
        }
    }

    save_json(split_config, SPLIT_CONFIG_PATH)


def create_manifests():
    items = build_items()
    train_items, val_items, test_items = split_items(items)

    save_json(build_manifest("train", train_items), TRAIN_MANIFEST_PATH)
    save_json(build_manifest("val", val_items), VAL_MANIFEST_PATH)
    save_json(build_manifest("test", test_items), TEST_MANIFEST_PATH)

    save_split_list(train_items, TRAIN_SPLIT_PATH)
    save_split_list(val_items, VAL_SPLIT_PATH)
    save_split_list(test_items, TEST_SPLIT_PATH)

    save_split_config(train_items, val_items, test_items)

    summary = {
        "total": build_split_stats(items),
        "train": build_split_stats(train_items),
        "val": build_split_stats(val_items),
        "test": build_split_stats(test_items),
    }

    return summary


def print_summary(summary):
    print("Manifest creation completed")
    print()
    print("Summary:")
    print(f"  Total images: {summary['total']['images']}")
    print(f"  Total objects: {summary['total']['objects']}")
    print(f"  Train images: {summary['train']['images']}")
    print(f"  Val images: {summary['val']['images']}")
    print(f"  Test images: {summary['test']['images']}")
    print()
    print("Generated directories:")
    print(f"  Manifests: {MANIFESTS_DIR}")
    print(f"  Splits: {SPLITS_DIR}")
    print(f"  Metadata: {METADATA_DIR}")


def main():
    summary = create_manifests()
    print_summary(summary)


if __name__ == "__main__":
    main()