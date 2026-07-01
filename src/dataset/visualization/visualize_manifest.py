import json
import random
import shutil
from pathlib import Path

from configs.loader import create_rng
from src.utils.paths import (
    FILTERED_IMAGES_DIR,
    TRAIN_MANIFEST_PATH,
    MANIFEST_VISUALIZATION_DIR,
)
from src.utils.visualization import save_manifest_annotation


SAMPLES_COUNT = 20


def prepare_output_dir():
    if MANIFEST_VISUALIZATION_DIR.exists():
        shutil.rmtree(MANIFEST_VISUALIZATION_DIR)

    MANIFEST_VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)


def load_manifest():
    if not TRAIN_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest file not found: {TRAIN_MANIFEST_PATH}")

    with TRAIN_MANIFEST_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def select_samples(images):
    if not images:
        raise RuntimeError("No images found in manifest")

    rng = create_rng()
    samples_count = min(SAMPLES_COUNT, len(images))

    return rng.sample(images, samples_count)


def visualize_manifest():
    prepare_output_dir()

    manifest = load_manifest()
    images = manifest["images"]
    selected_images = select_samples(images)

    created_images = 0
    missing_images = 0
    skipped_without_objects = 0

    for item in selected_images:
        image_file = FILTERED_IMAGES_DIR / item["file_name"]

        if not image_file.exists():
            missing_images += 1
            continue

        objects = item.get("objects", [])

        if not objects:
            skipped_without_objects += 1
            continue

        output_file = MANIFEST_VISUALIZATION_DIR / f"{Path(item['file_name']).stem}_annotated.png"

        save_manifest_annotation(
            image_path=image_file,
            objects=objects,
            output_path=output_file,
            line_width=2,
            font_size=18,
            use_ru_labels=True,
        )

        created_images += 1

    summary = {
        "selected_samples": len(selected_images),
        "created_images": created_images,
        "missing_images": missing_images,
        "skipped_without_objects": skipped_without_objects,
        "output_dir": str(MANIFEST_VISUALIZATION_DIR),
    }

    return summary


def print_summary(summary):
    print("Manifest visualization completed")
    print()
    print("Summary:")
    print(f"  Selected samples: {summary['selected_samples']}")
    print(f"  Created images: {summary['created_images']}")
    print(f"  Missing images: {summary['missing_images']}")
    print(f"  Skipped without objects: {summary['skipped_without_objects']}")
    print()
    print("Generated directory:")
    print(f"  Visualization: {summary['output_dir']}")


def main():
    summary = visualize_manifest()
    print_summary(summary)


if __name__ == "__main__":
    main()