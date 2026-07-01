from collections import Counter
import json

from PIL import Image

from configs.loader import KEEP_CLASSES
from src.utils.paths import (
    FILTERED_IMAGES_DIR,
    FILTERED_LABELS_DIR,
    FILTERED_KITTI_METRICS_DIR,
)


EXPECTED_FIELDS_COUNT = 15


def validate_filtered_kitti():
    if not FILTERED_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Filtered images directory not found: {FILTERED_IMAGES_DIR}")

    if not FILTERED_LABELS_DIR.exists():
        raise FileNotFoundError(f"Filtered labels directory not found: {FILTERED_LABELS_DIR}")

    image_files = sorted(FILTERED_IMAGES_DIR.glob("*.png"))
    label_files = sorted(FILTERED_LABELS_DIR.glob("*.txt"))

    if not image_files:
        raise FileNotFoundError(f"No filtered images found: {FILTERED_IMAGES_DIR}")

    if not label_files:
        raise FileNotFoundError(f"No filtered labels found: {FILTERED_LABELS_DIR}")

    errors = []
    warnings = []
    class_counter = Counter()

    image_stems = {file.stem for file in image_files}
    label_stems = {file.stem for file in label_files}

    images_without_labels = sorted(image_stems - label_stems)
    labels_without_images = sorted(label_stems - image_stems)

    for stem in images_without_labels:
        errors.append(f"Missing label file for image: {stem}.png")

    for stem in labels_without_images:
        errors.append(f"Missing image file for label: {stem}.txt")

    for label_file in label_files:
        image_file = FILTERED_IMAGES_DIR / f"{label_file.stem}.png"

        if not image_file.exists():
            continue

        try:
            with Image.open(image_file) as image:
                width, height = image.size
        except Exception as error:
            errors.append(f"Failed to open image {image_file.name}: {error}")
            continue

        with label_file.open("r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]

        if not lines:
            errors.append(f"Empty label file: {label_file.name}")
            continue

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()

            if len(parts) != EXPECTED_FIELDS_COUNT:
                errors.append(
                    f"{label_file.name}, line {line_number}: "
                    f"expected {EXPECTED_FIELDS_COUNT} fields, got {len(parts)}"
                )
                continue

            class_name = parts[0]

            if class_name not in KEEP_CLASSES:
                errors.append(
                    f"{label_file.name}, line {line_number}: "
                    f"unexpected class: {class_name}"
                )
                continue

            try:
                xmin = float(parts[4])
                ymin = float(parts[5])
                xmax = float(parts[6])
                ymax = float(parts[7])
            except ValueError:
                errors.append(
                    f"{label_file.name}, line {line_number}: "
                    f"bbox coordinates are not numeric"
                )
                continue

            if xmin >= xmax:
                errors.append(
                    f"{label_file.name}, line {line_number}: "
                    f"xmin >= xmax ({xmin} >= {xmax})"
                )

            if ymin >= ymax:
                errors.append(
                    f"{label_file.name}, line {line_number}: "
                    f"ymin >= ymax ({ymin} >= {ymax})"
                )

            if xmin < 0 or ymin < 0:
                errors.append(
                    f"{label_file.name}, line {line_number}: "
                    f"bbox has negative coordinates [{xmin}, {ymin}, {xmax}, {ymax}]"
                )

            if xmax > width or ymax > height:
                warnings.append(
                    f"{label_file.name}, line {line_number}: "
                    f"bbox is outside image bounds "
                    f"[{xmin}, {ymin}, {xmax}, {ymax}], image size: {width}x{height}"
                )

            class_counter[class_name] += 1

    missing_target_classes = [
        class_name
        for class_name in sorted(KEEP_CLASSES)
        if class_counter[class_name] == 0
    ]

    for class_name in missing_target_classes:
        warnings.append(f"Target class has no objects after filtering: {class_name}")

    report = {
        "summary": {
            "images": len(image_files),
            "labels": len(label_files),
            "classes": {
                class_name: class_counter[class_name]
                for class_name in sorted(KEEP_CLASSES)
            },
            "errors_count": len(errors),
            "warnings_count": len(warnings),
        },
        "errors": errors,
        "warnings": warnings,
    }

    report_path = save_validation_report(report)

    return report, report_path


def save_validation_report(report):
    FILTERED_KITTI_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = FILTERED_KITTI_METRICS_DIR / "validation.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=4)

    return report_path


def print_summary(report, report_path):
    summary = report["summary"]

    print("Filtered KITTI validation completed")
    print()
    print("Summary:")
    print(f"  Images: {summary['images']}")
    print(f"  Labels: {summary['labels']}")
    print(f"  Errors: {summary['errors_count']}")
    print(f"  Warnings: {summary['warnings_count']}")
    print()
    print("Generated files:")
    print(f"  Validation report: {report_path}")


def main():
    report, report_path = validate_filtered_kitti()
    print_summary(report, report_path)


if __name__ == "__main__":
    main()