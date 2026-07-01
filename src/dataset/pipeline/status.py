from datetime import datetime
import json

from src.utils.paths import (
    FILTERED_IMAGES_DIR,
    FILTERED_LABELS_DIR,
    MANIFESTS_DIR,
    METADATA_DIR,
    PREPARATION_STATUS_PATH,
    SPLIT_CONFIG_PATH,
    TEST_MANIFEST_PATH,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
    YOLO_DATA_YAML_PATH,
    YOLO_IMAGES_DIR,
    YOLO_LABELS_DIR,
)


MANIFEST_PATHS = {
    "train": TRAIN_MANIFEST_PATH,
    "val": VAL_MANIFEST_PATH,
    "test": TEST_MANIFEST_PATH,
}


def _count_files(directory, pattern):
    if not directory.exists():
        return 0

    return sum(1 for _ in directory.glob(pattern))


def _read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _validate_manifest(split_name, manifest_path, errors, summary):
    if not manifest_path.exists():
        errors.append(f"Missing {split_name} manifest: {manifest_path}")
        return

    try:
        manifest = _read_json(manifest_path)
    except Exception as error:
        errors.append(f"Cannot read {split_name} manifest: {error}")
        return

    images = manifest.get("images", [])
    images_count = manifest.get("images_count", len(images))

    if not images:
        errors.append(f"{split_name} manifest has no images")

    if images_count != len(images):
        errors.append(
            f"{split_name} manifest images_count mismatch: "
            f"{images_count} != {len(images)}"
        )

    summary["manifests"][split_name] = {
        "images": len(images),
        "objects": manifest.get("objects_count", 0),
    }


def _validate_yolo_split(split_name, errors, summary):
    images_count = _count_files(YOLO_IMAGES_DIR / split_name, "*.png")
    labels_count = _count_files(YOLO_LABELS_DIR / split_name, "*.txt")

    if images_count == 0:
        errors.append(f"YOLO {split_name} images are missing")

    if labels_count == 0:
        errors.append(f"YOLO {split_name} labels are missing")

    if images_count != labels_count:
        errors.append(
            f"YOLO {split_name} image/label count mismatch: "
            f"{images_count} != {labels_count}"
        )

    summary["yolo"][split_name] = {
        "images": images_count,
        "labels": labels_count,
    }


def get_dataset_status():
    errors = []
    warnings = []
    summary = {
        "filtered": {
            "images": _count_files(FILTERED_IMAGES_DIR, "*.png"),
            "labels": _count_files(FILTERED_LABELS_DIR, "*.txt"),
        },
        "manifests": {},
        "yolo": {},
    }

    if summary["filtered"]["images"] == 0:
        errors.append(f"Filtered images are missing: {FILTERED_IMAGES_DIR}")

    if summary["filtered"]["labels"] == 0:
        errors.append(f"Filtered labels are missing: {FILTERED_LABELS_DIR}")

    if summary["filtered"]["images"] != summary["filtered"]["labels"]:
        errors.append(
            "Filtered image/label count mismatch: "
            f"{summary['filtered']['images']} != {summary['filtered']['labels']}"
        )

    if not SPLIT_CONFIG_PATH.exists():
        errors.append(f"Missing split config: {SPLIT_CONFIG_PATH}")

    for split_name, manifest_path in MANIFEST_PATHS.items():
        _validate_manifest(split_name, manifest_path, errors, summary)

    if not YOLO_DATA_YAML_PATH.exists():
        errors.append(f"Missing YOLO data config: {YOLO_DATA_YAML_PATH}")

    for split_name in MANIFEST_PATHS:
        _validate_yolo_split(split_name, errors, summary)

    if not PREPARATION_STATUS_PATH.exists():
        warnings.append(f"Missing preparation marker: {PREPARATION_STATUS_PATH}")

    return {
        "prepared": not errors,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }


def is_dataset_prepared():
    return get_dataset_status()["prepared"]


def save_preparation_status(status=None):
    status = status or get_dataset_status()
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "prepared": status["prepared"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
    }

    with PREPARATION_STATUS_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=4)

    return PREPARATION_STATUS_PATH


def print_dataset_status(status):
    print("Dataset status:")
    print(f"  Prepared: {status['prepared']}")
    print(f"  Filtered images: {status['summary']['filtered']['images']}")
    print(f"  Filtered labels: {status['summary']['filtered']['labels']}")

    if status["errors"]:
        print("  Errors:")
        for error in status["errors"]:
            print(f"    - {error}")

    if status["warnings"]:
        print("  Warnings:")
        for warning in status["warnings"]:
            print(f"    - {warning}")


def ensure_dataset_prepared(force=False, verbose=True):
    status = get_dataset_status()

    if status["prepared"] and not force:
        if verbose:
            print("Prepared dataset found. Skipping dataset preparation.")

        if status["warnings"]:
            save_preparation_status(status)

        return status

    if verbose:
        if force:
            print("Forced dataset preparation requested.")
        else:
            print("Prepared dataset is missing or invalid.")
            print_dataset_status(status)
        print()

    from src.dataset.pipeline.prepare_dataset import prepare_dataset

    prepare_dataset(force=True)
    return get_dataset_status()
