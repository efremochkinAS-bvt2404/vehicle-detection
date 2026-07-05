from collections import Counter
import json
import shutil

import matplotlib.pyplot as plt

from configs.loader import KEEP_CLASSES
from src.utils.paths import (
    KITTI_IMAGES_DIR,
    KITTI_LABELS_DIR,
    FILTERED_DIR,
    FILTERED_IMAGES_DIR,
    FILTERED_LABELS_DIR,
    FILTERED_KITTI_METRICS_DIR,
    FILTERED_KITTI_PLOTS_DIR,
)


def prepare_dirs():
    if FILTERED_DIR.exists():
        shutil.rmtree(FILTERED_DIR)

    FILTERED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    FILTERED_LABELS_DIR.mkdir(parents=True, exist_ok=True)


def is_valid_kitti_line(parts):
    return len(parts) >= 15


def filter_kitti():
    if not KITTI_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images directory not found: {KITTI_IMAGES_DIR}")

    if not KITTI_LABELS_DIR.exists():
        raise FileNotFoundError(f"Labels directory not found: {KITTI_LABELS_DIR}")

    label_files = sorted(KITTI_LABELS_DIR.glob("*.txt"))

    if not label_files:
        raise FileNotFoundError(f"No label files found: {KITTI_LABELS_DIR}")

    prepare_dirs()

    class_counter = Counter()
    total_raw_objects = 0
    total_filtered_objects = 0
    copied_images = 0
    skipped_without_target_classes = 0
    missing_images = 0
    invalid_lines = 0

    for label_file in label_files:
        filtered_lines = []

        with label_file.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if not is_valid_kitti_line(parts):
                    invalid_lines += 1
                    continue

                total_raw_objects += 1

                class_name = parts[0]

                if class_name in KEEP_CLASSES:
                    filtered_lines.append(line + "\n")
                    class_counter[class_name] += 1
                    total_filtered_objects += 1

        if not filtered_lines:
            skipped_without_target_classes += 1
            continue

        image_file = KITTI_IMAGES_DIR / f"{label_file.stem}.png"

        if not image_file.exists():
            missing_images += 1
            continue

        shutil.copy(image_file, FILTERED_IMAGES_DIR / image_file.name)

        filtered_label_path = FILTERED_LABELS_DIR / label_file.name
        with filtered_label_path.open("w", encoding="utf-8") as file:
            file.writelines(filtered_lines)

        copied_images += 1

    stats = {
        "source_label_files": len(label_files),
        "copied_images": copied_images,
        "skipped_without_target_classes": skipped_without_target_classes,
        "missing_images": missing_images,
        "invalid_lines": invalid_lines,
        "total_raw_objects": total_raw_objects,
        "total_filtered_objects": total_filtered_objects,
        "removed_objects": total_raw_objects - total_filtered_objects,
        "target_classes": sorted(KEEP_CLASSES),
        "classes": {
            class_name: class_counter[class_name]
            for class_name in sorted(KEEP_CLASSES)
        },
    }

    metrics_path, plot_path = save_filter_results(stats)

    return stats, metrics_path, plot_path


def save_filter_results(stats):
    FILTERED_KITTI_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FILTERED_KITTI_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = FILTERED_KITTI_METRICS_DIR / "summary.json"
    plot_path = FILTERED_KITTI_PLOTS_DIR / "class_distribution.png"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=4)

    class_names = list(stats["classes"].keys())
    class_counts = list(stats["classes"].values())

    figure, axis = plt.subplots(figsize=(14, 8))
    bars = axis.bar(class_names, class_counts, color="#59a14f")

    axis.set_title(
        "Filtered KITTI Class Distribution",
        fontsize=24,
        fontweight="bold",
        pad=28,
    )
    axis.set_xlabel("Class", fontsize=18, labelpad=16)
    axis.set_ylabel("Objects", fontsize=18, labelpad=16)
    axis.tick_params(axis="x", labelrotation=25, labelsize=16)
    axis.tick_params(axis="y", labelsize=15)
    axis.grid(axis="y", linestyle="--", alpha=0.4)

    max_count = max(class_counts) if class_counts else 0
    axis.set_ylim(0, max_count * 1.16 if max_count else 1)
    for bar, value in zip(bars, class_counts):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=14,
        )

    figure.tight_layout(pad=2.5)
    figure.savefig(plot_path, dpi=300)
    plt.close(figure)

    return metrics_path, plot_path


def print_summary(stats, metrics_path, plot_path):
    print("KITTI filtering completed")
    print()

    print("Summary:")
    print(f"  Source label files: {stats['source_label_files']}")
    print(f"  Copied images: {stats['copied_images']}")
    print(f"  Skipped files without target classes: {stats['skipped_without_target_classes']}")
    print(f"  Missing images: {stats['missing_images']}")
    print(f"  Invalid annotation lines: {stats['invalid_lines']}")
    print(f"  Objects before filtering: {stats['total_raw_objects']}")
    print(f"  Objects after filtering: {stats['total_filtered_objects']}")
    print()

    print("Generated files:")
    print(f"  Metrics: {metrics_path}")
    print(f"  Plot: {plot_path}")


def main():
    stats, metrics_path, plot_path = filter_kitti()
    print_summary(stats, metrics_path, plot_path)


if __name__ == "__main__":
    main()
