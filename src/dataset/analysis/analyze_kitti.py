from collections import Counter
import json

import matplotlib.pyplot as plt

from src.utils.paths import KITTI_LABELS_DIR, KITTI_METRICS_DIR, KITTI_PLOTS_DIR


def analyze_kitti():
    if not KITTI_LABELS_DIR.exists():
        raise FileNotFoundError(f"Labels directory not found: {KITTI_LABELS_DIR}")

    label_files = sorted(KITTI_LABELS_DIR.glob("*.txt"))

    if not label_files:
        raise FileNotFoundError(f"No label files found: {KITTI_LABELS_DIR}")

    class_counter = Counter()
    images_with_objects = 0
    empty_label_files = 0
    total_objects = 0

    for label_file in label_files:
        has_objects = False

        with label_file.open("r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()

                if not parts:
                    continue

                class_name = parts[0]
                class_counter[class_name] += 1
                total_objects += 1
                has_objects = True

        if has_objects:
            images_with_objects += 1
        else:
            empty_label_files += 1

    stats = {
        "label_files": len(label_files),
        "images_with_objects": images_with_objects,
        "empty_label_files": empty_label_files,
        "total_objects": total_objects,
        "classes_count": len(class_counter),
        "classes": dict(class_counter.most_common()),
    }

    metrics_path, plot_path = save_analysis(stats)

    return stats, metrics_path, plot_path


def save_analysis(stats):
    KITTI_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    KITTI_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = KITTI_METRICS_DIR / "analysis.json"
    plot_path = KITTI_PLOTS_DIR / "class_distribution.png"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=4)

    class_names = list(stats["classes"].keys())
    class_counts = list(stats["classes"].values())

    figure, axis = plt.subplots(figsize=(16, 9))
    bars = axis.bar(class_names, class_counts, color="#4e79a7")

    axis.set_title(
        "Raw KITTI Class Distribution",
        fontsize=24,
        fontweight="bold",
        pad=28,
    )
    axis.set_xlabel("Class", fontsize=18, labelpad=16)
    axis.set_ylabel("Objects", fontsize=18, labelpad=16)
    axis.tick_params(axis="x", labelrotation=45, labelsize=15)
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
            fontsize=13,
        )

    figure.tight_layout(pad=2.5)
    figure.savefig(plot_path, dpi=300)
    plt.close(figure)

    return metrics_path, plot_path


def print_summary(stats, metrics_path, plot_path):
    print("KITTI dataset analysis completed")
    print()
    print("Summary:")
    print(f"  Label files: {stats['label_files']}")
    print(f"  Images with objects: {stats['images_with_objects']}")
    print(f"  Empty label files: {stats['empty_label_files']}")
    print(f"  Total objects: {stats['total_objects']}")
    print(f"  Classes: {stats['classes_count']}")
    print()
    print("Saved files:")
    print(f"  Metrics: {metrics_path}")
    print(f"  Plot: {plot_path}")


def main():
    stats, metrics_path, plot_path = analyze_kitti()
    print_summary(stats, metrics_path, plot_path)


if __name__ == "__main__":
    main()
