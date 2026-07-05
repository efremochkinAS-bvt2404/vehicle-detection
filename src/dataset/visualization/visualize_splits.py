import json

from src.utils.paths import DATASET_SPLIT_PLOTS_DIR, SPLIT_CONFIG_PATH


SPLIT_NAMES = ["train", "val", "test"]
SPLIT_LABELS = {
    "train": "Train",
    "val": "Validation",
    "test": "Test",
}
SPLIT_COLORS = {
    "train": "#4e79a7",
    "val": "#f28e2b",
    "test": "#59a14f",
}


def _load_split_config():
    if not SPLIT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Split config not found: {SPLIT_CONFIG_PATH}")

    with SPLIT_CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_bar_chart(path, labels, values, title, ylabel, color="#4e79a7"):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(13, 8))
    bars = axis.bar(labels, values, color=color)

    axis.set_title(title, fontsize=24, fontweight="bold", pad=28)
    axis.set_ylabel(ylabel, fontsize=18, labelpad=16)
    axis.tick_params(axis="x", labelsize=17)
    axis.tick_params(axis="y", labelsize=15)
    axis.grid(axis="y", linestyle="--", alpha=0.35)

    total = sum(values)
    max_value = max(values) if values else 0
    axis.set_ylim(0, max_value * 1.18 if max_value else 1)

    for bar, value in zip(bars, values):
        percent = value / total * 100 if total else 0
        value_label = f"{value:.2f}" if isinstance(value, float) else str(value)
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value_label}\n({percent:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=14,
        )

    figure.tight_layout(pad=2.5)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def _save_grouped_class_chart(path, config, normalized=False):
    import matplotlib.pyplot as plt
    import numpy as np

    classes = list(config["class_mapping"].keys())
    x_values = np.arange(len(classes))
    width = 0.25

    figure, axis = plt.subplots(figsize=(16, 9))

    for index, split_name in enumerate(SPLIT_NAMES):
        split = config["splits"][split_name]
        class_counts = [split["classes"].get(class_name, 0) for class_name in classes]

        if normalized:
            total = sum(class_counts)
            values = [count / total * 100 if total else 0 for count in class_counts]
        else:
            values = class_counts

        offset = (index - 1) * width
        axis.bar(
            x_values + offset,
            values,
            width,
            label=SPLIT_LABELS[split_name],
            color=SPLIT_COLORS[split_name],
        )

    if normalized:
        title = "Class Distribution by Split, %"
        ylabel = "Objects, %"
    else:
        title = "Class Distribution by Split"
        ylabel = "Objects"

    axis.set_title(title, fontsize=24, fontweight="bold", pad=28)
    axis.set_ylabel(ylabel, fontsize=18, labelpad=16)
    axis.set_xticks(x_values)
    axis.set_xticklabels(classes, fontsize=16)
    axis.tick_params(axis="y", labelsize=15)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend(fontsize=15)

    figure.tight_layout(pad=2.5)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def _save_split_ratio_pie(path, config):
    import matplotlib.pyplot as plt

    labels = [SPLIT_LABELS[split_name] for split_name in SPLIT_NAMES]
    values = [config["splits"][split_name]["images"] for split_name in SPLIT_NAMES]
    colors = [SPLIT_COLORS[split_name] for split_name in SPLIT_NAMES]

    figure, axis = plt.subplots(figsize=(10, 10))
    axis.pie(
        values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 16},
    )
    axis.set_title(
        "Dataset Split Ratio by Images",
        fontsize=24,
        fontweight="bold",
        pad=28,
    )
    axis.axis("equal")

    figure.tight_layout(pad=2.5)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def _save_summary_chart(path, config):
    import matplotlib.pyplot as plt
    import numpy as np

    split_labels = [SPLIT_LABELS[split_name] for split_name in SPLIT_NAMES]
    image_counts = [config["splits"][split_name]["images"] for split_name in SPLIT_NAMES]
    object_counts = [config["splits"][split_name]["objects"] for split_name in SPLIT_NAMES]
    avg_objects = [
        objects / images if images else 0
        for images, objects in zip(image_counts, object_counts)
    ]
    classes = list(config["class_mapping"].keys())
    total_by_class = [
        sum(
            config["splits"][split_name]["classes"].get(class_name, 0)
            for split_name in SPLIT_NAMES
        )
        for class_name in classes
    ]

    figure, axes = plt.subplots(2, 2, figsize=(19, 12))
    figure.suptitle("Dataset Split Summary", fontsize=26, fontweight="bold", y=0.98)

    axes[0, 0].bar(split_labels, image_counts, color="#4e79a7")
    axes[0, 0].set_title("Images by Split", fontsize=20, pad=18)
    axes[0, 0].set_ylabel("Images", fontsize=16, labelpad=12)

    axes[0, 1].bar(split_labels, object_counts, color="#f28e2b")
    axes[0, 1].set_title("Objects by Split", fontsize=20, pad=18)
    axes[0, 1].set_ylabel("Objects", fontsize=16, labelpad=12)

    axes[1, 0].bar(split_labels, avg_objects, color="#59a14f")
    axes[1, 0].set_title("Average Objects per Image", fontsize=20, pad=18)
    axes[1, 0].set_ylabel("Objects per image", fontsize=16, labelpad=12)

    axes[1, 1].bar(classes, total_by_class, color="#af7aa1")
    axes[1, 1].set_title("Total Objects by Class", fontsize=20, pad=18)
    axes[1, 1].set_ylabel("Objects", fontsize=16, labelpad=12)

    for axis in axes.flat:
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.tick_params(axis="x", labelsize=14)
        axis.tick_params(axis="y", labelsize=13)

    for axis in axes[1, :]:
        axis.tick_params(axis="x", rotation=20)

    figure.tight_layout(pad=2.8)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def visualize_split_statistics():
    config = _load_split_config()
    DATASET_SPLIT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    split_labels = [SPLIT_LABELS[split_name] for split_name in SPLIT_NAMES]
    image_counts = [config["splits"][split_name]["images"] for split_name in SPLIT_NAMES]
    object_counts = [config["splits"][split_name]["objects"] for split_name in SPLIT_NAMES]
    avg_objects = [
        objects / images if images else 0
        for images, objects in zip(image_counts, object_counts)
    ]

    created_plots = []

    image_counts_path = DATASET_SPLIT_PLOTS_DIR / "split_image_counts.png"
    _save_bar_chart(
        image_counts_path,
        split_labels,
        image_counts,
        "Images by Dataset Split",
        "Images",
        color="#4e79a7",
    )
    created_plots.append(image_counts_path)

    object_counts_path = DATASET_SPLIT_PLOTS_DIR / "split_object_counts.png"
    _save_bar_chart(
        object_counts_path,
        split_labels,
        object_counts,
        "Objects by Dataset Split",
        "Objects",
        color="#f28e2b",
    )
    created_plots.append(object_counts_path)

    objects_per_image_path = DATASET_SPLIT_PLOTS_DIR / "objects_per_image_by_split.png"
    _save_bar_chart(
        objects_per_image_path,
        split_labels,
        avg_objects,
        "Average Objects per Image by Split",
        "Objects per image",
        color="#59a14f",
    )
    created_plots.append(objects_per_image_path)

    class_distribution_path = DATASET_SPLIT_PLOTS_DIR / "split_class_distribution.png"
    _save_grouped_class_chart(class_distribution_path, config, normalized=False)
    created_plots.append(class_distribution_path)

    normalized_distribution_path = (
        DATASET_SPLIT_PLOTS_DIR / "split_class_distribution_normalized.png"
    )
    _save_grouped_class_chart(normalized_distribution_path, config, normalized=True)
    created_plots.append(normalized_distribution_path)

    split_ratio_path = DATASET_SPLIT_PLOTS_DIR / "split_ratio_pie.png"
    _save_split_ratio_pie(split_ratio_path, config)
    created_plots.append(split_ratio_path)

    summary_path = DATASET_SPLIT_PLOTS_DIR / "dataset_split_summary.png"
    _save_summary_chart(summary_path, config)
    created_plots.append(summary_path)

    summary = {
        "output_dir": str(DATASET_SPLIT_PLOTS_DIR),
        "plots": [str(path) for path in created_plots],
    }

    return summary


def print_summary(summary):
    print("Dataset split visualizations created")
    print(f"  Output directory: {summary['output_dir']}")
    print("  Plots:")
    for plot_path in summary["plots"]:
        print(f"    - {plot_path}")


def main():
    summary = visualize_split_statistics()
    print_summary(summary)


if __name__ == "__main__":
    main()
