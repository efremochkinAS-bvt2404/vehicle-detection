import csv
from pathlib import Path


PLOT_SPECS = {
    "loss.png": {
        "title": "Loss",
        "series": {
            "train loss": ["train/box_loss", "train/cls_loss", "train/dfl_loss", "train_loss"],
            "validation loss": ["val/box_loss", "val/cls_loss", "val/dfl_loss", "val_loss"],
        },
    },
    "precision.png": {
        "title": "Precision",
        "series": {
            "precision": ["metrics/precision(B)", "precision"],
        },
    },
    "recall.png": {
        "title": "Recall",
        "series": {
            "recall": ["metrics/recall(B)", "recall"],
        },
    },
    "map50.png": {
        "title": "mAP@50",
        "series": {
            "mAP@50": ["metrics/mAP50(B)", "map50", "mAP50"],
        },
    },
    "map50_95.png": {
        "title": "mAP@50:95",
        "series": {
            "mAP@50:95": ["metrics/mAP50-95(B)", "map50_95", "mAP50-95"],
        },
    },
    "lr.png": {
        "title": "Learning rate",
        "series": {
            "lr": ["lr/pg0", "lr/pg1", "lr/pg2", "learning_rate", "lr"],
        },
    },
}


REQUIRED_PLOTS = {
    "loss.png",
    "precision.png",
    "recall.png",
    "map50.png",
    "map50_95.png",
    "lr.png",
    "results.png",
}


def _normalize_key(key):
    return key.strip()


def _read_history(history_path):
    history_path = Path(history_path)

    with history_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [
            {_normalize_key(key): value for key, value in row.items()}
            for row in reader
        ]


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sum_columns(rows, columns):
    values = []

    for row in rows:
        row_values = [
            _to_float(row.get(column))
            for column in columns
            if column in row
        ]
        row_values = [value for value in row_values if value is not None]

        if row_values:
            values.append(sum(row_values))
        else:
            values.append(None)

    return values


def _first_existing_column(rows, candidates):
    if not rows:
        return None

    columns = set(rows[0].keys())

    for candidate in candidates:
        if candidate in columns:
            return candidate

    return None


def _compact_series(values):
    x_values = []
    y_values = []

    for index, value in enumerate(values, start=1):
        if value is None:
            continue

        x_values.append(index)
        y_values.append(value)

    return x_values, y_values


def plot_training_history(history_path, output_dir):
    import matplotlib.pyplot as plt

    history_path = Path(history_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not history_path.exists():
        return []

    rows = _read_history(history_path)

    if not rows:
        return []

    created_plots = []

    for file_name, spec in PLOT_SPECS.items():
        plt.figure(figsize=(10, 6))
        plotted = False

        for label, columns in spec["series"].items():
            if len(columns) > 1 and label in {"train loss", "validation loss"}:
                values = _sum_columns(rows, columns)
            else:
                column = _first_existing_column(rows, columns)
                values = [
                    _to_float(row.get(column))
                    for row in rows
                ] if column else []

            x_values, y_values = _compact_series(values)

            if not y_values:
                continue

            plt.plot(x_values, y_values, marker="o", label=label)
            plotted = True

        if not plotted:
            plt.close()
            continue

        plt.title(spec["title"])
        plt.xlabel("Epoch")
        plt.ylabel(spec["title"])
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        output_path = output_dir / file_name
        plt.savefig(output_path, dpi=300)
        plt.close()
        created_plots.append(output_path)

    results_path = plot_results_summary(rows, output_dir)

    if results_path is not None:
        created_plots.append(results_path)

    return created_plots


def _series_from_spec(rows, spec):
    result = {}

    for label, columns in spec["series"].items():
        if len(columns) > 1 and label in {"train loss", "validation loss"}:
            values = _sum_columns(rows, columns)
        else:
            column = _first_existing_column(rows, columns)
            values = [
                _to_float(row.get(column))
                for row in rows
            ] if column else []

        x_values, y_values = _compact_series(values)

        if y_values:
            result[label] = (x_values, y_values)

    return result


def plot_results_summary(rows, output_dir):
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_path = output_dir / "results.png"

    summary_files = [
        "loss.png",
        "precision.png",
        "recall.png",
        "map50.png",
        "map50_95.png",
        "lr.png",
    ]

    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    plotted_any = False

    for axis, file_name in zip(axes, summary_files):
        spec = PLOT_SPECS[file_name]
        series = _series_from_spec(rows, spec)

        if not series:
            axis.set_visible(False)
            continue

        for label, (x_values, y_values) in series.items():
            axis.plot(x_values, y_values, marker="o", label=label)

        axis.set_title(spec["title"])
        axis.set_xlabel("Epoch")
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend()
        plotted_any = True

    if not plotted_any:
        plt.close(figure)
        return None

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path
