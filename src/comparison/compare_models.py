import csv
import json
import math
from pathlib import Path
import textwrap
import time

import torch

from src.utils.paths import (
    EXPERIMENTS_DIR,
    FILTERED_IMAGES_DIR,
    RESULTS_DIR,
    TEST_MANIFEST_PATH,
    TRAIN_MANIFEST_PATH,
)


MODEL_ORDER = ["yolo", "faster_rcnn", "detr", "retinanet", "ssd"]
COMPARISON_DIR = RESULTS_DIR / "comparison"
MODEL_RUNS_COMPARISON_DIR = RESULTS_DIR / "model_runs_comparison"
PERFORMANCE_CACHE_PATH = COMPARISON_DIR / "performance_cache.json"

METRIC_COLUMNS = [
    "model",
    "mAP",
    "mAP@50",
    "mAP@50:95",
    "Precision",
    "Recall",
    "F1-score",
    "FPS",
    "Inference time per image (s)",
    "Checkpoint size (MB)",
]

HYPERPARAMETER_COLUMNS = [
    "model",
    "architecture",
    "framework",
    "pretrained",
    "epochs",
    "batch_size",
    "image_size",
    "workers",
    "learning_rate",
    "weight_decay",
    "optimizer",
]

TRAINING_CONFIGURATION_FILE = "training_configuration"

OPTIMIZER_BY_MODEL = {
    "yolo": "Ultralytics auto",
    "faster_rcnn": "AdamW",
    "detr": "AdamW",
    "retinanet": "AdamW",
    "ssd": "AdamW",
}


def _read_json(path):
    path = Path(path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_csv(path):
    path = Path(path)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows([
            {column: row.get(column, "") for column in columns}
            for row in rows
        ])

    return path


def _to_float(value):
    try:
        if value == "":
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(result) or math.isinf(result):
        return None

    return result


def _format_value(value, decimals=None):
    if value is None:
        return ""

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return ""
        if decimals is not None:
            return f"{value:.{decimals}f}"
        return f"{value:.6g}"

    return str(value)


def _format_table_value(value, width=18):
    text = _format_value(value)

    if len(text) <= width:
        return text

    return "\n".join(textwrap.wrap(text, width=width, break_long_words=True))


def _format_column_value(column, value):
    if column == "Checkpoint size (MB)":
        return _format_value(value, decimals=2)

    return _format_value(value)


def _find_best_experiment(model_name):
    model_dir = EXPERIMENTS_DIR / model_name
    if not model_dir.exists():
        return None

    candidates = []
    for path in model_dir.iterdir():
        if not path.is_dir() or not (path / "metrics.json").exists():
            continue

        metrics = _read_json(path / "metrics.json")
        if metrics.get("status") != "evaluated":
            continue

        candidates.append((path, _to_float(_get_nested(metrics, "metrics", "map50"))))

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: (
            item[1] if item[1] is not None else -1,
            item[0].stat().st_mtime,
        ),
    )[0]


def _find_evaluated_experiments(model_name):
    model_dir = EXPERIMENTS_DIR / model_name
    if not model_dir.exists():
        return []

    candidates = []
    for path in model_dir.iterdir():
        if not path.is_dir() or not (path / "metrics.json").exists():
            continue

        metrics = _read_json(path / "metrics.json")
        if metrics.get("status") != "evaluated":
            continue

        candidates.append(path)

    return sorted(candidates, key=lambda path: path.name)


def _next_numbered_dir(parent_dir, prefix="comparison"):
    parent_dir.mkdir(parents=True, exist_ok=True)
    index = 1

    while True:
        candidate = parent_dir / f"{prefix}_{index:03d}"
        if not candidate.exists():
            return candidate
        index += 1


def _get_nested(data, *keys):
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)

    return current


def _checkpoint_size_mb(run_dir, metrics):
    saved_size = _get_nested(metrics, "performance", "checkpoint_size_mb")
    if saved_size is not None:
        return saved_size

    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if checkpoint_path.exists():
        return checkpoint_path.stat().st_size / (1024 * 1024)

    return None


def _infer_yolo_auto_optimizer(info, training_config):
    try:
        from src.dataset.loaders.manifest import load_manifest

        train_size = len(load_manifest(TRAIN_MANIFEST_PATH).images)
    except Exception:
        train_size = 0

    epochs = _to_float(info.get("epochs", training_config.get("epochs"))) or 0
    batch_size = _to_float(info.get("batch_size", training_config.get("batch_size"))) or 0
    nominal_batch_size = _to_float(training_config.get("nbs")) or 64

    if train_size <= 0 or epochs <= 0 or batch_size <= 0:
        return "AdamW"

    iterations = math.ceil(train_size / max(batch_size, nominal_batch_size)) * epochs
    return "MuSGD" if iterations > 10000 else "AdamW"


def _best_checkpoint_path(model_name, run_dir):
    if model_name == "yolo":
        checkpoint_path = run_dir / "checkpoints" / "best.pt"
        if checkpoint_path.exists():
            return checkpoint_path
        return run_dir / "weights" / "best.pt"

    return run_dir / "checkpoints" / "best.pt"


def _cache_key(model_name, run_dir):
    checkpoint_path = _best_checkpoint_path(model_name, run_dir)
    modified_at = checkpoint_path.stat().st_mtime if checkpoint_path.exists() else 0
    return f"{model_name}:{run_dir.name}:{modified_at:.0f}"


def _load_performance_cache():
    return _read_json(PERFORMANCE_CACHE_PATH)


def _save_performance_cache(cache):
    PERFORMANCE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PERFORMANCE_CACHE_PATH.open("w", encoding="utf-8") as file:
        json.dump(cache, file, indent=4, ensure_ascii=False)


def _resolve_device(device_value):
    if torch.cuda.is_available() and str(device_value) != "cpu":
        return torch.device(f"cuda:{device_value}")
    return torch.device("cpu")


def _benchmark_yolo(run_dir, info):
    from ultralytics import YOLO
    from src.dataset.loaders.manifest import load_manifest

    checkpoint_path = _best_checkpoint_path("yolo", run_dir)
    if not checkpoint_path.exists():
        return {}

    manifest = load_manifest(TEST_MANIFEST_PATH)
    image_paths = [
        FILTERED_IMAGES_DIR / annotation.file_name
        for annotation in manifest.images
    ]
    image_paths = [path for path in image_paths if path.exists()]

    if not image_paths:
        return {}

    model = YOLO(str(checkpoint_path))

    # Warm-up.
    model.predict(source=str(image_paths[0]), verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    started_at = time.perf_counter()
    for image_path in image_paths:
        model.predict(source=str(image_path), verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time = time.perf_counter() - started_at
    inference_time = total_time / len(image_paths)

    return {
        "fps": 1 / inference_time if inference_time > 0 else 0.0,
        "inference_time_per_image_seconds": inference_time,
        "benchmark_images": len(image_paths),
    }


def _benchmark_faster_rcnn(run_dir, info):
    from configs.loader import load_model_config
    from src.dataset.loaders.manifest import load_manifest
    from src.dataset.pytorch.dataloaders import create_detection_dataloaders
    from src.models.faster_rcnn.model import load_model_from_checkpoint

    checkpoint_path = _best_checkpoint_path("faster_rcnn", run_dir)
    if not checkpoint_path.exists():
        return {}

    config = load_model_config("faster_rcnn")
    training_config = config["training"]
    device = _resolve_device(training_config.get("device", 0))
    manifest = load_manifest(TEST_MANIFEST_PATH)
    num_classes = len(manifest.classes) + 1

    model, _ = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        device=device,
        pretrained=False,
    )
    model.eval()

    _, _, test_loader = create_detection_dataloaders(
        batch_size=training_config["batch_size"],
        num_workers=training_config["workers"],
        shuffle_train=False,
        augmentation_config=config.get("augmentation", {}),
        label_offset=1,
    )

    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = [image.to(device) for image in images]

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            started_at = time.perf_counter()
            model(images)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            total_time += time.perf_counter() - started_at
            total_images += len(images)

    inference_time = total_time / total_images if total_images > 0 else 0.0
    return {
        "fps": 1 / inference_time if inference_time > 0 else 0.0,
        "inference_time_per_image_seconds": inference_time,
        "benchmark_images": total_images,
    }


def _benchmark_missing_performance(model_name, run_dir, info, cache):
    key = _cache_key(model_name, run_dir)
    if key in cache:
        return cache[key]

    if model_name == "yolo":
        result = _benchmark_yolo(run_dir, info)
    elif model_name == "faster_rcnn":
        result = _benchmark_faster_rcnn(run_dir, info)
    else:
        result = {}

    if result:
        cache[key] = result
        _save_performance_cache(cache)

    return result


def _history_column(rows, candidates, sum_columns=False):
    if not rows:
        return []

    if sum_columns:
        values = []
        for row in rows:
            parts = [
                _to_float(row.get(column))
                for column in candidates
                if column in row
            ]
            parts = [value for value in parts if value is not None]
            values.append(sum(parts) if parts else None)
        return values

    columns = set(rows[0].keys())
    selected = next((column for column in candidates if column in columns), None)
    if selected is None:
        return []

    return [_to_float(row.get(selected)) for row in rows]


def _compact_series(values):
    x_values = []
    y_values = []

    for index, value in enumerate(values, start=1):
        if value is None:
            continue
        x_values.append(index)
        y_values.append(value)

    return x_values, y_values


def _collect_run(model_name, run_dir, performance_cache=None):
    metrics = _read_json(run_dir / "metrics.json")
    info = _read_json(run_dir / "info.json")
    history = _read_csv(run_dir / "history.csv")
    config = info.get("config") or _read_json(run_dir / "config.json")

    metric_values = metrics.get("metrics", {})
    performance = metrics.get("performance", {})

    if not performance.get("fps") or not performance.get("inference_time_per_image_seconds"):
        benchmark_performance = _benchmark_missing_performance(
            model_name,
            run_dir,
            info,
            performance_cache if performance_cache is not None else {},
        )
        performance = {**benchmark_performance, **performance}

    model_config = config.get("model", {}) if isinstance(config, dict) else {}
    training_config = config.get("training", {}) if isinstance(config, dict) else {}
    architecture = model_config.get("architecture")

    if architecture is None and model_name == "yolo":
        architecture = model_config.get("display_name") or model_config.get("name")

        if model_config.get("pretrained_checkpoint"):
            architecture = f"{architecture} ({model_config['pretrained_checkpoint']})"

    pretrained = model_config.get("pretrained")

    if pretrained is None and model_name == "yolo":
        pretrained = bool(
            model_config.get("pretrained_checkpoint")
            or info.get("pretrained_checkpoint")
        )

    learning_rate = info.get("learning_rate", training_config.get("learning_rate"))
    weight_decay = info.get("weight_decay", training_config.get("weight_decay"))
    optimizer = (
        info.get("optimizer")
        or training_config.get("optimizer")
        or OPTIMIZER_BY_MODEL.get(model_name, "")
    )

    if model_name == "yolo" and optimizer == "auto":
        optimizer = _infer_yolo_auto_optimizer(info, training_config)
        learning_rate = learning_rate if learning_rate is not None else "auto"
        weight_decay = weight_decay if weight_decay is not None else "auto"

    metric_row = {
        "model": model_name,
        "run_name": metrics.get("run_name") or info.get("run_name") or run_dir.name,
        "mAP": metric_values.get("map"),
        "mAP@50": metric_values.get("map50"),
        "mAP@50:95": metric_values.get("map50_95"),
        "Precision": metric_values.get("precision"),
        "Recall": metric_values.get("recall"),
        "F1-score": metric_values.get("f1"),
        "FPS": performance.get("fps"),
        "Inference time per image (s)": performance.get("inference_time_per_image_seconds"),
        "Checkpoint size (MB)": _checkpoint_size_mb(run_dir, metrics),
    }

    hyperparameter_row = {
        "model": model_name,
        "run_name": metric_row["run_name"],
        "architecture": architecture,
        "framework": model_config.get("framework"),
        "pretrained": pretrained,
        "epochs": info.get("epochs", training_config.get("epochs")),
        "batch_size": info.get("batch_size", training_config.get("batch_size")),
        "image_size": training_config.get("image_size") or info.get("image_size"),
        "workers": info.get("workers", training_config.get("workers")),
        "device": info.get("device", training_config.get("device")),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": optimizer,
    }

    return {
        "model": model_name,
        "run_dir": run_dir,
        "metrics": metrics,
        "info": info,
        "history": history,
        "metric_row": metric_row,
        "hyperparameter_row": hyperparameter_row,
    }


def _save_table_png(path, rows, columns, title):
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure_width = max(20.4, len(columns) * 2.05)
    figure_height = 6.85 if len(columns) >= 9 else 6.2
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    axis.axis("off")
    axis.text(
        0.5,
        0.91,
        title,
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        transform=axis.transAxes,
    )

    table_data = [
        [_format_table_value(row.get(column)) for column in columns]
        for row in rows
    ]
    table = axis.table(
        cellText=table_data,
        colLabels=columns,
        bbox=[0.035, 0.13, 0.93, 0.62],
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.1, 2.25)

    for column_index in range(len(columns)):
        table.auto_set_column_width(column_index)

    for (row_index, _), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_text_props(weight="bold", color="white", fontsize=12)
            cell.set_facecolor("#2f5597")
        else:
            cell.set_text_props(fontsize=12)
            cell.set_facecolor("#f7f9fc" if row_index % 2 else "#ffffff")

    figure.tight_layout()
    figure.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(figure)
    return path


def _bar_chart(path, rows, column, title, ylabel):
    import matplotlib.pyplot as plt

    values = [_to_float(row.get(column)) for row in rows]
    labels = [row["model"] for row in rows]
    plot_values = [value if value is not None else 0 for value in values]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, plot_values, color="#4e79a7")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, values):
        if value is None:
            label = "n/a"
            height = 0
        else:
            label = f"{value:.3f}" if value < 10 else f"{value:.1f}"
            height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def _grouped_quality_chart(path, rows):
    import matplotlib.pyplot as plt

    metrics = ["mAP@50", "Precision", "Recall", "F1-score"]
    labels = [row["model"] for row in rows]
    width = 0.18
    x_positions = list(range(len(labels)))

    plt.figure(figsize=(11, 6))
    for metric_index, metric in enumerate(metrics):
        offsets = [
            position + (metric_index - 1.5) * width
            for position in x_positions
        ]
        values = [
            _to_float(row.get(metric)) or 0
            for row in rows
        ]
        plt.bar(offsets, values, width=width, label=metric)

    plt.xticks(x_positions, labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Quality Metrics by Model")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def _line_chart(path, runs, key, title, ylabel, fallback_key=None):
    import matplotlib.pyplot as plt

    series_specs = {
        "train_loss": (["train_loss", "train/box_loss", "train/cls_loss", "train/dfl_loss"], True),
        "val_loss": (["val_loss", "val/box_loss", "val/cls_loss", "val/dfl_loss"], True),
        "map50": (["map50", "mAP50", "metrics/mAP50(B)"], False),
        "map50_95": (["map50_95", "mAP50-95", "metrics/mAP50-95(B)"], False),
        "precision": (["precision", "metrics/precision(B)"], False),
        "recall": (["recall", "metrics/recall(B)"], False),
    }

    plt.figure(figsize=(11, 6))
    plotted = False

    for run in runs:
        columns, sum_columns = series_specs[key]
        values = _history_column(run["history"], columns, sum_columns=sum_columns)

        if not any(value is not None for value in values) and fallback_key:
            columns, sum_columns = series_specs[fallback_key]
            values = _history_column(run["history"], columns, sum_columns=sum_columns)

        x_values, y_values = _compact_series(values)
        if not y_values:
            continue

        plt.plot(x_values, y_values, marker="o", label=run["model"])
        plotted = True

    if not plotted:
        plt.close()
        return None

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def _radar_chart(path, rows):
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["mAP@50", "Precision", "Recall", "F1-score", "normalized FPS"]
    max_fps = max([
        _to_float(row.get("FPS")) or 0
        for row in rows
    ] or [0])
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    figure, axis = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for row in rows:
        fps = _to_float(row.get("FPS")) or 0
        values = [
            _to_float(row.get("mAP@50")) or 0,
            _to_float(row.get("Precision")) or 0,
            _to_float(row.get("Recall")) or 0,
            _to_float(row.get("F1-score")) or 0,
            fps / max_fps if max_fps > 0 else 0,
        ]
        values += values[:1]
        axis.plot(angles, values, linewidth=2, label=row["model"])
        axis.fill(angles, values, alpha=0.08)

    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1)
    axis.set_title("Quality and Speed Radar", pad=20)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12))
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def _best_row(rows, column, smallest=False):
    valid = [
        row
        for row in rows
        if _to_float(row.get(column)) is not None
    ]
    if not valid:
        return None
    return min(valid, key=lambda row: _to_float(row[column])) if smallest else max(
        valid,
        key=lambda row: _to_float(row[column]),
    )


def _write_summary(path, runs, metric_rows, table_paths, graph_paths):
    best_map50 = _best_row(metric_rows, "mAP@50")
    best_f1 = _best_row(metric_rows, "F1-score")
    fastest = _best_row(metric_rows, "FPS")
    smallest = _best_row(metric_rows, "Checkpoint size (MB)", smallest=True)

    summary = {
        "models_compared": [run["model"] for run in runs],
        "runs_used": {
            run["model"]: run["metric_row"]["run_name"]
            for run in runs
        },
        "best_model_by_map50": best_map50["model"] if best_map50 else None,
        "best_model_by_f1_score": best_f1["model"] if best_f1 else None,
        "fastest_model_by_fps": fastest["model"] if fastest else None,
        "smallest_model_by_checkpoint_size": smallest["model"] if smallest else None,
        "tables": [path.name for path in table_paths],
        "graphs": [path.name for path in graph_paths],
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    return path, summary


def _write_model_runs_summary(path, model_name, runs, metric_rows, table_paths, graph_paths):
    best_map50 = _best_row(metric_rows, "mAP@50")
    best_f1 = _best_row(metric_rows, "F1-score")
    fastest = _best_row(metric_rows, "FPS")
    smallest = _best_row(metric_rows, "Checkpoint size (MB)", smallest=True)

    summary = {
        "model": model_name,
        "runs_compared": [run["model"] for run in runs],
        "runs_used": {
            run["model"]: run["metric_row"]["run_name"]
            for run in runs
        },
        "best_run_by_map50": best_map50["model"] if best_map50 else None,
        "best_run_by_f1_score": best_f1["model"] if best_f1 else None,
        "fastest_run_by_fps": fastest["model"] if fastest else None,
        "smallest_run_by_checkpoint_size": smallest["model"] if smallest else None,
        "tables": [path.name for path in table_paths],
        "graphs": [path.name for path in graph_paths],
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    return path, summary


def _write_report(path, summary, metric_rows):
    lines = [
        "# Model Comparison Report",
        "",
        "## Compared Models",
    ]

    for model, run_name in summary["runs_used"].items():
        lines.append(f"- {model}: `{run_name}`")

    lines.extend([
        "",
        "## Best Models",
        f"- Best mAP@50: **{summary['best_model_by_map50']}**",
        f"- Best F1-score: **{summary['best_model_by_f1_score']}**",
        f"- Fastest by FPS: **{summary['fastest_model_by_fps']}**",
        f"- Smallest checkpoint: **{summary['smallest_model_by_checkpoint_size']}**",
        "",
        "## Per-Model Notes",
    ])

    for row in metric_rows:
        lines.append(
            "- {model}: mAP@50={map50}, F1={f1}, FPS={fps}, checkpoint={size} MB.".format(
                model=row["model"],
                map50=_format_value(row.get("mAP@50")) or "n/a",
                f1=_format_value(row.get("F1-score")) or "n/a",
                fps=_format_value(row.get("FPS")) or "n/a",
                size=_format_value(row.get("Checkpoint size (MB)"), decimals=2) or "n/a",
            )
        )

    if any(not row.get("FPS") for row in metric_rows):
        lines.extend([
            "",
            "Empty FPS values mean that inference performance was not recorded for that experiment.",
        ])

    lines.extend([
        "",
        "Training time was intentionally excluded from this comparison.",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_model_runs_report(path, summary, metric_rows):
    lines = [
        f"# {summary['model']} Runs Comparison Report",
        "",
        "## Compared Runs",
    ]

    for alias, run_name in summary["runs_used"].items():
        lines.append(f"- {alias}: `{run_name}`")

    lines.extend([
        "",
        "## Best Runs",
        f"- Best mAP@50: **{summary['best_run_by_map50']}**",
        f"- Best F1-score: **{summary['best_run_by_f1_score']}**",
        f"- Fastest by FPS: **{summary['fastest_run_by_fps']}**",
        f"- Smallest checkpoint: **{summary['smallest_run_by_checkpoint_size']}**",
        "",
        "## Per-Run Notes",
    ])

    for row in metric_rows:
        lines.append(
            "- {model}: mAP@50={map50}, F1={f1}, FPS={fps}, checkpoint={size} MB.".format(
                model=row["model"],
                map50=_format_value(row.get("mAP@50")) or "n/a",
                f1=_format_value(row.get("F1-score")) or "n/a",
                fps=_format_value(row.get("FPS")) or "n/a",
                size=_format_value(row.get("Checkpoint size (MB)"), decimals=2) or "n/a",
            )
        )

    lines.extend([
        "",
        "Training time was intentionally excluded from this comparison.",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build_comparison_outputs(output_dir, runs, title_prefix, summary_writer, report_writer):
    output_dir.mkdir(parents=True, exist_ok=True)

    for obsolete_name in (
        "comparison_hyperparameters.csv",
        "comparison_hyperparameters_table.png",
    ):
        obsolete_path = output_dir / obsolete_name
        if obsolete_path.exists():
            obsolete_path.unlink()

    metric_rows = [
        {
            column: _format_column_value(column, run["metric_row"].get(column))
            for column in METRIC_COLUMNS
        }
        for run in runs
    ]
    hyperparameter_rows = [
        {
            column: _format_value(run["hyperparameter_row"].get(column))
            for column in HYPERPARAMETER_COLUMNS
        }
        for run in runs
    ]

    table_paths = [
        _write_csv(output_dir / "comparison_metrics.csv", metric_rows, METRIC_COLUMNS),
        _save_table_png(
            output_dir / "comparison_metrics_table.png",
            metric_rows,
            METRIC_COLUMNS,
            f"{title_prefix} Metrics",
        ),
        _write_csv(
            output_dir / f"comparison_{TRAINING_CONFIGURATION_FILE}.csv",
            hyperparameter_rows,
            HYPERPARAMETER_COLUMNS,
        ),
        _save_table_png(
            output_dir / f"comparison_{TRAINING_CONFIGURATION_FILE}_table.png",
            hyperparameter_rows,
            HYPERPARAMETER_COLUMNS,
            f"{title_prefix} Training Configuration",
        ),
    ]

    graph_paths = []
    graph_specs = [
        ("bar_map.png", "mAP", "mAP", "mAP"),
        ("bar_map50.png", "mAP@50", "mAP@50", "mAP@50"),
        ("bar_map50_95.png", "mAP@50:95", "mAP@50:95", "mAP@50:95"),
        ("bar_precision.png", "Precision", "Precision", "Precision"),
        ("bar_recall.png", "Recall", "Recall", "Recall"),
        ("bar_f1.png", "F1-score", "F1-score", "F1-score"),
        ("bar_fps.png", "FPS", "FPS", "FPS"),
        (
            "bar_inference_time.png",
            "Inference time per image (s)",
            "Inference Time per Image",
            "Seconds",
        ),
        (
            "bar_checkpoint_size.png",
            "Checkpoint size (MB)",
            "Checkpoint Size",
            "MB",
        ),
    ]

    for file_name, column, title, ylabel in graph_specs:
        graph_paths.append(
            _bar_chart(output_dir / file_name, metric_rows, column, title, ylabel)
        )

    graph_paths.append(
        _grouped_quality_chart(output_dir / "grouped_quality_metrics.png", metric_rows)
    )

    line_specs = [
        ("line_train_loss.png", "train_loss", "Train Loss by Epoch", "Train loss", None),
        ("line_val_loss.png", "val_loss", "Validation Loss by Epoch", "Validation loss", None),
        ("line_map50.png", "map50", "mAP@50 by Epoch", "mAP@50", None),
        ("line_map50_95.png", "map50_95", "mAP@50:95 by Epoch", "mAP@50:95", None),
        ("line_precision.png", "precision", "Precision by Epoch", "Precision", None),
        ("line_recall.png", "recall", "Recall by Epoch", "Recall", None),
        (
            "line_loss_all_models.png",
            "val_loss",
            "Loss by Epoch",
            "Validation loss or train loss",
            "train_loss",
        ),
        (
            "line_map50_all_models.png",
            "map50",
            "mAP@50 by Epoch",
            "mAP@50",
            None,
        ),
    ]

    for file_name, key, title, ylabel, fallback_key in line_specs:
        path = _line_chart(
            output_dir / file_name,
            runs,
            key,
            title,
            ylabel,
            fallback_key=fallback_key,
        )
        if path is not None:
            graph_paths.append(path)

    graph_paths.append(_radar_chart(output_dir / "radar_quality_metrics.png", metric_rows))

    summary_path, summary = summary_writer(
        output_dir / "summary.json",
        runs,
        metric_rows,
        table_paths,
        graph_paths,
    )
    report_path = report_writer(output_dir / "comparison_report.md", summary, metric_rows)

    return {
        "metric_rows": metric_rows,
        "hyperparameter_rows": hyperparameter_rows,
        "table_paths": table_paths,
        "graph_paths": graph_paths,
        "summary_path": summary_path,
        "report_path": report_path,
        "summary": summary,
    }


def compare_models(verbose=True):
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    performance_cache = _load_performance_cache()

    runs = []
    for model_name in MODEL_ORDER:
        run_dir = _find_best_experiment(model_name)
        if run_dir is None:
            continue
        runs.append(_collect_run(model_name, run_dir, performance_cache=performance_cache))

    outputs = _build_comparison_outputs(
        output_dir=COMPARISON_DIR,
        runs=runs,
        title_prefix="Comparison",
        summary_writer=_write_summary,
        report_writer=_write_report,
    )
    summary = outputs["summary"]

    if verbose:
        print("=" * 50, flush=True)
        print("Model comparison completed", flush=True)
        print("=" * 50, flush=True)
        print(f"Compared models: {', '.join(summary['models_compared'])}", flush=True)
        print(f"Output directory: {COMPARISON_DIR}", flush=True)
        print(f"Summary: {outputs['summary_path']}", flush=True)
        print(f"Report: {outputs['report_path']}", flush=True)
        print("=" * 50, flush=True)

    return COMPARISON_DIR


def _model_alias(model_name, index):
    display_name = {
        "yolo": "YOLO",
        "faster_rcnn": "Faster R-CNN",
        "detr": "DETR",
        "retinanet": "RetinaNet",
        "ssd": "SSD",
    }.get(model_name, model_name)

    return f"{display_name} {index}"


def compare_model_runs(model_name, verbose=True):
    if model_name not in MODEL_ORDER:
        supported = ", ".join(MODEL_ORDER)
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {supported}")

    output_dir = _next_numbered_dir(MODEL_RUNS_COMPARISON_DIR / model_name)
    performance_cache = _load_performance_cache()
    run_dirs = _find_evaluated_experiments(model_name)

    runs = []
    for index, run_dir in enumerate(run_dirs, start=1):
        run = _collect_run(model_name, run_dir, performance_cache=performance_cache)
        alias = _model_alias(model_name, index)
        run["model"] = alias
        run["metric_row"]["model"] = alias
        run["hyperparameter_row"]["model"] = alias
        runs.append(run)

    outputs = _build_comparison_outputs(
        output_dir=output_dir,
        runs=runs,
        title_prefix=f"{_model_alias(model_name, '').strip()} Runs Comparison",
        summary_writer=lambda path, runs, metric_rows, table_paths, graph_paths: (
            _write_model_runs_summary(
                path,
                model_name,
                runs,
                metric_rows,
                table_paths,
                graph_paths,
            )
        ),
        report_writer=_write_model_runs_report,
    )

    if verbose:
        print("=" * 50, flush=True)
        print("Model runs comparison completed", flush=True)
        print("=" * 50, flush=True)
        print(f"Model: {model_name}", flush=True)
        print(f"Compared runs: {len(runs)}", flush=True)
        print(f"Output directory: {output_dir}", flush=True)
        print(f"Summary: {outputs['summary_path']}", flush=True)
        print(f"Report: {outputs['report_path']}", flush=True)
        print("=" * 50, flush=True)

    return output_dir


def main():
    compare_models(verbose=True)


if __name__ == "__main__":
    main()
