import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil

import yaml

from src.utils.paths import EXPERIMENTS_DIR, EXPERIMENTS_REGISTRY_PATH


REGISTRY_COLUMNS = [
    "model",
    "run_name",
    "status",
    "created_at",
    "completed_at",
    "epochs",
    "image_size",
    "batch_size",
    "device",
    "experiment_dir",
    "best_checkpoint",
    "last_checkpoint",
    "history_csv",
    "metrics_json",
    "config_yaml",
]


@dataclass
class Experiment:
    model_name: str
    run_name: str
    created_at: str
    root_dir: Path
    checkpoints_dir: Path
    plots_dir: Path
    predictions_dir: Path
    config_path: Path
    history_path: Path
    metrics_path: Path
    info_path: Path


def build_run_name(model_name, created_at=None):
    created_at = created_at or datetime.now()
    timestamp = created_at.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{model_name}_{timestamp}"


def create_experiment(model_name, config, run_name=None):
    created_at = datetime.now()
    run_name = run_name or build_run_name(model_name, created_at)
    root_dir = EXPERIMENTS_DIR / model_name / run_name

    if root_dir.exists():
        base_run_name = run_name
        suffix = 2

        while root_dir.exists():
            run_name = f"{base_run_name}_{suffix:02d}"
            root_dir = EXPERIMENTS_DIR / model_name / run_name
            suffix += 1

    experiment = Experiment(
        model_name=model_name,
        run_name=run_name,
        created_at=created_at.isoformat(timespec="seconds"),
        root_dir=root_dir,
        checkpoints_dir=root_dir / "checkpoints",
        plots_dir=root_dir / "plots",
        predictions_dir=root_dir / "predictions",
        config_path=root_dir / "config.yaml",
        history_path=root_dir / "history.csv",
        metrics_path=root_dir / "metrics.json",
        info_path=root_dir / "info.json",
    )

    for directory in (
        experiment.root_dir,
        experiment.checkpoints_dir,
        experiment.plots_dir,
        experiment.predictions_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    save_yaml(experiment.config_path, config)

    return experiment


def save_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)

    return path


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    return path


def save_experiment_info(experiment_dir: Path, info: dict) -> Path:
    info_path = experiment_dir / "info.json"
    return save_json(info_path, info)


def save_metrics(experiment, metrics):
    return save_json(experiment.metrics_path, metrics)


def copy_file(source, target):
    source = Path(source)
    target = Path(target)

    if not source.exists():
        return None

    target.parent.mkdir(parents=True, exist_ok=True)

    if source.resolve() == target.resolve():
        return target

    shutil.copy2(source, target)
    return target


def copy_matching_files(source_dir, target_dir, patterns):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    copied = []

    if not source_dir.exists():
        return copied

    target_dir.mkdir(parents=True, exist_ok=True)

    for pattern in patterns:
        for source in source_dir.glob(pattern):
            if source.is_file():
                copied_path = copy_file(source, target_dir / source.name)
                if copied_path is not None:
                    copied.append(copied_path)

    return copied


def append_experiment_to_registry(registry_path: Path, row: dict) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = registry_path.exists()

    with registry_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REGISTRY_COLUMNS)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            column: row.get(column, "")
            for column in REGISTRY_COLUMNS
        })


def register_experiment(experiment, row):
    append_experiment_to_registry(
        EXPERIMENTS_REGISTRY_PATH,
        {
            "model": experiment.model_name,
            "run_name": experiment.run_name,
            "created_at": experiment.created_at,
            "experiment_dir": str(experiment.root_dir),
            "config_yaml": str(experiment.config_path),
            "history_csv": str(experiment.history_path)
            if experiment.history_path.exists()
            else "",
            "metrics_json": str(experiment.metrics_path)
            if experiment.metrics_path.exists()
            else "",
            **row,
        },
    )


def clean_directory(directory, allowed_names):
    directory = Path(directory)

    if not directory.exists():
        return

    for path in directory.iterdir():
        if path.name in allowed_names:
            continue

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def clean_experiment_artifacts(experiment_dir):
    experiment_dir = Path(experiment_dir)

    clean_directory(
        experiment_dir,
        {
            "config.yaml",
            "info.json",
            "metrics.json",
            "history.csv",
            "checkpoints",
            "plots",
            "evaluation",
            "predictions",
        },
    )

    clean_directory(
        experiment_dir / "checkpoints",
        {"best.pt", "last.pt"},
    )

    clean_directory(
        experiment_dir / "plots",
        {
            "loss.png",
            "precision.png",
            "recall.png",
            "map50.png",
            "map50_95.png",
            "lr.png",
            "results.png",
        },
    )

    clean_directory(
        experiment_dir / "evaluation",
        {
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "pr_curve.png",
            "precision_curve.png",
            "recall_curve.png",
            "f1_curve.png",
        },
    )

    clean_directory(
        experiment_dir / "predictions",
        {"test"},
    )
