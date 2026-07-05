from datetime import datetime
from pathlib import Path
import shutil

from ultralytics import YOLO

from configs.loader import load_model_config
from src.dataset.pipeline.status import ensure_dataset_prepared
from src.utils.checkpoints import ensure_yolo_checkpoint
from src.utils.experiment_manager import (
    clean_experiment_artifacts,
    copy_file,
    create_experiment,
    register_experiment,
    save_experiment_info,
    save_metrics,
    relative_path,
)
from src.utils.paths import (
    YOLO_DATA_YAML_PATH,
)
from src.utils.plotting import plot_training_history


def train_yolo(verbose=True):
    ensure_dataset_prepared(verbose=verbose)

    config = load_model_config("yolo")
    training_config = config["training"]

    epochs = training_config["epochs"]
    image_size = training_config["image_size"]
    batch_size = training_config["batch_size"]
    workers = training_config["workers"]
    device = training_config["device"]
    fraction = training_config.get("fraction")
    max_minutes = training_config.get("max_minutes")
    patience = training_config.get("patience")
    seed = training_config.get("seed")
    deterministic = training_config.get("deterministic")
    optimizer = training_config.get("optimizer")
    learning_rate = training_config.get("learning_rate")
    weight_decay = training_config.get("weight_decay")
    momentum = training_config.get("momentum")

    checkpoint_path = ensure_yolo_checkpoint()

    if not YOLO_DATA_YAML_PATH.exists():
        raise FileNotFoundError(
            f"YOLO data config not found: {YOLO_DATA_YAML_PATH}"
        )

    experiment = create_experiment("yolo", config)

    if verbose:
        print("=" * 50)
        print("YOLO training started")
        print("=" * 50)
        print(f"Experiment: {experiment.run_name}")
        print(f"Model: {checkpoint_path}")
        print(f"Data: {YOLO_DATA_YAML_PATH}")
        print(f"Epochs: {epochs}")
        print(f"Image size: {image_size}")
        print(f"Batch size: {batch_size}")
        print(f"Device: {device}")
        print(f"Fraction: {fraction}")
        print(f"Max minutes: {max_minutes}")
        print(f"Seed: {seed}")
        print(f"Deterministic: {deterministic}")
        print(f"Optimizer: {optimizer}")
        print(f"Learning rate: {learning_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Momentum: {momentum}")
        print("=" * 50)

    model = YOLO(str(checkpoint_path))

    train_kwargs = {
        "data": str(YOLO_DATA_YAML_PATH),
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch_size,
        "workers": workers,
        "device": device,
        "project": str(experiment.root_dir.parent),
        "name": experiment.run_name,
        "exist_ok": True,
        "plots": False,
    }

    if seed is not None:
        train_kwargs["seed"] = seed

    if deterministic is not None:
        train_kwargs["deterministic"] = deterministic

    if fraction is not None:
        train_kwargs["fraction"] = fraction

    if max_minutes is not None:
        train_kwargs["time"] = max_minutes / 60

    if patience is not None:
        train_kwargs["patience"] = patience

    if optimizer is not None:
        train_kwargs["optimizer"] = optimizer

    if learning_rate is not None:
        train_kwargs["lr0"] = learning_rate

    if weight_decay is not None:
        train_kwargs["weight_decay"] = weight_decay

    if momentum is not None:
        train_kwargs["momentum"] = momentum

    results = model.train(**train_kwargs)

    experiment_dir = Path(results.save_dir)

    weights_dir = experiment_dir / "weights"
    ultralytics_best_checkpoint = weights_dir / "best.pt"
    ultralytics_last_checkpoint = weights_dir / "last.pt"
    results_csv = experiment_dir / "results.csv"

    if not ultralytics_best_checkpoint.exists():
        raise FileNotFoundError(f"best.pt not found: {ultralytics_best_checkpoint}")

    best_checkpoint = copy_file(
        ultralytics_best_checkpoint,
        experiment.checkpoints_dir / "best.pt",
    )
    last_checkpoint = copy_file(
        ultralytics_last_checkpoint,
        experiment.checkpoints_dir / "last.pt",
    )

    if weights_dir.exists():
        shutil.rmtree(weights_dir)

    history_path = copy_file(results_csv, experiment.history_path)
    generated_plots = plot_training_history(experiment.history_path, experiment.plots_dir)

    metrics = {
        "model": "yolo",
        "run_name": experiment.run_name,
        "status": "pending_evaluation",
        "note": "Final test metrics will be written by the evaluate command.",
    }
    metrics_path = save_metrics(experiment, metrics)

    info = {
        "run_name": experiment.run_name,
        "model": "yolo",
        "created_at": experiment.created_at,
        "epochs": epochs,
        "image_size": image_size,
        "batch_size": batch_size,
        "workers": workers,
        "device": device,
        "fraction": fraction,
        "max_minutes": max_minutes,
        "patience": patience,
        "seed": seed,
        "deterministic": deterministic,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "config": config,
        "pretrained_checkpoint": relative_path(checkpoint_path),
        "data_yaml": relative_path(YOLO_DATA_YAML_PATH),
        "experiment_dir": relative_path(experiment_dir),
        "best_checkpoint": relative_path(best_checkpoint),
        "last_checkpoint": relative_path(last_checkpoint) if last_checkpoint else "",
        "history_csv": relative_path(history_path) if history_path else "",
        "metrics_json": relative_path(metrics_path),
        "config_yaml": relative_path(experiment.config_path),
        "plots": [relative_path(path) for path in generated_plots],
    }

    info_path = save_experiment_info(experiment_dir, info)

    register_experiment(
        experiment,
        {
            "status": "trained",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "epochs": epochs,
            "image_size": image_size,
            "batch_size": batch_size,
            "device": device,
            "seed": seed,
            "deterministic": deterministic,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "best_checkpoint": relative_path(best_checkpoint),
            "last_checkpoint": relative_path(last_checkpoint) if last_checkpoint else "",
        },
    )

    if verbose:
        print()
        print("=" * 50)
        print("YOLO training completed")
        print("=" * 50)
        print(f"Experiment: {experiment.run_name}")
        print(f"Experiment directory: {experiment_dir}")
        print(f"Best checkpoint: {best_checkpoint}")
        print(f"History: {history_path}")
        print(f"Metrics: {metrics_path}")
        print(f"Info: {info_path}")
        print("=" * 50)

    clean_experiment_artifacts(experiment_dir)

    if verbose:
        print()
        print("=" * 50)
        print("Running YOLO test evaluation")
        print("=" * 50)

    from src.evaluation.yolo import evaluate_yolo

    evaluate_yolo(experiment_dir=experiment_dir, verbose=verbose)

    return results


def main():
    train_yolo(verbose=True)


if __name__ == "__main__":
    main()
