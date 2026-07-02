import csv
import random
import time
from datetime import datetime

import torch

from configs.loader import load_model_config
from src.dataset.loaders.manifest import load_manifest
from src.dataset.pipeline.status import ensure_dataset_prepared
from src.dataset.pytorch.dataloaders import create_detection_dataloaders
from src.evaluation.detection_metrics import evaluate_predictions
from src.evaluation.ssd import collect_predictions, evaluate_ssd
from src.models.ssd.model import build_ssd, save_checkpoint
from src.utils.experiment_manager import (
    clean_experiment_artifacts,
    create_experiment,
    register_experiment,
    relative_path,
    save_experiment_info,
    save_metrics,
)
from src.utils.paths import TRAIN_MANIFEST_PATH
from src.utils.plotting import plot_training_history


def _resolve_device(config_device):
    if torch.cuda.is_available() and str(config_device) != "cpu":
        return torch.device(f"cuda:{config_device}")

    return torch.device("cpu")


def _set_seed(seed, deterministic):
    if seed is None:
        return

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _move_targets_to_device(targets, device):
    return [
        {key: value.to(device) for key, value in target.items()}
        for target in targets
    ]


def train_one_epoch(
    model,
    optimizer,
    loader,
    device,
    epoch,
    total_epochs,
    max_batches=None,
    log_interval=50,
):
    model.train()
    total_loss = 0.0
    batches = 0
    started_at = time.perf_counter()
    total_batches = len(loader)

    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    for batch_index, (images, targets) in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        images = [image.to(device) for image in images]
        targets = _move_targets_to_device(targets, device)

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(loss):
            print(
                f"Epoch {epoch}/{total_epochs} | "
                f"batch {batch_index}/{total_batches} | "
                "non-finite loss detected, skipping batch",
                flush=True,
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        batches += 1

        if batch_index == 1 or batch_index % log_interval == 0:
            elapsed = time.perf_counter() - started_at
            print(
                f"Epoch {epoch}/{total_epochs} | "
                f"batch {batch_index}/{total_batches} | "
                f"loss={loss.item():.4f} | "
                f"elapsed={elapsed / 60:.1f}m",
                flush=True,
            )

    return total_loss / max(batches, 1)


def evaluate_loss(model, loader, device, max_batches=None):
    model.train()
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break

            images = [image.to(device) for image in images]
            targets = _move_targets_to_device(targets, device)
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(loss):
                continue

            total_loss += float(loss.detach().cpu())
            batches += 1

    return total_loss / max(batches, 1)


def write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "lr",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def train_ssd(verbose=True):
    ensure_dataset_prepared(verbose=verbose)

    config = load_model_config("ssd")
    training_config = config["training"]
    augmentation_config = config.get("augmentation", {})

    epochs = training_config["epochs"]
    batch_size = training_config["batch_size"]
    workers = training_config["workers"]
    learning_rate = training_config["learning_rate"]
    weight_decay = training_config["weight_decay"]
    seed = training_config.get("seed")
    deterministic = training_config.get("deterministic", False)
    max_train_batches = training_config.get("max_train_batches")
    max_val_batches = training_config.get("max_val_batches")
    device = _resolve_device(training_config.get("device", 0))

    _set_seed(seed, deterministic)

    manifest = load_manifest(TRAIN_MANIFEST_PATH)
    class_names = manifest.classes
    class_ids = list(range(1, len(class_names) + 1))
    num_classes = len(class_names) + 1

    train_loader, val_loader, _ = create_detection_dataloaders(
        batch_size=batch_size,
        num_workers=workers,
        shuffle_train=True,
        augmentation_config=augmentation_config,
        label_offset=1,
        drop_last_train=True,
    )

    experiment = create_experiment("ssd", config)

    if verbose:
        print("=" * 50, flush=True)
        print("SSD training started", flush=True)
        print("=" * 50, flush=True)
        print(f"Experiment: {experiment.run_name}", flush=True)
        print(f"Epochs: {epochs}", flush=True)
        print(f"Batch size: {batch_size}", flush=True)
        print(f"Workers: {workers}", flush=True)
        print(f"Device: {device}", flush=True)
        print(f"Seed: {seed}", flush=True)
        print(f"Deterministic: {deterministic}", flush=True)
        print(f"Max train batches: {max_train_batches}", flush=True)
        print(f"Max val batches: {max_val_batches}", flush=True)
        print("=" * 50, flush=True)

    model = build_ssd(
        num_classes=num_classes,
        pretrained=config["model"].get("pretrained", True),
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = []
    best_map50 = -1.0
    best_metrics = {}
    train_started_at = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            max_batches=max_train_batches,
        )
        val_loss = evaluate_loss(
            model,
            val_loader,
            device,
            max_batches=max_val_batches,
        )
        predictions, targets, _ = collect_predictions(
            model,
            val_loader,
            device,
            max_batches=max_val_batches,
        )
        val_metrics = evaluate_predictions(
            predictions,
            targets,
            class_ids=class_ids,
            confidence_threshold=config["evaluation"]["confidence_threshold"],
        )
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "map50": val_metrics["map50"],
            "map50_95": val_metrics["map50_95"],
            "lr": current_lr,
        }
        history.append(row)
        write_history(experiment.history_path, history)

        save_checkpoint(
            experiment.checkpoints_dir / "last.pt",
            model,
            optimizer,
            epoch,
            val_metrics,
        )

        if val_metrics["map50"] >= best_map50:
            best_map50 = val_metrics["map50"]
            best_metrics = val_metrics
            save_checkpoint(
                experiment.checkpoints_dir / "best.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
            )

        if verbose:
            print(
                f"Epoch {epoch}/{epochs} completed | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"mAP50={val_metrics['map50']:.4f}",
                flush=True,
            )

    training_time = time.perf_counter() - train_started_at
    plots = plot_training_history(experiment.history_path, experiment.plots_dir)

    metrics = {
        "model": "ssd",
        "run_name": experiment.run_name,
        "status": "pending_evaluation",
        "best_validation_metrics": {
            key: best_metrics.get(key)
            for key in ("map", "map50", "map50_95", "precision", "recall", "f1")
        },
        "performance": {
            "training_time_seconds": training_time,
            "average_epoch_time_seconds": training_time / max(epochs, 1),
        },
        "note": "Final test metrics will be written by the evaluate command.",
    }
    metrics_path = save_metrics(experiment, metrics)

    info = {
        "run_name": experiment.run_name,
        "model": "ssd",
        "created_at": experiment.created_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "epochs": epochs,
        "batch_size": batch_size,
        "workers": workers,
        "device": str(device),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "deterministic": deterministic,
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "max_test_batches": training_config.get("max_test_batches"),
        "classes": class_names,
        "num_classes": num_classes,
        "config": config,
        "best_checkpoint": relative_path(experiment.checkpoints_dir / "best.pt"),
        "last_checkpoint": relative_path(experiment.checkpoints_dir / "last.pt"),
        "history_csv": relative_path(experiment.history_path),
        "metrics_json": relative_path(metrics_path),
        "config_yaml": relative_path(experiment.config_path),
        "plots": [relative_path(path) for path in plots],
    }
    info_path = save_experiment_info(experiment.root_dir, info)

    register_experiment(
        experiment,
        {
            "status": "trained",
            "completed_at": info["completed_at"],
            "epochs": epochs,
            "image_size": training_config["image_size"],
            "batch_size": batch_size,
            "device": str(device),
            "best_checkpoint": relative_path(experiment.checkpoints_dir / "best.pt"),
            "last_checkpoint": relative_path(experiment.checkpoints_dir / "last.pt"),
        },
    )

    clean_experiment_artifacts(experiment.root_dir)

    if verbose:
        print()
        print("=" * 50, flush=True)
        print("SSD training completed", flush=True)
        print("=" * 50, flush=True)
        print(f"Experiment directory: {experiment.root_dir}", flush=True)
        print(f"Metrics: {metrics_path}", flush=True)
        print(f"Info: {info_path}", flush=True)
        print("=" * 50, flush=True)
        print()
        print("=" * 50, flush=True)
        print("Running SSD test evaluation", flush=True)
        print("=" * 50, flush=True)

    evaluate_ssd(experiment_dir=experiment.root_dir, verbose=verbose)

    return experiment.root_dir


def main():
    train_ssd(verbose=True)


if __name__ == "__main__":
    main()
