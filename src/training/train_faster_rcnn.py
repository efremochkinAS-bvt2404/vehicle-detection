import csv
import random
from datetime import datetime

import torch

from configs.loader import load_model_config
from src.dataset.loaders.manifest import load_manifest
from src.dataset.pipeline.status import ensure_dataset_prepared
from src.dataset.pytorch.dataloaders import create_detection_dataloaders
from src.evaluation.detection_metrics import evaluate_predictions
from src.evaluation.faster_rcnn import collect_predictions, evaluate_faster_rcnn
from src.models.faster_rcnn.model import build_faster_rcnn, save_checkpoint
from src.utils.experiment_manager import (
    clean_experiment_artifacts,
    create_experiment,
    register_experiment,
    save_experiment_info,
    save_metrics,
    relative_path,
)
from src.utils.optimizers import build_optimizer
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


def _loss_to_float(loss_dict):
    return float(sum(loss for loss in loss_dict.values()).detach().cpu())


def train_one_epoch(model, optimizer, loader, device, max_batches=None):
    model.train()
    total_loss = 0.0
    batches = 0

    for batch_index, (images, targets) in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        images = [image.to(device) for image in images]
        targets = _move_targets_to_device(targets, device)

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        batches += 1

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
            total_loss += _loss_to_float(model(images, targets))
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


def train_faster_rcnn(verbose=True):
    ensure_dataset_prepared(verbose=verbose)

    config = load_model_config("faster_rcnn")
    training_config = config["training"]
    augmentation_config = config.get("augmentation", {})

    epochs = training_config["epochs"]
    batch_size = training_config["batch_size"]
    workers = training_config["workers"]
    learning_rate = training_config["learning_rate"]
    weight_decay = training_config["weight_decay"]
    optimizer_name = training_config.get("optimizer", "AdamW")
    momentum = training_config.get("momentum", 0.9)
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
    )

    experiment = create_experiment("faster_rcnn", config)

    if verbose:
        print("=" * 50)
        print("Faster R-CNN training started")
        print("=" * 50)
        print(f"Experiment: {experiment.run_name}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Device: {device}")
        print(f"Seed: {seed}")
        print(f"Deterministic: {deterministic}")
        print(f"Optimizer: {optimizer_name}")
        print(f"Learning rate: {learning_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Momentum: {momentum}")
        print(f"Max train batches: {max_train_batches}")
        print(f"Max val batches: {max_val_batches}")
        print("=" * 50)

    model = build_faster_rcnn(
        num_classes=num_classes,
        pretrained=config["model"].get("pretrained", True),
    )
    model.to(device)

    optimizer = build_optimizer(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        name=optimizer_name,
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
    )

    history = []
    best_map50 = -1.0
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            max_batches=max_train_batches,
        )
        val_loss = evaluate_loss(
            model,
            val_loader,
            device,
            max_batches=max_val_batches,
        )
        predictions, targets = collect_predictions(
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

        last_checkpoint = experiment.checkpoints_dir / "last.pt"
        save_checkpoint(last_checkpoint, model, optimizer, epoch, val_metrics)

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
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"mAP50={val_metrics['map50']:.4f}"
            )

    plots = plot_training_history(experiment.history_path, experiment.plots_dir)

    metrics = {
        "model": "faster_rcnn",
        "run_name": experiment.run_name,
        "status": "pending_evaluation",
        "best_validation_metrics": {
            key: best_metrics.get(key)
            for key in ("map", "map50", "map50_95", "precision", "recall", "f1")
        },
        "note": "Final test metrics will be written by the evaluate command.",
    }
    metrics_path = save_metrics(experiment, metrics)

    info = {
        "run_name": experiment.run_name,
        "model": "faster_rcnn",
        "created_at": experiment.created_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "epochs": epochs,
        "batch_size": batch_size,
        "workers": workers,
        "device": str(device),
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "seed": seed,
        "deterministic": deterministic,
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "max_test_batches": training_config.get("max_test_batches"),
        "classes": class_names,
        "num_classes_with_background": num_classes,
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
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "best_checkpoint": str(experiment.checkpoints_dir / "best.pt"),
            "last_checkpoint": str(experiment.checkpoints_dir / "last.pt"),
        },
    )

    clean_experiment_artifacts(experiment.root_dir)

    if verbose:
        print()
        print("=" * 50)
        print("Faster R-CNN training completed")
        print("=" * 50)
        print(f"Experiment directory: {experiment.root_dir}")
        print(f"Metrics: {metrics_path}")
        print(f"Info: {info_path}")
        print("=" * 50)
        print()
        print("=" * 50)
        print("Running Faster R-CNN test evaluation")
        print("=" * 50)

    evaluate_faster_rcnn(experiment_dir=experiment.root_dir, verbose=verbose)

    return experiment.root_dir


def main():
    train_faster_rcnn(verbose=True)


if __name__ == "__main__":
    main()
