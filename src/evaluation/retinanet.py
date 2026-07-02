from pathlib import Path
import time

import torch

from configs.loader import load_model_config
from src.dataset.loaders.manifest import load_manifest
from src.dataset.pipeline.status import ensure_dataset_prepared
from src.dataset.pytorch.dataloaders import create_detection_dataloaders
from src.evaluation.detection_metrics import (
    evaluate_predictions,
    save_detection_evaluation_plots,
)
from src.models.retinanet.model import load_model_from_checkpoint
from src.utils.experiment_manager import clean_experiment_artifacts, relative_path, save_json
from src.utils.paths import EXPERIMENTS_DIR, FILTERED_IMAGES_DIR, TEST_MANIFEST_PATH
from src.utils.plotting import plot_training_history
from src.utils.visualization import save_prediction_annotation


def find_latest_experiment(model_name):
    model_dir = EXPERIMENTS_DIR / model_name

    if not model_dir.exists():
        raise FileNotFoundError(f"No experiments found for model: {model_name}")

    candidates = [
        path
        for path in model_dir.iterdir()
        if path.is_dir() and (path / "checkpoints" / "best.pt").exists()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No trained experiments with checkpoints found in: {model_dir}"
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_device(config_device):
    if torch.cuda.is_available() and str(config_device) != "cpu":
        return torch.device(f"cuda:{config_device}")

    return torch.device("cpu")


def collect_predictions(model, loader, device, max_batches=None):
    model.eval()
    predictions = []
    targets = []
    total_inference_time = 0.0
    total_images = 0

    with torch.no_grad():
        for batch_index, (images, batch_targets) in enumerate(loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break

            images = [image.to(device) for image in images]

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            started_at = time.perf_counter()
            outputs = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            total_inference_time += time.perf_counter() - started_at
            total_images += len(images)

            predictions.extend([
                _labels_to_metric_format(output)
                for output in outputs
            ])
            targets.extend([
                _labels_to_metric_format(target)
                for target in batch_targets
            ])

    inference_time = total_inference_time / total_images if total_images > 0 else 0.0
    return predictions, targets, inference_time


def _labels_to_metric_format(item):
    converted = {key: value.detach().cpu() for key, value in item.items()}

    if "labels" in converted:
        converted["labels"] = converted["labels"] + 1

    return converted


def _prediction_to_dict(prediction, class_names, confidence_threshold):
    converted = []
    boxes = prediction["boxes"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    scores = prediction["scores"].detach().cpu()

    for box, label, score in zip(boxes, labels, scores):
        score = float(score)

        if score < confidence_threshold:
            continue

        class_index = int(label)
        class_name = (
            class_names[class_index]
            if 0 <= class_index < len(class_names)
            else str(int(label))
        )

        converted.append(
            {
                "class_id": class_index,
                "class_name": class_name,
                "confidence": score,
                "bbox": [float(value) for value in box.tolist()],
            }
        )

    return converted


def save_test_predictions(model, experiment_dir, device, confidence_threshold, limit):
    manifest = load_manifest(TEST_MANIFEST_PATH)
    output_dir = experiment_dir / "predictions" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = manifest.classes
    saved = []
    model.eval()

    with torch.no_grad():
        for annotation in manifest.images[:limit]:
            image_path = FILTERED_IMAGES_DIR / annotation.file_name

            if not image_path.exists():
                continue

            from PIL import Image
            from torchvision.transforms import functional as F

            with Image.open(image_path) as source_image:
                image = F.to_tensor(source_image.convert("RGB")).to(device)

            prediction = model([image])[0]
            predictions = _prediction_to_dict(
                prediction,
                class_names,
                confidence_threshold,
            )
            output_path = output_dir / f"{Path(annotation.file_name).stem}_predicted.png"

            save_prediction_annotation(
                image_path=image_path,
                predictions=predictions,
                output_path=output_path,
                line_width=2,
                font_size=18,
                use_ru_labels=False,
                show_confidence=True,
            )
            saved.append(output_path)

    return saved


def evaluate_retinanet(experiment_dir=None, verbose=True):
    ensure_dataset_prepared(verbose=verbose)

    config = load_model_config("retinanet")
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    confidence_threshold = evaluation_config["confidence_threshold"]
    visualization_confidence_threshold = evaluation_config.get(
        "visualization_confidence_threshold",
        confidence_threshold,
    )
    prediction_images = evaluation_config["prediction_images"]
    device = _resolve_device(training_config.get("device", 0))

    experiment_dir = (
        Path(experiment_dir)
        if experiment_dir is not None
        else find_latest_experiment("retinanet")
    )
    checkpoint_path = experiment_dir / "checkpoints" / "best.pt"

    manifest = load_manifest(TEST_MANIFEST_PATH)
    class_names = manifest.classes
    class_ids = list(range(1, len(class_names) + 1))
    num_classes = len(class_names)

    if verbose:
        print("=" * 50, flush=True)
        print("RetinaNet evaluation started", flush=True)
        print("=" * 50, flush=True)
        print(f"Experiment: {experiment_dir.name}", flush=True)
        print(f"Checkpoint: {checkpoint_path}", flush=True)
        print(f"Device: {device}", flush=True)
        print("=" * 50, flush=True)

    model, _ = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        device=device,
        pretrained=False,
    )

    _, _, test_loader = create_detection_dataloaders(
        batch_size=training_config["batch_size"],
        num_workers=training_config["workers"],
        shuffle_train=False,
        augmentation_config=config.get("augmentation", {}),
        label_offset=0,
    )

    predictions, targets, inference_time = collect_predictions(
        model,
        test_loader,
        device,
        max_batches=training_config.get("max_test_batches"),
    )
    metric_values = evaluate_predictions(
        predictions,
        targets,
        class_ids=class_ids,
        confidence_threshold=confidence_threshold,
    )

    evaluation_dir = experiment_dir / "evaluation"
    evaluation_plots = save_detection_evaluation_plots(
        predictions=predictions,
        targets=targets,
        class_ids=class_ids,
        class_names=class_names,
        output_dir=evaluation_dir,
    )

    plots = plot_training_history(experiment_dir / "history.csv", experiment_dir / "plots")
    prediction_paths = save_test_predictions(
        model=model,
        experiment_dir=experiment_dir,
        device=device,
        confidence_threshold=visualization_confidence_threshold,
        limit=prediction_images,
    )

    metrics = {
        "model": "retinanet",
        "run_name": experiment_dir.name,
        "status": "evaluated",
        "split": "test",
        "metrics": {
            key: metric_values[key]
            for key in ("map", "map50", "map50_95", "precision", "recall", "f1")
        },
        "performance": {
            "inference_time_per_image_seconds": inference_time,
            "fps": 1 / inference_time if inference_time > 0 else 0.0,
            "checkpoint_size_mb": (
                checkpoint_path.stat().st_size / (1024 * 1024)
                if checkpoint_path.exists()
                else 0.0
            ),
        },
        "predictions": {
            "confidence_threshold": confidence_threshold,
            "visualization_confidence_threshold": visualization_confidence_threshold,
            "requested_images": prediction_images,
            "saved_images": len(prediction_paths),
        },
        "experiment_dir": relative_path(experiment_dir),
        "plots": [relative_path(path) for path in plots],
        "evaluation_plots": [relative_path(path) for path in evaluation_plots],
    }

    metrics_path = save_json(experiment_dir / "metrics.json", metrics)
    clean_experiment_artifacts(experiment_dir)

    if verbose:
        print()
        print("=" * 50, flush=True)
        print("RetinaNet evaluation completed", flush=True)
        print("=" * 50, flush=True)
        print(f"Metrics: {metrics_path}", flush=True)
        print(f"Prediction images: {len(prediction_paths)}", flush=True)
        print("=" * 50, flush=True)

    return metrics


def main():
    evaluate_retinanet(verbose=True)


if __name__ == "__main__":
    main()
