from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from configs.loader import load_model_config
from src.dataset.loaders.manifest import load_manifest
from src.dataset.pipeline.status import ensure_dataset_prepared
from src.dataset.pytorch.detr_dataset import DetrDetectionDataset, detr_collate_fn
from src.evaluation.detection_metrics import (
    evaluate_predictions,
    save_detection_evaluation_plots,
)
from src.models.detr.model import build_detr_processor, load_model_from_checkpoint
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


def create_detr_test_loader(config, image_processor):
    training_config = config["training"]

    test_dataset = DetrDetectionDataset(
        manifest_file=TEST_MANIFEST_PATH,
        images_dir=FILTERED_IMAGES_DIR,
        image_processor=image_processor,
        augmentation_config=config.get("augmentation", {}),
        train=False,
    )

    return DataLoader(
        test_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["workers"],
        collate_fn=detr_collate_fn,
    )


def _move_labels_to_device(labels, device):
    return [
        {key: value.to(device) for key, value in label.items()}
        for label in labels
    ]


def _target_sizes_from_labels(labels):
    return torch.stack([
        label["orig_size"].detach().cpu()
        for label in labels
    ])


def _convert_post_processed(post_processed):
    predictions = []

    for item in post_processed:
        predictions.append(
            {
                "boxes": item["boxes"].detach().cpu(),
                "labels": item["labels"].detach().cpu() + 1,
                "scores": item["scores"].detach().cpu(),
            }
        )

    return predictions


def collect_predictions(
    model,
    image_processor,
    loader,
    device,
    confidence_threshold=0.25,
    max_batches=None,
):
    model.eval()
    predictions = []
    targets = []
    total_inference_time = 0.0
    total_images = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break

            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            start_time = time.perf_counter()
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            total_inference_time += time.perf_counter() - start_time
            total_images += pixel_values.shape[0]

            target_sizes = _target_sizes_from_labels(batch["labels"])
            post_processed = image_processor.post_process_object_detection(
                outputs,
                threshold=confidence_threshold,
                target_sizes=target_sizes,
            )

            predictions.extend(_convert_post_processed(post_processed))
            targets.extend([
                {
                    key: value.detach().cpu()
                    for key, value in target.items()
                    if key in {"boxes", "labels", "image_id"}
                }
                for target in batch["metric_targets"]
            ])

    avg_inference_time = (
        total_inference_time / total_images
        if total_images > 0
        else 0.0
    )

    return predictions, targets, avg_inference_time


def _prediction_to_dict(prediction, class_names):
    converted = []

    for box, label, score in zip(
        prediction["boxes"],
        prediction["labels"],
        prediction["scores"],
    ):
        class_index = int(label) - 1
        class_name = (
            class_names[class_index]
            if 0 <= class_index < len(class_names)
            else str(int(label))
        )
        converted.append(
            {
                "class_id": class_index,
                "class_name": class_name,
                "confidence": float(score),
                "bbox": [float(value) for value in box.tolist()],
            }
        )

    return converted


def save_test_predictions(
    model,
    image_processor,
    experiment_dir,
    device,
    confidence_threshold,
    limit,
):
    manifest = load_manifest(TEST_MANIFEST_PATH)
    output_dir = experiment_dir / "predictions" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = manifest.classes
    saved = []
    model.eval()

    from PIL import Image

    with torch.no_grad():
        for annotation in manifest.images[:limit]:
            image_path = FILTERED_IMAGES_DIR / annotation.file_name

            if not image_path.exists():
                continue

            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")

            encoding = image_processor(images=image, return_tensors="pt")
            pixel_values = encoding["pixel_values"].to(device)
            pixel_mask = encoding["pixel_mask"].to(device)
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            target_sizes = torch.tensor([image.size[::-1]])
            post_processed = image_processor.post_process_object_detection(
                outputs,
                threshold=confidence_threshold,
                target_sizes=target_sizes,
            )
            prediction = _convert_post_processed(post_processed)[0]
            predictions = _prediction_to_dict(prediction, class_names)
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


def evaluate_detr(experiment_dir=None, verbose=True):
    ensure_dataset_prepared(verbose=verbose)

    config = load_model_config("detr")
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    confidence_threshold = evaluation_config["confidence_threshold"]
    prediction_images = evaluation_config["prediction_images"]
    device = _resolve_device(training_config.get("device", 0))
    model_name = config["model"]["architecture"]

    experiment_dir = (
        Path(experiment_dir)
        if experiment_dir is not None
        else find_latest_experiment("detr")
    )
    checkpoint_path = experiment_dir / "checkpoints" / "best.pt"

    manifest = load_manifest(TEST_MANIFEST_PATH)
    class_names = manifest.classes
    class_ids = list(range(1, len(class_names) + 1))
    num_labels = len(class_names)

    if verbose:
        print("=" * 50, flush=True)
        print("DETR evaluation started", flush=True)
        print("=" * 50, flush=True)
        print(f"Experiment: {experiment_dir.name}", flush=True)
        print(f"Checkpoint: {checkpoint_path}", flush=True)
        print(f"Device: {device}", flush=True)
        print("=" * 50, flush=True)

    image_processor = build_detr_processor(model_name)
    model, _ = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        num_labels=num_labels,
        device=device,
    )

    test_loader = create_detr_test_loader(config, image_processor)
    predictions, targets, inference_time = collect_predictions(
        model=model,
        image_processor=image_processor,
        loader=test_loader,
        device=device,
        confidence_threshold=confidence_threshold,
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

    plots = plot_training_history(
        experiment_dir / "history.csv",
        experiment_dir / "plots",
    )
    prediction_paths = save_test_predictions(
        model=model,
        image_processor=image_processor,
        experiment_dir=experiment_dir,
        device=device,
        confidence_threshold=confidence_threshold,
        limit=prediction_images,
    )

    metrics = {
        "model": "detr",
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
        print("DETR evaluation completed", flush=True)
        print("=" * 50, flush=True)
        print(f"Metrics: {metrics_path}", flush=True)
        print(f"Prediction images: {len(prediction_paths)}", flush=True)
        print("=" * 50, flush=True)

    return metrics


def main():
    evaluate_detr(verbose=True)


if __name__ == "__main__":
    main()
