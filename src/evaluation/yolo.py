from pathlib import Path

from src.dataset.loaders.manifest import load_manifest
from src.dataset.pipeline.status import ensure_dataset_prepared
from src.utils.paths import (
    EXPERIMENTS_DIR,
    FILTERED_IMAGES_DIR,
    TEST_MANIFEST_PATH,
    YOLO_DATA_YAML_PATH,
)
from src.utils.experiment_manager import clean_experiment_artifacts, relative_path
from src.utils.plotting import plot_training_history


def save_json(path, data):
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    return path


def find_latest_experiment(model_name):
    model_dir = EXPERIMENTS_DIR / model_name

    if not model_dir.exists():
        raise FileNotFoundError(f"No experiments found for model: {model_name}")

    candidates = [
        path
        for path in model_dir.iterdir()
        if path.is_dir() and get_best_checkpoint_path(path).exists()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No trained experiments with checkpoints found in: {model_dir}"
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def get_best_checkpoint_path(experiment_dir):
    experiment_dir = Path(experiment_dir)
    checkpoint_path = experiment_dir / "checkpoints" / "best.pt"

    if checkpoint_path.exists():
        return checkpoint_path

    return experiment_dir / "weights" / "best.pt"


def _get_metric(metrics, *names, default=0.0):
    for name in names:
        value = getattr(metrics.box, name, None)

        if value is not None:
            return float(value)

    return default


def _build_metrics(results, run_name, experiment_dir):
    precision = _get_metric(results, "mp")
    recall = _get_metric(results, "mr")
    map50 = _get_metric(results, "map50")
    map50_95 = _get_metric(results, "map")
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "model": "yolo",
        "run_name": run_name,
        "status": "evaluated",
        "split": "test",
        "metrics": {
            "map": map50_95,
            "map50": map50,
            "map50_95": map50_95,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
        },
        "experiment_dir": relative_path(experiment_dir),
    }


def _prediction_to_dict(box, class_names):
    class_id = int(box.cls.item())

    if isinstance(class_names, dict):
        class_name = class_names.get(class_id, str(class_id))
    else:
        class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)

    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": float(box.conf.item()),
        "bbox": [
            float(value)
            for value in box.xyxy[0].tolist()
        ],
    }


def _select_test_images(limit):
    manifest = load_manifest(TEST_MANIFEST_PATH)
    return manifest.images[:limit]


def save_test_predictions(model, experiment_dir, confidence_threshold, limit):
    from src.utils.visualization import save_prediction_annotation

    output_dir = experiment_dir / "predictions" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_images = _select_test_images(limit)
    class_names = model.names
    saved = []

    for annotation in selected_images:
        image_path = FILTERED_IMAGES_DIR / annotation.file_name

        if not image_path.exists():
            continue

        results = model.predict(
            source=str(image_path),
            conf=confidence_threshold,
            verbose=False,
        )

        predictions = [
            _prediction_to_dict(box, class_names)
            for box in results[0].boxes
        ]

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


def normalize_evaluation_outputs(evaluation_dir):
    evaluation_dir = Path(evaluation_dir)

    rename_map = {
        "BoxPR_curve.png": "pr_curve.png",
        "BoxP_curve.png": "precision_curve.png",
        "BoxR_curve.png": "recall_curve.png",
        "BoxF1_curve.png": "f1_curve.png",
    }

    for source_name, target_name in rename_map.items():
        source_path = evaluation_dir / source_name
        target_path = evaluation_dir / target_name

        if source_path.exists():
            if target_path.exists():
                target_path.unlink()

            source_path.rename(target_path)


def evaluate_yolo(experiment_dir=None, verbose=True):
    from ultralytics import YOLO
    from configs.loader import load_model_config

    ensure_dataset_prepared(verbose=verbose)

    config = load_model_config("yolo")
    evaluation_config = config["evaluation"]
    confidence_threshold = evaluation_config["confidence_threshold"]
    prediction_images = evaluation_config["prediction_images"]

    experiment_dir = (
        Path(experiment_dir)
        if experiment_dir is not None
        else find_latest_experiment("yolo")
    )
    run_name = experiment_dir.name
    checkpoint_path = get_best_checkpoint_path(experiment_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    if not YOLO_DATA_YAML_PATH.exists():
        raise FileNotFoundError(f"YOLO data config not found: {YOLO_DATA_YAML_PATH}")

    if verbose:
        print("=" * 50)
        print("YOLO evaluation started")
        print("=" * 50)
        print(f"Experiment: {run_name}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Data: {YOLO_DATA_YAML_PATH}")
        print("=" * 50)

    model = YOLO(str(checkpoint_path))

    results = model.val(
        data=str(YOLO_DATA_YAML_PATH),
        split="test",
        conf=confidence_threshold,
        project=str(experiment_dir),
        name="evaluation",
        exist_ok=True,
        plots=True,
        verbose=False,
    )

    evaluation_dir = experiment_dir / "evaluation"
    normalize_evaluation_outputs(evaluation_dir)

    metrics = _build_metrics(results, run_name, experiment_dir)
    metrics["predictions"] = {
        "confidence_threshold": confidence_threshold,
        "requested_images": prediction_images,
    }

    history_path = experiment_dir / "history.csv"

    if not history_path.exists():
        history_path = experiment_dir / "results.csv"

    plots_dir = experiment_dir / "plots"
    plots = plot_training_history(history_path, plots_dir)

    prediction_paths = save_test_predictions(
        model=model,
        experiment_dir=experiment_dir,
        confidence_threshold=confidence_threshold,
        limit=prediction_images,
    )

    metrics["predictions"]["saved_images"] = len(prediction_paths)
    metrics["plots"] = [relative_path(path) for path in plots]

    metrics_path = save_json(experiment_dir / "metrics.json", metrics)
    clean_experiment_artifacts(experiment_dir)

    if verbose:
        print()
        print("=" * 50)
        print("YOLO evaluation completed")
        print("=" * 50)
        print(f"Metrics: {metrics_path}")
        print(f"Prediction images: {len(prediction_paths)}")
        print(f"Plots: {len(plots)}")
        print("=" * 50)

    return metrics
