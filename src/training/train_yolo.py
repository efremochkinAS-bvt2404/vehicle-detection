from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from src.models.yolo.config import (
    EPOCHS,
    IMAGE_SIZE,
    BATCH_SIZE,
    WORKERS,
    DEVICE,
)
from src.utils.checkpoints import ensure_yolo_checkpoint
from src.utils.experiment_manager import (
    save_experiment_info,
    append_experiment_to_registry,
)
from src.utils.paths import (
    YOLO_DATA_YAML_PATH,
    YOLO_EXPERIMENTS_DIR,
    EXPERIMENTS_REGISTRY_PATH,
)


def generate_experiment_name():
    existing_numbers = []

    YOLO_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    for path in YOLO_EXPERIMENTS_DIR.iterdir():
        if not path.is_dir():
            continue

        name = path.name

        if not name.startswith("experiment_"):
            continue

        number_part = name.replace("experiment_", "").split("-")[0]

        if number_part.isdigit():
            existing_numbers.append(int(number_part))

    next_number = max(existing_numbers, default=0) + 1

    return f"experiment_{next_number:03d}"


def train_yolo(verbose=True):
    checkpoint_path = ensure_yolo_checkpoint()

    if not YOLO_DATA_YAML_PATH.exists():
        raise FileNotFoundError(
            f"YOLO data config not found: {YOLO_DATA_YAML_PATH}"
        )

    experiment_id = generate_experiment_name()

    if verbose:
        print("=" * 50)
        print("YOLO training started")
        print("=" * 50)
        print(f"Experiment: {experiment_id}")
        print(f"Model: {checkpoint_path}")
        print(f"Data: {YOLO_DATA_YAML_PATH}")
        print(f"Epochs: {EPOCHS}")
        print(f"Image size: {IMAGE_SIZE}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Device: {DEVICE}")
        print("=" * 50)

    model = YOLO(str(checkpoint_path))

    results = model.train(
        data=str(YOLO_DATA_YAML_PATH),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        project=str(YOLO_EXPERIMENTS_DIR),
        name=experiment_id,
        exist_ok=False,
    )

    experiment_dir = Path(results.save_dir)

    weights_dir = experiment_dir / "weights"
    best_checkpoint = weights_dir / "best.pt"
    last_checkpoint = weights_dir / "last.pt"
    results_csv = experiment_dir / "results.csv"
    args_yaml = experiment_dir / "args.yaml"

    if not best_checkpoint.exists():
        raise FileNotFoundError(f"best.pt not found: {best_checkpoint}")

    created_at = datetime.now().isoformat(timespec="seconds")

    info = {
        "experiment_id": experiment_dir.name,
        "model": "yolo",
        "created_at": created_at,
        "epochs": EPOCHS,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "workers": WORKERS,
        "device": DEVICE,
        "pretrained_checkpoint": str(checkpoint_path),
        "data_yaml": str(YOLO_DATA_YAML_PATH),
        "experiment_dir": str(experiment_dir),
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint) if last_checkpoint.exists() else "",
        "results_csv": str(results_csv) if results_csv.exists() else "",
        "args_yaml": str(args_yaml) if args_yaml.exists() else "",
    }

    info_path = save_experiment_info(experiment_dir, info)

    append_experiment_to_registry(
        EXPERIMENTS_REGISTRY_PATH,
        {
            "experiment_id": experiment_dir.name,
            "model": "yolo",
            "created_at": created_at,
            "epochs": EPOCHS,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "device": DEVICE,
            "experiment_dir": str(experiment_dir),
            "best_checkpoint": str(best_checkpoint),
            "results_csv": str(results_csv) if results_csv.exists() else "",
            "args_yaml": str(args_yaml) if args_yaml.exists() else "",
        },
    )

    if verbose:
        print()
        print("=" * 50)
        print("YOLO training completed")
        print("=" * 50)
        print(f"Experiment: {experiment_dir.name}")
        print(f"Experiment directory: {experiment_dir}")
        print(f"Best checkpoint: {best_checkpoint}")
        print(f"Info: {info_path}")
        print(f"Registry: {EXPERIMENTS_REGISTRY_PATH}")
        print("=" * 50)

    return results


def main():
    train_yolo(verbose=True)


if __name__ == "__main__":
    main()