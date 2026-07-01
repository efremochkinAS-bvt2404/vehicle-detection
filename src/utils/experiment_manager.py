import csv
import json
from pathlib import Path


REGISTRY_COLUMNS = [
    "experiment_id",
    "model",
    "created_at",
    "epochs",
    "image_size",
    "batch_size",
    "device",
    "experiment_dir",
    "best_checkpoint",
    "results_csv",
    "args_yaml",
]


def save_experiment_info(experiment_dir: Path, info: dict) -> Path:
    info_path = experiment_dir / "info.json"

    with info_path.open("w", encoding="utf-8") as file:
        json.dump(info, file, indent=4, ensure_ascii=False)

    return info_path


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