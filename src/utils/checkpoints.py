import shutil
from pathlib import Path

from ultralytics import YOLO

from src.utils.paths import PRETRAINED_CHECKPOINTS_DIR


YOLO_PRETRAINED_FILE_NAME = "yolo11n.pt"
YOLO_PRETRAINED_CHECKPOINT_PATH = PRETRAINED_CHECKPOINTS_DIR / YOLO_PRETRAINED_FILE_NAME


def ensure_yolo_checkpoint():
    PRETRAINED_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    if YOLO_PRETRAINED_CHECKPOINT_PATH.exists():
        return YOLO_PRETRAINED_CHECKPOINT_PATH

    YOLO(YOLO_PRETRAINED_FILE_NAME)

    downloaded_path = Path(YOLO_PRETRAINED_FILE_NAME)

    if not downloaded_path.exists():
        raise FileNotFoundError(f"Downloaded YOLO checkpoint not found: {downloaded_path}")

    shutil.move(str(downloaded_path), str(YOLO_PRETRAINED_CHECKPOINT_PATH))

    return YOLO_PRETRAINED_CHECKPOINT_PATH