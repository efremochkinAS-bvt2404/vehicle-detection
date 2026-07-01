import random
from copy import deepcopy
from pathlib import Path

import yaml

from src.utils.paths import CONFIGS_DIR, DEFAULT_CONFIG_PATH


MODEL_CONFIG_FILES = {
    "yolo": "yolo.yaml",
    "ssd": "ssd.yaml",
    "retinanet": "retinanet.yaml",
    "faster_rcnn": "faster_rcnn.yaml",
    "detr": "detr.yaml",
}


def deep_merge(base, override):
    merged = deepcopy(base)

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


def load_yaml(path):
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return data or {}


def load_config(config_path=None):
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_model_config(model_name, config_path=None):
    if model_name not in MODEL_CONFIG_FILES:
        supported = ", ".join(sorted(MODEL_CONFIG_FILES))
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {supported}")

    base_config = load_yaml(DEFAULT_CONFIG_PATH)
    model_path = Path(config_path) if config_path else CONFIGS_DIR / MODEL_CONFIG_FILES[model_name]
    model_config = load_yaml(model_path)

    return deep_merge(base_config, model_config)


CONFIG = load_config()

KEEP_CLASSES = set(CONFIG["classes"]["keep"])

CLASS_NAMES_RU = CONFIG["classes"]["labels_ru"]
CLASS_COLORS = CONFIG["classes"]["colors"]

SPLIT_RATIOS = CONFIG["split"]

TRAIN_RATIO = SPLIT_RATIOS["train"]
VAL_RATIO = SPLIT_RATIOS["val"]
TEST_RATIO = SPLIT_RATIOS["test"]

RANDOM_SEED = CONFIG["random_seed"]


def create_rng():
    return random.Random(RANDOM_SEED)
