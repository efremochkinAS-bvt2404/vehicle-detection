import random

import yaml

from src.utils.paths import DEFAULT_CONFIG_PATH


def load_config():
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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