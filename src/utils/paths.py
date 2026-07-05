from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# Configs
# =============================================================================

CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.yaml"


# =============================================================================
# Data
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# =============================================================================
# Raw KITTI
# =============================================================================

KITTI_RAW_DIR = RAW_DATA_DIR / "KITTI"
KITTI_IMAGES_DIR = KITTI_RAW_DIR / "image_2"
KITTI_LABELS_DIR = KITTI_RAW_DIR / "label_2"


# =============================================================================
# Filtered KITTI
# =============================================================================

FILTERED_DIR = PROCESSED_DATA_DIR / "filtered"
FILTERED_IMAGES_DIR = FILTERED_DIR / "image_2"
FILTERED_LABELS_DIR = FILTERED_DIR / "label_2"


# =============================================================================
# Manifests
# =============================================================================

MANIFESTS_DIR = PROCESSED_DATA_DIR / "manifests"

TRAIN_MANIFEST_PATH = MANIFESTS_DIR / "train.json"
VAL_MANIFEST_PATH = MANIFESTS_DIR / "val.json"
TEST_MANIFEST_PATH = MANIFESTS_DIR / "test.json"


# =============================================================================
# Dataset splits
# =============================================================================

SPLITS_DIR = PROCESSED_DATA_DIR / "splits"

TRAIN_SPLIT_PATH = SPLITS_DIR / "train.txt"
VAL_SPLIT_PATH = SPLITS_DIR / "val.txt"
TEST_SPLIT_PATH = SPLITS_DIR / "test.txt"


# =============================================================================
# Metadata
# =============================================================================

METADATA_DIR = PROCESSED_DATA_DIR / "metadata"
SPLIT_CONFIG_PATH = METADATA_DIR / "split_config.json"
PREPARATION_STATUS_PATH = METADATA_DIR / "preparation_status.json"


# =============================================================================
# YOLO dataset
# =============================================================================

YOLO_DATA_DIR = PROCESSED_DATA_DIR / "yolo"

YOLO_IMAGES_DIR = YOLO_DATA_DIR / "images"
YOLO_LABELS_DIR = YOLO_DATA_DIR / "labels"

YOLO_TRAIN_IMAGES_DIR = YOLO_IMAGES_DIR / "train"
YOLO_VAL_IMAGES_DIR = YOLO_IMAGES_DIR / "val"
YOLO_TEST_IMAGES_DIR = YOLO_IMAGES_DIR / "test"

YOLO_TRAIN_LABELS_DIR = YOLO_LABELS_DIR / "train"
YOLO_VAL_LABELS_DIR = YOLO_LABELS_DIR / "val"
YOLO_TEST_LABELS_DIR = YOLO_LABELS_DIR / "test"

YOLO_DATA_YAML_PATH = YOLO_DATA_DIR / "data.yaml"


# =============================================================================
# Results
# =============================================================================

RESULTS_DIR = PROJECT_ROOT / "results"

METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = RESULTS_DIR / "logs"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
VISUALIZATION_DIR = RESULTS_DIR / "visualization"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

YOLO_RESULTS_DIR = RESULTS_DIR / "yolo"

EXPERIMENTS_DIR = RESULTS_DIR / "experiments"

YOLO_EXPERIMENTS_REGISTRY_PATH = EXPERIMENTS_DIR / "yolo_experiments.csv"

# =============================================================================
# Analysis results
# =============================================================================

KITTI_METRICS_DIR = METRICS_DIR / "kitti"
KITTI_PLOTS_DIR = PLOTS_DIR / "kitti"

FILTERED_KITTI_METRICS_DIR = METRICS_DIR / "filtered_kitti"
FILTERED_KITTI_PLOTS_DIR = PLOTS_DIR / "filtered_kitti"
DATASET_SPLIT_PLOTS_DIR = PLOTS_DIR / "dataset" / "splits"


# =============================================================================
# YOLO results
# =============================================================================

YOLO_METRICS_DIR = METRICS_DIR / "yolo"
YOLO_PLOTS_DIR = PLOTS_DIR / "yolo"
YOLO_LOGS_DIR = LOGS_DIR / "yolo"
YOLO_PREDICTIONS_DIR = PREDICTIONS_DIR / "yolo"
YOLO_VISUALIZATION_DIR = VISUALIZATION_DIR / "yolo"



# =============================================================================
# Visualizations
# =============================================================================

MANIFEST_VISUALIZATION_DIR = VISUALIZATION_DIR / "manifest"

BASELINE_VISUALIZATION_DIR = VISUALIZATION_DIR / "baseline"
BASELINE_YOLO_VISUALIZATION_DIR = BASELINE_VISUALIZATION_DIR / "yolo"


# =============================================================================
# Checkpoints
# =============================================================================

PRETRAINED_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "pretrained"
TRAINED_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "trained"

YOLO_PRETRAINED_CHECKPOINT_PATH = (
    PRETRAINED_CHECKPOINTS_DIR / "yolo11n.pt"
)

YOLO_TRAINED_CHECKPOINTS_DIR = (
    TRAINED_CHECKPOINTS_DIR / "yolo"
)


# =============================================================================
# Notebooks
# =============================================================================

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# experiments
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
EXPERIMENTS_REGISTRY_PATH = EXPERIMENTS_DIR / "registry.csv"

YOLO_EXPERIMENTS_DIR = EXPERIMENTS_DIR / "yolo"
