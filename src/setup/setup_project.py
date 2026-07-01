from src.dataset.pipeline.prepare_dataset import prepare_dataset
from src.utils.checkpoints import ensure_yolo_checkpoint
from src.utils.paths import YOLO_DATA_YAML_PATH


def setup_project(verbose=True):
    if verbose:
        print("Project setup started")
        print()

    if not YOLO_DATA_YAML_PATH.exists():
        if verbose:
            print("YOLO dataset not found")
            print("Preparing dataset...")
            print()

        prepare_dataset()
    else:
        if verbose:
            print("YOLO dataset found")

    checkpoint_path = ensure_yolo_checkpoint()

    if verbose:
        print(f"YOLO pretrained checkpoint ready: {checkpoint_path}")
        print()
        print("Project setup completed successfully")

    return {
        "yolo_data_yaml": YOLO_DATA_YAML_PATH,
        "yolo_checkpoint": checkpoint_path,
    }


def main():
    setup_project(verbose=True)


if __name__ == "__main__":
    main()