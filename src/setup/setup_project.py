from src.dataset.pipeline.status import ensure_dataset_prepared
from src.utils.checkpoints import ensure_yolo_checkpoint


def setup_project(verbose=True):
    if verbose:
        print("Project setup started")
        print()

    dataset_status = ensure_dataset_prepared(verbose=verbose)

    checkpoint_path = ensure_yolo_checkpoint()

    if verbose:
        print(f"YOLO pretrained checkpoint ready: {checkpoint_path}")
        print()
        print("Project setup completed successfully")

    return {
        "dataset": dataset_status,
        "yolo_checkpoint": checkpoint_path,
    }


def main():
    setup_project(verbose=True)


if __name__ == "__main__":
    main()
