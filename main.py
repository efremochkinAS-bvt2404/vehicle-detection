import argparse

from src.dataset.pipeline.prepare_dataset import prepare_dataset
from src.setup.setup_project import setup_project
from src.training.train_yolo import train_yolo


def main():
    parser = argparse.ArgumentParser(
        description="Emergency Vehicle Detection"
    )

    parser.add_argument(
        "--setup",
        action="store_true",
        help="Prepare project for training",
    )

    parser.add_argument(
        "--prepare-dataset",
        action="store_true",
        help="Prepare dataset",
    )

    parser.add_argument(
        "--train",
        choices=["yolo"],
        help="Train selected model",
    )

    args = parser.parse_args()

    if args.setup:
        setup_project()
        return

    if args.prepare_dataset:
        prepare_dataset()
        return

    if args.train == "yolo":
        train_yolo()
        return

    parser.print_help()


if __name__ == "__main__":
    main()