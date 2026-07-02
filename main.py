import argparse
import sys


MODEL_NAMES = ["detr", "faster_rcnn", "retinanet", "ssd", "yolo"]


def train_model(model_name):
    if model_name == "yolo":
        from src.training.train_yolo import train_yolo

        return train_yolo()

    if model_name == "faster_rcnn":
        from src.training.train_faster_rcnn import train_faster_rcnn

        return train_faster_rcnn()

    if model_name == "detr":
        from src.training.train_detr import train_detr

        return train_detr()

    raise NotImplementedError(
        f"Training for model '{model_name}' is not implemented yet"
    )


def evaluate_model(model_name):
    if model_name == "yolo":
        from src.evaluation.yolo import evaluate_yolo

        return evaluate_yolo()

    if model_name == "faster_rcnn":
        from src.evaluation.faster_rcnn import evaluate_faster_rcnn

        return evaluate_faster_rcnn()

    if model_name == "detr":
        from src.evaluation.detr import evaluate_detr

        return evaluate_detr()

    raise NotImplementedError(
        f"Evaluation for model '{model_name}' is not implemented yet"
    )


def compare_models():
    raise NotImplementedError("Model comparison is not implemented yet")


def clear_generated_data(keep_pretrained=True):
    from src.utils.cleanup import clear_generated_data as run_clear

    return run_clear(keep_pretrained=keep_pretrained)


def main():
    parser = argparse.ArgumentParser(
        description="Vehicle detection on KITTI"
    )

    subparsers = parser.add_subparsers(dest="command")

    setup_parser = subparsers.add_parser(
        "setup",
        help="Prepare project directories",
    )
    setup_parser.set_defaults(handler=lambda args: run_setup())

    prepare_parser = subparsers.add_parser(
        "prepare-dataset",
        help="Prepare KITTI dataset",
    )
    prepare_parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild processed dataset even if it is already prepared",
    )
    prepare_parser.set_defaults(handler=lambda args: run_prepare_dataset(args.force))

    train_parser = subparsers.add_parser(
        "train",
        help="Train selected model",
    )
    train_parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    train_parser.set_defaults(handler=lambda args: train_model(args.model))

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate selected model",
    )
    evaluate_parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    evaluate_parser.set_defaults(handler=lambda args: evaluate_model(args.model))

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare completed model experiments",
    )
    compare_parser.set_defaults(handler=lambda args: compare_models())

    clear_parser = subparsers.add_parser(
        "clear",
        help="Remove generated data and results except data/raw",
    )
    clear_parser.add_argument(
        "--remove-pretrained",
        action="store_true",
        help="Also remove pretrained checkpoints from results/checkpoints/pretrained",
    )
    clear_parser.set_defaults(
        handler=lambda args: clear_generated_data(
            keep_pretrained=not args.remove_pretrained,
        )
    )

    args = parser.parse_args()

    if not hasattr(args, "handler"):
        parser.print_help()
        return

    try:
        args.handler(args)
    except NotImplementedError as error:
        print(error)
        sys.exit(1)


def run_setup():
    from src.setup.setup_project import setup_project

    setup_project()


def run_prepare_dataset(force=False):
    from src.dataset.pipeline.prepare_dataset import prepare_dataset

    prepare_dataset(force=force)


if __name__ == "__main__":
    main()
