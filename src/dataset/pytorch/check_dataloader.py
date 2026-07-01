import torch

from src.dataset.pytorch.dataloaders import create_detection_dataloaders


def validate_batch(images, targets, split_name):
    if not isinstance(images, list):
        raise TypeError(f"{split_name}: images must be list, got {type(images)}")

    if not isinstance(targets, list):
        raise TypeError(f"{split_name}: targets must be list, got {type(targets)}")

    if len(images) != len(targets):
        raise ValueError(
            f"{split_name}: images and targets count mismatch: "
            f"{len(images)} != {len(targets)}"
        )

    for index, (image, target) in enumerate(zip(images, targets)):
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"{split_name}, sample {index}: image must be torch.Tensor")

        if image.ndim != 3:
            raise ValueError(f"{split_name}, sample {index}: image must have 3 dimensions")

        if image.shape[0] != 3:
            raise ValueError(f"{split_name}, sample {index}: image must have 3 channels")

        required_keys = {"boxes", "labels", "image_id", "area", "iscrowd"}
        missing_keys = required_keys - target.keys()

        if missing_keys:
            raise KeyError(f"{split_name}, sample {index}: missing keys: {missing_keys}")

        boxes = target["boxes"]
        labels = target["labels"]
        area = target["area"]
        iscrowd = target["iscrowd"]

        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(
                f"{split_name}, sample {index}: boxes must have shape [N, 4], "
                f"got {tuple(boxes.shape)}"
            )

        objects_count = boxes.shape[0]

        if labels.shape[0] != objects_count:
            raise ValueError(f"{split_name}, sample {index}: labels count mismatch")

        if area.shape[0] != objects_count:
            raise ValueError(f"{split_name}, sample {index}: area count mismatch")

        if iscrowd.shape[0] != objects_count:
            raise ValueError(f"{split_name}, sample {index}: iscrowd count mismatch")

        if boxes.dtype != torch.float32:
            raise TypeError(f"{split_name}, sample {index}: boxes dtype must be float32")

        if labels.dtype != torch.int64:
            raise TypeError(f"{split_name}, sample {index}: labels dtype must be int64")

        if area.dtype != torch.float32:
            raise TypeError(f"{split_name}, sample {index}: area dtype must be float32")

        if iscrowd.dtype != torch.int64:
            raise TypeError(f"{split_name}, sample {index}: iscrowd dtype must be int64")

        xmin = boxes[:, 0]
        ymin = boxes[:, 1]
        xmax = boxes[:, 2]
        ymax = boxes[:, 3]

        if not torch.all(xmin < xmax):
            raise ValueError(f"{split_name}, sample {index}: found boxes with xmin >= xmax")

        if not torch.all(ymin < ymax):
            raise ValueError(f"{split_name}, sample {index}: found boxes with ymin >= ymax")


def check_loader(loader, split_name, verbose=False):
    images, targets = next(iter(loader))
    validate_batch(images, targets, split_name)

    objects_count = sum(len(target["boxes"]) for target in targets)

    if verbose:
        print(f"{split_name}: OK")
        print(f"  Batch images: {len(images)}")
        print(f"  Batch objects: {objects_count}")
        print(f"  First image shape: {tuple(images[0].shape)}")
        print()


def check_dataloaders(verbose=False):
    train_loader, val_loader, test_loader = create_detection_dataloaders(
        batch_size=2,
        num_workers=0,
    )

    check_loader(train_loader, "train", verbose)
    check_loader(val_loader, "val", verbose)
    check_loader(test_loader, "test", verbose)

    if verbose:
        print("PyTorch DataLoader check completed successfully")


def main():
    check_dataloaders(verbose=True)


if __name__ == "__main__":
    main()