from torch.utils.data import DataLoader

from src.dataset.pytorch.collate import detection_collate_fn
from src.dataset.pytorch.detection_dataset import DetectionDataset
from src.dataset.pytorch.transforms import get_detection_transforms
from src.utils.paths import (
    FILTERED_IMAGES_DIR,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
    TEST_MANIFEST_PATH,
)


def create_detection_dataloaders(
    batch_size=4,
    num_workers=2,
    shuffle_train=True,
    augmentation_config=None,
    label_offset=1,
    drop_last_train=False,
):
    train_transforms = get_detection_transforms(
        augmentation_config=augmentation_config,
        train=True,
    )
    eval_transforms = get_detection_transforms(
        augmentation_config=augmentation_config,
        train=False,
    )

    train_dataset = DetectionDataset(
        manifest_file=TRAIN_MANIFEST_PATH,
        images_dir=FILTERED_IMAGES_DIR,
        transforms=train_transforms,
        label_offset=label_offset,
    )

    val_dataset = DetectionDataset(
        manifest_file=VAL_MANIFEST_PATH,
        images_dir=FILTERED_IMAGES_DIR,
        transforms=eval_transforms,
        label_offset=label_offset,
    )

    test_dataset = DetectionDataset(
        manifest_file=TEST_MANIFEST_PATH,
        images_dir=FILTERED_IMAGES_DIR,
        transforms=eval_transforms,
        label_offset=label_offset,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        drop_last=drop_last_train,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )

    return train_loader, val_loader, test_loader
