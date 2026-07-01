from src.dataset.preprocessing.filter_kitti import filter_kitti
from src.dataset.validation.validate_filtered_kitti import validate_filtered_kitti
from src.dataset.converters.create_manifest import create_manifests
from src.dataset.visualization.visualize_manifest import visualize_manifest
from src.dataset.converters.manifest_to_yolo import convert_manifest_to_yolo
from src.dataset.pytorch.check_dataloader import check_dataloaders


def run_step(title, function):
    print("=" * 60)
    print(title)
    print("=" * 60)

    try:
        result = function()
    except Exception as error:
        print(f"{title}: FAILED")
        print(error)
        raise

    print(f"{title}: OK")
    print()

    return result


def prepare_dataset():
    print("Dataset preparation started")
    print()

    run_step("1. Filtering KITTI dataset", filter_kitti)
    run_step("2. Validating filtered KITTI dataset", validate_filtered_kitti)
    run_step("3. Creating detection manifests", create_manifests)
    run_step("4. Creating manifest visualizations", visualize_manifest)
    run_step("5. Converting manifest to YOLO format", convert_manifest_to_yolo)
    run_step("6. Checking PyTorch DataLoaders", check_dataloaders)

    print("=" * 60)
    print("Dataset preparation completed successfully")
    print("=" * 60)


def main():
    prepare_dataset()


if __name__ == "__main__":
    main()