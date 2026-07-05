from src.dataset.pipeline.status import (
    get_dataset_status,
    save_preparation_status,
)


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


def prepare_dataset(force=False):
    if not force:
        status = get_dataset_status()

        if status["prepared"]:
            from src.dataset.visualization.visualize_splits import (
                visualize_split_statistics,
            )

            run_step(
                "Creating dataset split visualizations",
                visualize_split_statistics,
            )
            marker_path = save_preparation_status(status)
            print("Prepared dataset found. Skipping dataset preparation.")
            print(f"Preparation marker: {marker_path}")
            return status

    print("Dataset preparation started")
    print()

    from src.dataset.analysis.analyze_kitti import analyze_kitti
    from src.dataset.preprocessing.filter_kitti import filter_kitti
    from src.dataset.validation.validate_filtered_kitti import validate_filtered_kitti
    from src.dataset.converters.create_manifest import create_manifests
    from src.dataset.visualization.visualize_splits import visualize_split_statistics
    from src.dataset.visualization.visualize_manifest import visualize_manifest
    from src.dataset.converters.manifest_to_yolo import convert_manifest_to_yolo
    from src.dataset.pytorch.check_dataloader import check_dataloaders

    run_step("1. Analyzing raw KITTI dataset", analyze_kitti)
    run_step("2. Filtering KITTI dataset", filter_kitti)
    run_step("3. Validating filtered KITTI dataset", validate_filtered_kitti)
    run_step("4. Creating detection manifests", create_manifests)
    run_step("5. Creating dataset split visualizations", visualize_split_statistics)
    run_step("6. Creating manifest visualizations", visualize_manifest)
    run_step("7. Converting manifest to YOLO format", convert_manifest_to_yolo)
    run_step("8. Checking PyTorch DataLoaders", check_dataloaders)

    status = get_dataset_status()
    marker_path = save_preparation_status(status)

    if not status["prepared"]:
        print("Dataset preparation finished, but validation status is not prepared.")
        for error in status["errors"]:
            print(f"  - {error}")
        raise RuntimeError("Prepared dataset validation failed")

    print("=" * 60)
    print("Dataset preparation completed successfully")
    print(f"Preparation marker: {marker_path}")
    print("=" * 60)

    return status


def main():
    prepare_dataset()


if __name__ == "__main__":
    main()
