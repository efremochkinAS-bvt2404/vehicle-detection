import shutil
from pathlib import Path

from src.utils.paths import (
    CHECKPOINTS_DIR,
    PRETRAINED_CHECKPOINTS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
)


def _resolve(path):
    return Path(path).resolve()


def _ensure_inside_project(path):
    resolved_path = _resolve(path)
    resolved_root = _resolve(PROJECT_ROOT)

    if resolved_path == resolved_root:
        raise ValueError("Refusing to remove project root")

    if resolved_root not in resolved_path.parents:
        raise ValueError(f"Refusing to remove path outside project: {resolved_path}")

    return resolved_path


def _remove_path(path):
    path = Path(path)

    if not path.exists():
        return False

    _ensure_inside_project(path)

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()

    return True


def _clear_results(keep_pretrained=True):
    removed = []

    if not RESULTS_DIR.exists():
        return removed

    if keep_pretrained:
        for path in RESULTS_DIR.iterdir():
            if path == CHECKPOINTS_DIR:
                continue

            if _remove_path(path):
                removed.append(path)

        if CHECKPOINTS_DIR.exists():
            for path in CHECKPOINTS_DIR.iterdir():
                if path == PRETRAINED_CHECKPOINTS_DIR:
                    continue

                if _remove_path(path):
                    removed.append(path)
    else:
        if _remove_path(RESULTS_DIR):
            removed.append(RESULTS_DIR)

    return removed


def clear_generated_data(keep_pretrained=True, verbose=True):
    removed = []

    if _remove_path(PROCESSED_DATA_DIR):
        removed.append(PROCESSED_DATA_DIR)

    removed.extend(_clear_results(keep_pretrained=keep_pretrained))

    if verbose:
        print("Generated data cleanup completed")
        print()
        print("Removed:")

        if removed:
            for path in removed:
                print(f"  - {path}")
        else:
            print("  Nothing to remove")

        print()
        print("Kept:")
        print(f"  - {PROJECT_ROOT / 'data' / 'raw'}")

        if keep_pretrained:
            print(f"  - {PRETRAINED_CHECKPOINTS_DIR}")

    return removed
