# Dataset Analysis

This document summarizes the exploratory dataset analysis used in the project.
The project uses scripts instead of an interactive notebook for reproducibility,
while this file records the main findings and points to generated artifacts.

## Dataset

The project uses the KITTI object detection dataset. The original annotations
contain 9 classes:

- Car
- Van
- Truck
- Pedestrian
- Person_sitting
- Cyclist
- Tram
- Misc
- DontCare

The project focuses on vehicle detection, therefore only transport-related
classes are kept:

- Car
- Van
- Truck
- Cyclist
- Tram

## Raw Dataset Statistics

Raw KITTI analysis is saved in:

```text
results/metrics/kitti/analysis.json
```

Main values:

| Metric | Value |
|---|---:|
| Label files | 7481 |
| Images with objects | 7481 |
| Empty label files | 0 |
| Total objects | 51865 |
| Number of classes | 9 |

Raw class distribution:

```text
results/plots/kitti/class_distribution.png
```

## Filtering

Filtering removes objects outside the selected vehicle classes and skips images
without target objects.

Filtering summary is saved in:

```text
results/metrics/filtered_kitti/summary.json
```

Main values:

| Metric | Value |
|---|---:|
| Source label files | 7481 |
| Copied images | 6942 |
| Images skipped without target classes | 539 |
| Missing images | 0 |
| Invalid annotation lines | 0 |
| Raw objects | 51865 |
| Filtered objects | 34888 |
| Removed objects | 16977 |

Filtered class distribution:

```text
results/plots/filtered_kitti/class_distribution.png
```

Filtered class counts:

| Class | Objects |
|---|---:|
| Car | 28742 |
| Van | 2914 |
| Cyclist | 1627 |
| Truck | 1094 |
| Tram | 511 |

## Validation

The prepared filtered dataset was validated after preprocessing.

Validation report:

```text
results/metrics/filtered_kitti/validation.json
```

Validation summary:

| Metric | Value |
|---|---:|
| Images | 6942 |
| Labels | 6942 |
| Errors | 0 |
| Warnings | 0 |

## Preprocessing

The preprocessing pipeline includes:

- parsing KITTI annotations;
- filtering object classes;
- converting annotations to a unified project format;
- splitting data into train, validation and test sets;
- adapting labels for model-specific formats;
- resizing images according to model configuration;
- normalization in model-specific transforms;
- data augmentation during training.

Model-specific adapters are used where needed:

- YOLO uses converted YOLO annotations;
- DETR uses a dedicated DETR dataset adapter;
- Faster R-CNN, RetinaNet and SSD use PyTorch detection datasets with
  model-specific label handling.

## Augmentation

The configured training augmentations are:

- horizontal flip;
- color jitter;

Augmentation settings are stored in:

```text
configs/*.yaml
```

## Experiment Artifacts

Model training and evaluation results are stored in:

```text
results/experiments/
```

Final model comparison artifacts are stored in:

```text
results/comparison/
```

Run comparisons for a single model are stored in:

```text
results/model_runs_comparison/
```

These folders contain CSV tables, PNG tables, training curves, quality charts,
performance charts, summaries and short markdown reports.
