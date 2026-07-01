from collections import defaultdict
from pathlib import Path

import torch


IOU_THRESHOLDS = [round(0.5 + index * 0.05, 2) for index in range(10)]


def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    width_height = (right_bottom - left_top).clamp(min=0)
    intersection = width_height[:, :, 0] * width_height[:, :, 1]
    union = area1[:, None] + area2 - intersection

    return intersection / union.clamp(min=1e-6)


def _prepare_predictions(predictions, targets, confidence_threshold):
    prepared_predictions = []
    prepared_targets = []

    for image_index, (prediction, target) in enumerate(zip(predictions, targets)):
        scores = prediction["scores"].detach().cpu()
        keep = scores >= confidence_threshold

        prepared_predictions.append(
            {
                "image_id": image_index,
                "boxes": prediction["boxes"].detach().cpu()[keep],
                "labels": prediction["labels"].detach().cpu()[keep],
                "scores": scores[keep],
            }
        )
        prepared_targets.append(
            {
                "image_id": image_index,
                "boxes": target["boxes"].detach().cpu(),
                "labels": target["labels"].detach().cpu(),
            }
        )

    return prepared_predictions, prepared_targets


def _match_at_threshold(predictions, targets, iou_threshold):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    confusion = defaultdict(int)

    for prediction, target in zip(predictions, targets):
        pred_boxes = prediction["boxes"]
        pred_labels = prediction["labels"]
        scores = prediction["scores"]
        target_boxes = target["boxes"]
        target_labels = target["labels"]
        matched_targets = set()

        order = torch.argsort(scores, descending=True)

        for prediction_index in order.tolist():
            label = int(pred_labels[prediction_index])
            same_label_targets = torch.where(target_labels == label)[0]
            available = [
                index
                for index in same_label_targets.tolist()
                if index not in matched_targets
            ]

            if not available:
                false_positive += 1
                confusion[(0, label)] += 1
                continue

            ious = box_iou(
                pred_boxes[prediction_index].unsqueeze(0),
                target_boxes[available],
            )[0]
            best_local_index = int(torch.argmax(ious))
            best_iou = float(ious[best_local_index])
            best_target_index = available[best_local_index]

            if best_iou >= iou_threshold:
                true_positive += 1
                matched_targets.add(best_target_index)
                confusion[(label, label)] += 1
            else:
                false_positive += 1
                confusion[(0, label)] += 1

        for target_index, label in enumerate(target_labels.tolist()):
            if target_index not in matched_targets:
                false_negative += 1
                confusion[(int(label), 0)] += 1

    return true_positive, false_positive, false_negative, confusion


def _average_precision_for_class(predictions, targets, class_id, iou_threshold):
    detections = []
    ground_truths = defaultdict(list)

    for prediction in predictions:
        keep = prediction["labels"] == class_id
        for box, score in zip(prediction["boxes"][keep], prediction["scores"][keep]):
            detections.append(
                {
                    "image_id": prediction["image_id"],
                    "box": box,
                    "score": float(score),
                }
            )

    for target in targets:
        keep = target["labels"] == class_id
        for box in target["boxes"][keep]:
            ground_truths[target["image_id"]].append({"box": box, "matched": False})

    total_ground_truths = sum(len(items) for items in ground_truths.values())

    if total_ground_truths == 0:
        return None

    detections.sort(key=lambda item: item["score"], reverse=True)
    true_positives = []
    false_positives = []

    for detection in detections:
        candidates = ground_truths[detection["image_id"]]

        if not candidates:
            true_positives.append(0.0)
            false_positives.append(1.0)
            continue

        boxes = torch.stack([candidate["box"] for candidate in candidates])
        ious = box_iou(detection["box"].unsqueeze(0), boxes)[0]
        best_index = int(torch.argmax(ious))
        best_iou = float(ious[best_index])

        if best_iou >= iou_threshold and not candidates[best_index]["matched"]:
            candidates[best_index]["matched"] = True
            true_positives.append(1.0)
            false_positives.append(0.0)
        else:
            true_positives.append(0.0)
            false_positives.append(1.0)

    if not detections:
        return 0.0

    true_positives = torch.tensor(true_positives).cumsum(0)
    false_positives = torch.tensor(false_positives).cumsum(0)
    recalls = true_positives / max(total_ground_truths, 1)
    precisions = true_positives / (true_positives + false_positives).clamp(min=1e-6)

    recall_points = torch.linspace(0, 1, 101)
    precision_at_recall = []

    for recall_point in recall_points:
        mask = recalls >= recall_point
        precision_at_recall.append(float(precisions[mask].max()) if mask.any() else 0.0)

    return sum(precision_at_recall) / len(precision_at_recall)


def evaluate_predictions(
    predictions,
    targets,
    class_ids,
    confidence_threshold=0.25,
    iou_threshold=0.5,
):
    predictions, targets = _prepare_predictions(
        predictions,
        targets,
        confidence_threshold,
    )
    tp, fp, fn, confusion = _match_at_threshold(predictions, targets, iou_threshold)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    aps_50 = [
        _average_precision_for_class(predictions, targets, class_id, 0.5)
        for class_id in class_ids
    ]
    aps_50 = [value for value in aps_50 if value is not None]

    aps_all = []
    for threshold in IOU_THRESHOLDS:
        for class_id in class_ids:
            value = _average_precision_for_class(
                predictions,
                targets,
                class_id,
                threshold,
            )
            if value is not None:
                aps_all.append(value)

    map50 = sum(aps_50) / len(aps_50) if aps_50 else 0.0
    map50_95 = sum(aps_all) / len(aps_all) if aps_all else 0.0

    return {
        "map": map50_95,
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "confusion": {f"{key[0]}->{key[1]}": value for key, value in confusion.items()},
    }


def save_detection_evaluation_plots(
    predictions,
    targets,
    class_ids,
    class_names,
    output_dir,
):
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [round(index / 20, 2) for index in range(1, 20)]
    precision_values = []
    recall_values = []
    f1_values = []

    for threshold in thresholds:
        metrics = evaluate_predictions(
            predictions,
            targets,
            class_ids,
            confidence_threshold=threshold,
            iou_threshold=0.5,
        )
        precision_values.append(metrics["precision"])
        recall_values.append(metrics["recall"])
        f1_values.append(metrics["f1"])

    curves = {
        "pr_curve.png": ("Recall", "Precision", recall_values, precision_values),
        "precision_curve.png": ("Confidence", "Precision", thresholds, precision_values),
        "recall_curve.png": ("Confidence", "Recall", thresholds, recall_values),
        "f1_curve.png": ("Confidence", "F1", thresholds, f1_values),
    }

    created = []

    for file_name, (xlabel, ylabel, x_values, y_values) in curves.items():
        plt.figure(figsize=(8, 6))
        plt.plot(x_values, y_values, marker="o")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        path = output_dir / file_name
        plt.savefig(path, dpi=300)
        plt.close()
        created.append(path)

    metrics = evaluate_predictions(predictions, targets, class_ids)
    matrix_size = len(class_ids) + 1
    matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)

    for key, value in metrics["confusion"].items():
        actual, predicted = [int(part) for part in key.split("->")]
        actual_index = actual if actual == 0 else class_ids.index(actual) + 1
        predicted_index = predicted if predicted == 0 else class_ids.index(predicted) + 1
        matrix[actual_index, predicted_index] += value

    labels = ["background"] + class_names

    for file_name, normalize in (
        ("confusion_matrix.png", False),
        ("confusion_matrix_normalized.png", True),
    ):
        plot_matrix = matrix.copy()

        if normalize:
            row_sums = plot_matrix.sum(axis=1, keepdims=True)
            plot_matrix = np.divide(
                plot_matrix,
                row_sums,
                out=np.zeros_like(plot_matrix),
                where=row_sums != 0,
            )

        plt.figure(figsize=(8, 7))
        plt.imshow(plot_matrix, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(matrix_size), labels, rotation=45, ha="right")
        plt.yticks(range(matrix_size), labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        path = output_dir / file_name
        plt.savefig(path, dpi=300)
        plt.close()
        created.append(path)

    return created
