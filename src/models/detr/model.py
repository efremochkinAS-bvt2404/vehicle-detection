import torch


def _import_transformers():
    try:
        from transformers import DetrForObjectDetection, DetrImageProcessor
    except ImportError as error:
        raise ImportError(
            "DETR requires transformers and timm. Install them with: "
            ".\\.venv\\Scripts\\python.exe -m pip install transformers timm"
        ) from error

    return DetrForObjectDetection, DetrImageProcessor


def build_detr_model(model_name, num_labels, pretrained=True):
    DetrForObjectDetection, _ = _import_transformers()

    id2label = {index: str(index) for index in range(num_labels)}
    label2id = {label: index for index, label in id2label.items()}

    if pretrained:
        return DetrForObjectDetection.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    return DetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )


def build_detr_processor(model_name):
    _, DetrImageProcessor = _import_transformers()
    return DetrImageProcessor.from_pretrained(model_name)


def save_checkpoint(path, model, optimizer, epoch, metrics, model_name):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_model_from_checkpoint(checkpoint_path, model_name, num_labels, device):
    model = build_detr_model(
        model_name=model_name,
        num_labels=num_labels,
        pretrained=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint
