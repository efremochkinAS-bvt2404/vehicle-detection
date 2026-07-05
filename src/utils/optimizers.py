import torch


def build_optimizer(parameters, name="AdamW", lr=0.001, weight_decay=0.0, momentum=0.9):
    optimizer_name = (name or "AdamW").lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    supported = "AdamW, Adam, SGD"
    raise ValueError(f"Unsupported optimizer '{name}'. Supported: {supported}")
