from __future__ import annotations

import torch


def resolve_device(device: str | None = None) -> str:
    """Resolve training/inference device string for Ultralytics."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_torch_device(device: str | None = None) -> torch.device:
    resolved = resolve_device(device)
    if resolved == "cpu":
        return torch.device("cpu")
    if resolved == "mps":
        return torch.device("mps")
    return torch.device(f"cuda:{resolved}" if resolved.isdigit() else resolved)
