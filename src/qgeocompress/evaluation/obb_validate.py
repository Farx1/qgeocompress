from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from ultralytics import YOLO


@contextmanager
def no_fuse() -> Iterator[None]:
    """Patch Ultralytics to skip Conv+BN fusion (required for low-rank blocks)."""
    from ultralytics.nn.backends.pytorch import PyTorchBackend

    original_load = PyTorchBackend.load_model

    def _load_without_fuse(self, weight: str | torch.nn.Module) -> None:
        from ultralytics.nn.tasks import load_checkpoint

        if isinstance(weight, torch.nn.Module):
            pytorch_model = weight.to(self.device)
        else:
            pytorch_model, _ = load_checkpoint(weight, device=self.device, fuse=False)

        if hasattr(pytorch_model, "kpt_shape"):
            self.kpt_shape = pytorch_model.kpt_shape
        self.stride = (
            max(int(pytorch_model.stride.max()), 32) if hasattr(pytorch_model, "stride") else 32
        )
        self.names = (
            pytorch_model.module.names
            if hasattr(pytorch_model, "module")
            else getattr(pytorch_model, "names", {})
        )
        self.channels = pytorch_model.yaml.get("channels", 3) if hasattr(pytorch_model, "yaml") else 3
        pytorch_model.half() if self.fp16 else pytorch_model.float()

        for p in pytorch_model.parameters():
            p.requires_grad = False
        self.model = pytorch_model

    PyTorchBackend.load_model = _load_without_fuse
    try:
        yield
    finally:
        PyTorchBackend.load_model = original_load


def validate_no_fuse(
    model: YOLO,
    data_yaml: str,
    imgsz: int = 640,
    device: str | None = None,
    half: bool = False,
) -> Any:
    """Run Ultralytics validation without Conv+BN fusion."""
    with no_fuse():
        return model.val(data=data_yaml, imgsz=imgsz, device=device, half=half, verbose=False)


def predict_no_fuse(model: YOLO, **kwargs: Any) -> Any:
    """Run Ultralytics predict without Conv+BN fusion."""
    with no_fuse():
        return model.predict(**kwargs)
