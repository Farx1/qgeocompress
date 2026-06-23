"""TensorRT export — requires NVIDIA GPU and optional tensorrt extra."""

from __future__ import annotations

from pathlib import Path


def export_tensorrt(
    weights: str | Path,
    imgsz: int = 640,
    half: bool = True,
) -> Path:
    """Export YOLO weights to TensorRT engine via Ultralytics."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    export_path = model.export(format="engine", imgsz=imgsz, half=half)
    return Path(export_path)
