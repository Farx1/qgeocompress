from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def export_torchscript(
    weights: str | Path,
    output_dir: str | Path | None = None,
    imgsz: int = 640,
    device: str = "cpu",
) -> Path:
    """Export YOLO weights to TorchScript via Ultralytics."""
    model = YOLO(str(weights))
    export_path = model.export(format="torchscript", imgsz=imgsz, device=device)
    path = Path(export_path)
    if output_dir is not None:
        out = Path(output_dir) / path.name
        out.parent.mkdir(parents=True, exist_ok=True)
        path.replace(out)
        return out
    return path
