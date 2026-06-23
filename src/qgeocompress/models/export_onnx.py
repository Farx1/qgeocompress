from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def export_onnx(
    weights: str | Path,
    output_dir: str | Path | None = None,
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
) -> Path:
    """Export YOLO weights to ONNX."""
    model = YOLO(str(weights))
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
    )
    path = Path(export_path)
    if output_dir is not None:
        out = Path(output_dir) / path.name
        out.parent.mkdir(parents=True, exist_ok=True)
        path.replace(out)
        return out
    return path
