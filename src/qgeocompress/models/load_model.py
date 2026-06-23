from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from qgeocompress.utils.config import load_model_config


def load_yolo(weights: str | Path, task: str = "obb") -> YOLO:
    """Load an Ultralytics YOLO model."""
    return YOLO(str(weights), task=task)


def load_from_config(model_name: str) -> YOLO:
    cfg = load_model_config(model_name)
    weights = cfg.get("weights", "yolo11n-obb.pt")
    return load_yolo(weights)


def model_size_mb(path: str | Path) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    return p.stat().st_size / (1024 * 1024)


def count_parameters(model: Any) -> int:
    pytorch_model = model.model if hasattr(model, "model") else model
    return sum(p.numel() for p in pytorch_model.parameters())


def extract_detection_metrics(ultralytics_results: Any) -> dict[str, float]:
    """Extract mAP metrics from Ultralytics validation results."""
    metrics: dict[str, float] = {}
    box = getattr(ultralytics_results, "box", None)
    if box is not None:
        metrics["map50"] = float(getattr(box, "map50", 0.0) or 0.0)
        metrics["map50_95"] = float(getattr(box, "map", 0.0) or 0.0)
        metrics["precision"] = float(getattr(box, "mp", 0.0) or 0.0)
        metrics["recall"] = float(getattr(box, "mr", 0.0) or 0.0)
    results_dict = getattr(ultralytics_results, "results_dict", None)
    if results_dict:
        for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"):
            if key in results_dict:
                short = key.split("/")[-1].replace("(B)", "").lower()
                metrics[short.replace("-", "_")] = float(results_dict[key])
    return metrics
