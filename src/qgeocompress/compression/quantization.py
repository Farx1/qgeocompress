from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


def apply_fp16(model: YOLO) -> YOLO:
    """Return model configured for FP16 inference."""
    model.model.half()
    return model


def apply_int8_ptq(
    weights: str | Path,
    calibration_images: list[str] | None = None,
) -> tuple[YOLO, dict[str, Any]]:
    """Post-training INT8 quantization (best-effort via dynamic quant on CPU)."""
    model = YOLO(str(weights))
    pytorch_model = copy.deepcopy(model.model)
    pytorch_model.cpu().eval()

    try:
        quantized = torch.quantization.quantize_dynamic(
            pytorch_model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8,
        )
        model.model = quantized
        meta = {"method": "int8-ptq", "backend": "torch_dynamic", "calibration_images": len(calibration_images or [])}
    except Exception as exc:
        meta = {"method": "int8-ptq", "status": "fallback_fp32", "error": str(exc)}
    return model, meta


def compression_stats(model: YOLO, path: Path | None = None) -> dict[str, Any]:
    pytorch_model = model.model
    n_params = sum(p.numel() for p in pytorch_model.parameters())
    n_bytes = sum(p.numel() * p.element_size() for p in pytorch_model.parameters())
    stats = {
        "num_parameters": n_params,
        "param_memory_mb": round(n_bytes / (1024 * 1024), 3),
    }
    if path and Path(path).exists():
        stats["file_size_mb"] = round(Path(path).stat().st_size / (1024 * 1024), 3)
    return stats
