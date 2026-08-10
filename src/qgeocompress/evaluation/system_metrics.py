from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

from qgeocompress.data.prepare_dota import get_data_yaml
from qgeocompress.evaluation.gt_matching import list_val_images
from qgeocompress.evaluation.obb_validate import predict_no_fuse
from qgeocompress.models.load_model import model_size_mb


def _has_structural_layers(model: YOLO) -> bool:
    from qgeocompress.compression.structural_low_rank import has_factorized_layers

    return has_factorized_layers(model)


def _sample_images(
    dataset: str,
    max_images: int = 32,
    data_yaml: str | Path | None = None,
) -> list[str]:
    if data_yaml is not None:
        return [str(p) for _, p in list_val_images(dataset=dataset, data_yaml=data_yaml)[:max_images]]

    yaml_path = get_data_yaml(dataset)
    info = check_det_dataset(yaml_path)
    val = info.get("val") or info.get("train")
    if not val:
        return []
    p = Path(val)
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    if p.is_file():
        with open(p) as f:
            return [line.strip() for line in f if line.strip()][:max_images]
    if p.is_dir():
        return [str(f) for f in sorted(p.iterdir()) if f.suffix.lower() in exts][:max_images]
    return []


def benchmark_inference(
    weights: str | Path,
    dataset: str = "dota128",
    batch_sizes: list[int] | None = None,
    imgsz: int = 640,
    device: str | None = None,
    warmup: int = 5,
    repeats: int = 20,
    data_yaml: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Benchmark latency, throughput, and VRAM for multiple batch sizes."""
    batch_sizes = batch_sizes or [1, 4, 8]
    model = YOLO(str(weights), task="obb")
    structural = _has_structural_layers(model)
    max_needed = max(64, max(batch_sizes) * repeats)
    images = _sample_images(dataset, max_images=max_needed, data_yaml=data_yaml)
    if not images:
        raise RuntimeError(f"No validation images found for dataset '{dataset}'")

    def _predict(batch: list[str]) -> None:
        kwargs: dict[str, Any] = {"imgsz": imgsz, "device": device, "verbose": False}
        if structural:
            predict_no_fuse(model, source=batch, **kwargs)
        else:
            model.predict(batch, **kwargs)

    results: list[dict[str, Any]] = []
    for bs in batch_sizes:
        latencies_ms: list[float] = []
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        for i in range(warmup + repeats):
            batch = images[i % len(images) : (i % len(images)) + bs]
            if len(batch) < bs:
                batch = (batch * ((bs // len(batch)) + 1))[:bs]
            start = time.perf_counter()
            _predict(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            if i >= warmup:
                latencies_ms.append(elapsed_ms / bs)

        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        mean_lat = statistics.mean(latencies_ms) if latencies_ms else 0.0
        results.append({
            "batch_size": bs,
            "latency_ms_mean": round(mean_lat, 3),
            "latency_ms_sd": round(statistics.stdev(latencies_ms), 3) if len(latencies_ms) > 1 else 0.0,
            "latency_ms_p50": round(float(np.percentile(latencies_ms, 50)), 3) if latencies_ms else 0.0,
            "latency_ms_p95": round(float(np.percentile(latencies_ms, 95)), 3) if latencies_ms else 0.0,
            "throughput_img_s": round(1000.0 / mean_lat, 2) if mean_lat > 0 else 0.0,
            "vram_mb_max": round(vram_mb, 1),
            "model_size_mb": round(model_size_mb(weights), 2),
            "device": device or ("cuda" if torch.cuda.is_available() else "cpu"),
            "structural_no_fuse": structural,
        })
    return results
