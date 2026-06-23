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
from qgeocompress.models.load_model import model_size_mb


def _sample_images(dataset: str, max_images: int = 32) -> list[str]:
    data_yaml = get_data_yaml(dataset)
    info = check_det_dataset(data_yaml)
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
) -> list[dict[str, Any]]:
    """Benchmark latency, throughput, and VRAM for multiple batch sizes."""
    batch_sizes = batch_sizes or [1, 4, 8]
    model = YOLO(str(weights))
    images = _sample_images(dataset, max_images=max(64, max(batch_sizes) * repeats))
    if not images:
        raise RuntimeError(f"No validation images found for dataset '{dataset}'")

    results: list[dict[str, Any]] = []
    for bs in batch_sizes:
        latencies_ms: list[float] = []
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        for i in range(warmup + repeats):
            batch = images[i % len(images) : (i % len(images)) + bs]
            if len(batch) < bs:
                batch = (batch * ((bs // len(batch)) + 1))[:bs]
            start = time.perf_counter()
            model.predict(batch, imgsz=imgsz, device=device, verbose=False)
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
            "latency_ms_p50": round(float(np.percentile(latencies_ms, 50)), 3) if latencies_ms else 0.0,
            "latency_ms_p95": round(float(np.percentile(latencies_ms, 95)), 3) if latencies_ms else 0.0,
            "throughput_img_s": round(1000.0 / mean_lat, 2) if mean_lat > 0 else 0.0,
            "vram_mb_max": round(vram_mb, 1),
            "model_size_mb": round(model_size_mb(weights), 2),
            "device": device or ("cuda" if torch.cuda.is_available() else "cpu"),
        })
    return results
