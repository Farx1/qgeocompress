from __future__ import annotations

import resource
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from qgeocompress.cli.evaluate_calibration import evaluate_calibration
from qgeocompress.evaluation.gt_matching import list_val_images
from qgeocompress.evaluation.obb_validate import predict_no_fuse
from qgeocompress.models.load_model import count_parameters, model_size_mb
from qgeocompress.utils.device import resolve_device


def _has_structural_layers(model: YOLO) -> bool:
    from qgeocompress.compression.structural_low_rank import StructuralLowRankConv2d

    return any(isinstance(m, StructuralLowRankConv2d) for m in model.model.modules())


def benchmark_on_data_yaml(
    weights: Path,
    data_yaml: Path,
    *,
    batch_size: int = 1,
    imgsz: int = 640,
    device: str | None = None,
    warmup: int = 3,
    repeats: int = 10,
) -> dict[str, Any]:
    """Measure per-image latency and throughput on images from a data YAML."""
    device = resolve_device(device)
    model = YOLO(str(weights), task="obb")
    images = [str(p) for _, p in list_val_images(data_yaml=data_yaml)]
    if not images:
        raise RuntimeError(f"No validation images found in {data_yaml}")

    structural = _has_structural_layers(model)
    latencies_ms: list[float] = []
    if hasattr(resource, "getrusage"):
        resource.getrusage(resource.RUSAGE_SELF)  # reset peak tracking baseline

    for i in range(warmup + repeats):
        batch = images[i % len(images) : (i % len(images)) + batch_size]
        if len(batch) < batch_size:
            batch = (batch * ((batch_size // len(batch)) + 1))[:batch_size]
        start = time.perf_counter()
        if structural:
            predict_no_fuse(model, source=batch, imgsz=imgsz, device=device, verbose=False)
        else:
            model.predict(batch, imgsz=imgsz, device=device, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        if i >= warmup:
            latencies_ms.append(elapsed_ms / batch_size)

    mean_lat = statistics.mean(latencies_ms) if latencies_ms else 0.0
    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    cpu_ram_mb = 0.0
    if hasattr(resource, "getrusage"):
        # Linux: ru_maxrss in KB; macOS: bytes
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        cpu_ram_mb = rss / 1024 if rss > 10_000_000 else rss / (1024 * 1024)

    return {
        "batch_size": batch_size,
        "latency_ms_mean": round(mean_lat, 3),
        "throughput_img_s": round(1000.0 / mean_lat, 2) if mean_lat > 0 else 0.0,
        "vram_mb_max": round(vram_mb, 1),
        "cpu_ram_mb_max": round(cpu_ram_mb, 1),
        "model_size_mb": round(model_size_mb(weights), 2),
        "num_parameters": count_parameters(model),
        "structural_no_fuse": structural,
        "device": device,
    }


def _model_metrics(cal: dict[str, Any], bench: dict[str, Any]) -> dict[str, Any]:
    return {
        "map50": cal.get("map50"),
        "map50_95": cal.get("map50_95"),
        "cer08": cal.get("confident_error_rate_08"),
        "ece": cal.get("ece"),
        "latency_ms_mean": bench.get("latency_ms_mean"),
        "throughput_img_s": bench.get("throughput_img_s"),
        "params": bench.get("num_parameters"),
        "model_size_mb": bench.get("model_size_mb"),
        "vram_mb_max": bench.get("vram_mb_max"),
        "cpu_ram_mb_max": bench.get("cpu_ram_mb_max"),
        "num_tp": cal.get("num_tp"),
        "num_fp": cal.get("num_fp"),
        "num_fn": cal.get("num_fn"),
    }


def run_inference_compare(
    baseline_weights: Path,
    compressed_weights: Path,
    test_yaml: Path,
    *,
    dataset: str = "dota128",
    compression: str = "structural-low-rank",
    rank_ratio: float | None = 0.84,
    iou_threshold: float = 0.5,
    batch_size: int = 1,
    imgsz: int = 640,
    device: str | None = None,
) -> dict[str, Any]:
    """Compare baseline vs compressed model on the same test split."""
    baseline_cal = evaluate_calibration(
        weights=baseline_weights,
        dataset=dataset,
        data_yaml=test_yaml,
        compression="baseline",
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        device=device,
    )
    compressed_cal = evaluate_calibration(
        weights=compressed_weights,
        dataset=dataset,
        data_yaml=test_yaml,
        compression=compression,
        rank_ratio=rank_ratio,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        device=device,
    )

    baseline_bench = benchmark_on_data_yaml(
        baseline_weights, test_yaml, batch_size=batch_size, imgsz=imgsz, device=device,
    )
    compressed_bench = benchmark_on_data_yaml(
        compressed_weights, test_yaml, batch_size=batch_size, imgsz=imgsz, device=device,
    )

    baseline = _model_metrics(baseline_cal, baseline_bench)
    compressed = _model_metrics(compressed_cal, compressed_bench)

    map50_drop = None
    if baseline.get("map50") is not None and compressed.get("map50") is not None:
        map50_drop = round(max(0.0, float(baseline["map50"]) - float(compressed["map50"])), 6)

    cer08_delta = None
    if baseline.get("cer08") is not None and compressed.get("cer08") is not None:
        cer08_delta = round(float(compressed["cer08"]) - float(baseline["cer08"]), 6)

    param_reduction_pct = None
    if baseline.get("params") and compressed.get("params"):
        b_params = float(baseline["params"])
        c_params = float(compressed["params"])
        if b_params > 0:
            param_reduction_pct = round(100.0 * (b_params - c_params) / b_params, 2)

    latency_speedup_pct = None
    b_lat = baseline.get("latency_ms_mean")
    c_lat = compressed.get("latency_ms_mean")
    if b_lat and c_lat and float(c_lat) > 0:
        latency_speedup_pct = round(100.0 * (float(b_lat) - float(c_lat)) / float(b_lat), 2)

    delta = {
        "map50_drop": map50_drop,
        "cer08_delta": cer08_delta,
        "param_reduction_pct": param_reduction_pct,
        "latency_speedup_pct": latency_speedup_pct,
    }

    return {
        "stage": "inference_compare",
        "test_yaml": str(test_yaml),
        "baseline_weights": str(baseline_weights),
        "compressed_weights": str(compressed_weights),
        "baseline": baseline,
        "compressed": compressed,
        "delta": delta,
        "baseline_calibration": baseline_cal,
        "compressed_calibration": compressed_cal,
        "inference_latency": {
            "baseline": baseline_bench,
            "compressed": compressed_bench,
        },
        "calibration_comparison": {
            "baseline_ece": baseline.get("ece"),
            "compressed_ece": compressed.get("ece"),
            "baseline_cer08": baseline.get("cer08"),
            "compressed_cer08": compressed.get("cer08"),
            "ece_delta": (
                round(float(compressed["ece"]) - float(baseline["ece"]), 6)
                if baseline.get("ece") is not None and compressed.get("ece") is not None
                else None
            ),
        },
    }
