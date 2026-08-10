from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from qgeocompress.cli.compress_model import _evaluate_model, _map50_only
from qgeocompress.cli.evaluate_calibration import evaluate_calibration
from qgeocompress.compression import compress_model
from qgeocompress.data.prepare_dota import resolve_data_yaml
from qgeocompress.utils.config import make_run_id
from qgeocompress.utils.device import resolve_device


def run_baseline_eval(
    model: Path,
    test_yaml: Path,
    *,
    dataset: str = "dota128",
    iou_threshold: float = 0.5,
    imgsz: int = 640,
    device: str | None = None,
) -> dict[str, Any]:
    calibration = evaluate_calibration(
        weights=model,
        dataset=dataset,
        data_yaml=test_yaml,
        compression="baseline",
        run_label="valohai_baseline",
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        device=device,
    )
    metrics = {
        "stage": "baseline_eval",
        "baseline_map50": calibration.get("map50"),
        "baseline_cer08": calibration.get("confident_error_rate_08"),
        "baseline_ece": calibration.get("ece"),
        "baseline_params": None,
        "baseline_model_size_mb": None,
        "weights": str(model),
        "test_yaml": str(test_yaml),
    }
    try:
        yolo = YOLO(str(model), task="obb")
        from qgeocompress.models.load_model import count_parameters, model_size_mb

        metrics["baseline_params"] = count_parameters(yolo)
        metrics["baseline_model_size_mb"] = round(model_size_mb(model), 2)
    except Exception:
        pass

    predictions = {
        "num_predictions": calibration.get("num_predictions"),
        "num_gt": calibration.get("num_gt"),
        "num_tp": calibration.get("num_tp"),
        "num_fp": calibration.get("num_fp"),
        "num_fn": calibration.get("num_fn"),
        "weights": str(model),
    }
    return {
        "metrics": metrics,
        "calibration": calibration,
        "predictions": predictions,
    }


def run_structural_probe(
    model: Path,
    select_yaml: Path,
    *,
    dataset: str = "dota128",
    rank_ratio: float = 0.84,
    target_layers: list[str] | None = None,
    imgsz: int = 640,
    device: str | None = None,
    sensitivity_out: Path | None = None,
) -> dict[str, Any]:
    device = resolve_device(device)
    target_layers = target_layers or ["backbone", "neck"]
    probe_yaml = resolve_data_yaml(dataset=dataset, data_yaml=select_yaml)

    def _probe_eval(m: YOLO) -> float:
        return _map50_only(m, dataset, imgsz, device, data_yaml=probe_yaml)

    meta, _ = compress_model(
        model,
        "structural-probe",
        Path("runs/compressed/structural_probe"),
        rank_ratio=rank_ratio,
        target_layers=target_layers,
        dataset=dataset,
        data_yaml=str(probe_yaml),
        imgsz=imgsz,
        device=device,
        eval_map50_fn=_probe_eval,
        run_label="valohai_probe",
        sensitivity_report_path=str(sensitivity_out) if sensitivity_out else None,
    )
    report_path = Path(meta["sensitivity_report"])
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    layers = report.get("layers") or []
    best = max(layers, key=lambda r: r.get("param_gain_abs") or (r.get("params_before", 0) - r.get("params_after", 0)))
    summary = {
        "stage": "structural_probe",
        "candidate_layers": report.get("num_candidates", len(layers)),
        "best_layer": best.get("layer_name"),
        "best_layer_map50_drop": best.get("map50_drop"),
        "baseline_map50": report.get("baseline_map50"),
        "sensitivity_report": str(report_path),
    }
    return {
        "summary": summary,
        "sensitivity_report_path": report_path,
        "report": report,
    }


def run_compress_selective(
    model: Path,
    sensitivity_json: Path,
    train_yaml: Path,
    test_yaml: Path,
    *,
    dataset: str = "dota128",
    rank_ratio: float = 0.84,
    max_replaced_layers: int = 5,
    selection_strategy: str = "pareto-gain",
    bn_recalibration_batches: int = 20,
    max_map50_drop: float = 0.03,
    target_layers: list[str] | None = None,
    imgsz: int = 640,
    device: str | None = None,
    run_label: str = "valohai_compress",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    device = resolve_device(device)
    target_layers = target_layers or ["backbone", "neck"]
    out_subdir = output_dir or Path("runs/compressed/structural_low_rank")

    meta, in_memory_model = compress_model(
        model,
        "structural-low-rank",
        out_subdir,
        rank_ratio=rank_ratio,
        target_layers=target_layers,
        selection_strategy=selection_strategy,
        max_replaced_layers=max_replaced_layers,
        bn_recalibration_batches=bn_recalibration_batches,
        max_map50_drop=max_map50_drop,
        dataset=dataset,
        bn_data_yaml=str(train_yaml),
        sensitivity_file=str(sensitivity_json),
        run_label=run_label,
        imgsz=imgsz,
        device=device,
    )

    det_metrics: dict[str, Any] = {}
    bench_row: dict[str, Any] = {}
    if meta.get("status") == "ok" and meta.get("ultralytics_compatible"):
        eval_model = in_memory_model if in_memory_model is not None else YOLO(str(meta["output_weights"]))
        det_metrics, bench_row = _evaluate_model(
            eval_model,
            Path(meta["output_weights"]),
            dataset,
            imgsz,
            device,
            data_yaml=test_yaml,
        )

    compression_summary = {
        "run_id": make_run_id("structural-low-rank", {
            "rank_ratio": rank_ratio,
            "selection_strategy": selection_strategy,
            "max_replaced_layers": max_replaced_layers,
            "run_label": run_label,
        }),
        "stage": "compress_selective",
        "method": "structural-low-rank",
        "rank_ratio": rank_ratio,
        "selection_strategy": selection_strategy,
        "max_replaced_layers": max_replaced_layers,
        "bn_recalibration_batches": bn_recalibration_batches,
        "train_yaml": str(train_yaml),
        "test_yaml": str(test_yaml),
        **det_metrics,
        **bench_row,
        **meta,
    }

    selected_layers = {
        "selected_layer_names": meta.get("selected_layer_names") or meta.get("replaced_layer_names"),
        "num_replaced_layers": meta.get("num_replaced_layers"),
        "real_param_reduction_pct": meta.get("real_param_reduction_pct"),
    }

    stdout_metrics = {
        "stage": "compress_selective",
        "method": "structural-low-rank",
        "rank_ratio": rank_ratio,
        "max_replaced_layers": max_replaced_layers,
        "real_param_reduction_pct": meta.get("real_param_reduction_pct"),
        "compressed_model_size_mb": meta.get("file_size_mb") or meta.get("model_size_mb"),
        "map50": det_metrics.get("map50"),
        "status": meta.get("status"),
    }
    return {
        "stdout_metrics": stdout_metrics,
        "compression_summary": compression_summary,
        "selected_layers": selected_layers,
        "output_weights": Path(meta["output_weights"]) if meta.get("output_weights") else None,
    }
