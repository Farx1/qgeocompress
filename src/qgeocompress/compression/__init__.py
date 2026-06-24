from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from qgeocompress.compression.layer_sensitivity import (
    load_sensitivity_report,
    recalibrate_batchnorm,
    run_structural_probe,
    save_sensitivity_report,
)
from qgeocompress.compression.low_rank import apply_low_rank, count_params
from qgeocompress.compression.pruning import magnitude_prune
from qgeocompress.compression.quantization import compression_stats
from qgeocompress.compression.structural_low_rank import apply_structural_low_rank


def _base_meta(method: str, weights: str | Path) -> dict[str, Any]:
    return {
        "method": method,
        "source_weights": str(weights),
        "status": "ok",
        "ultralytics_compatible": False,
        "output_format": None,
        "output_weights": None,
        "reason": None,
    }


def compress_model(
    weights: str | Path,
    method: str,
    output_dir: str | Path,
    **kwargs: Any,
) -> tuple[dict[str, Any], YOLO | None]:
    """Apply a compression method and return metadata plus optional in-memory model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if method == "structural-probe":
        meta = _base_meta(method, weights)
        rank_ratio = float(kwargs.get("rank_ratio", 0.84))
        target_layers = kwargs.get("target_layers", ["backbone", "neck"])
        dataset = kwargs.get("dataset", "dota128")
        imgsz = int(kwargs.get("imgsz", 640))
        device = kwargs.get("device", "cpu")
        eval_map50_fn = kwargs.get("eval_map50_fn")
        max_probe_layers = kwargs.get("max_probe_layers")
        data_yaml = kwargs.get("data_yaml")
        run_label = kwargs.get("run_label")
        sensitivity_path = kwargs.get("sensitivity_report_path")
        report = run_structural_probe(
            weights,
            dataset=kwargs.get("dataset", "dota128"),
            rank_ratio=rank_ratio,
            target_layers=target_layers,
            imgsz=imgsz,
            device=device,
            eval_map50_fn=eval_map50_fn,
            max_probe_layers=max_probe_layers,
            data_yaml=data_yaml,
        )
        out_path = save_sensitivity_report(
            report,
            path=Path(sensitivity_path) if sensitivity_path else None,
            run_label=run_label,
        )
        meta.update({
            "status": "ok",
            "ultralytics_compatible": False,
            "compression_mode": "structural_probe",
            "sensitivity_report": str(out_path),
            "num_candidates": report.get("num_candidates"),
            "baseline_map50": report.get("baseline_map50"),
            "top_layers": report.get("top_layers"),
        })
        return meta, None

    model = YOLO(str(weights))
    meta = _base_meta(method, weights)

    if method == "fp16":
        if not torch.cuda.is_available():
            meta.update({
                "status": "skipped",
                "reason": "FP16 meaningful benchmark requires CUDA",
                "output_format": "runtime_half",
            })
            return meta, None

        meta.update({
            "status": "ok",
            "ultralytics_compatible": False,
            "output_format": "runtime_half",
            "output_weights": str(weights),
            "reason": "Evaluated with Ultralytics half=True on source checkpoint",
        })
        return meta, model

    if method == "int8-ptq":
        meta.update({
            "status": "unsupported_for_ultralytics_eval",
            "ultralytics_compatible": False,
            "output_format": "torch_dynamic_quant_stub",
            "reason": (
                "INT8 dynamic PTQ is not compatible with YOLO() checkpoint reload; "
                "use TensorRT / ONNX Runtime / OpenVINO for deployment INT8"
            ),
        })
        return meta, None

    if method == "magnitude-pruning":
        sparsity = float(kwargs.get("sparsity", 0.2))
        model, meta_prune = magnitude_prune(model, sparsity=sparsity)
        meta.update(meta_prune)
        out_path = output_dir / f"pruned_s{int(sparsity * 100)}.pt"
        model.save(str(out_path))
        meta.update({
            "status": "ok",
            "ultralytics_compatible": True,
            "output_format": "ultralytics_pt",
            "output_weights": str(out_path),
        })
        meta.update(compression_stats(model, out_path))
        return meta, model

    if method == "low-rank":
        rank_ratio = float(kwargs.get("rank_ratio", 0.5))
        target_layers = kwargs.get("target_layers", ["backbone", "neck"])
        model, meta_lr = apply_low_rank(model, rank_ratio=rank_ratio, target_layers=target_layers)
        meta.update(meta_lr)

        if meta_lr["num_replaced_layers"] == 0:
            meta.update({
                "status": "failed_no_layers_replaced",
                "ultralytics_compatible": False,
                "reason": "No compressible layers matched target_layers; check layer naming",
            })
            return meta, None

        out_path = output_dir / f"low_rank_r{rank_ratio:.3f}.pt"
        model.save(str(out_path))
        meta.update({
            "status": "ok",
            "ultralytics_compatible": True,
            "output_format": "ultralytics_pt",
            "output_weights": str(out_path),
        })
        meta.update(compression_stats(model, out_path))
        return meta, model

    if method == "structural-low-rank":
        rank_ratio = float(kwargs.get("rank_ratio", 0.84))
        target_layers = kwargs.get("target_layers", ["backbone", "neck"])
        target_scope = kwargs.get("target_scope", "backbone-neck")
        selection_strategy = kwargs.get("selection_strategy", "all")
        max_replaced_layers = kwargs.get("max_replaced_layers")
        sensitivity_report = kwargs.get("sensitivity_report")
        max_map50_drop = float(kwargs.get("max_map50_drop", 0.03))
        sensitivity_file = kwargs.get("sensitivity_file")
        if sensitivity_report is None and sensitivity_file:
            sensitivity_report = load_sensitivity_report(Path(sensitivity_file))
        elif sensitivity_report is None and selection_strategy == "sensitivity":
            try:
                sensitivity_report = load_sensitivity_report()
            except FileNotFoundError:
                pass

        model, meta_slr = apply_structural_low_rank(
            model,
            rank_ratio=rank_ratio,
            target_layers=target_layers,
            target_scope=target_scope,
            max_replaced_layers=max_replaced_layers,
            selection_strategy=selection_strategy,
            sensitivity_report=sensitivity_report,
            max_map50_drop=max_map50_drop,
        )
        meta.update(meta_slr)

        if meta_slr["num_replaced_layers"] == 0:
            meta.update({
                "status": "failed_no_layers_replaced",
                "ultralytics_compatible": False,
                "reason": "No compressible layers matched selection; run structural-probe first",
            })
            return meta, None

        bn_batches = int(kwargs.get("bn_recalibration_batches") or 0)
        if bn_batches > 0:
            ran = recalibrate_batchnorm(
                model,
                kwargs.get("dataset"),
                int(kwargs.get("imgsz", 640)),
                kwargs.get("device", "cpu"),
                num_batches=bn_batches,
                data_yaml=kwargs.get("bn_data_yaml") or kwargs.get("data_yaml"),
            )
            meta["bn_recalibration_batches_ran"] = ran
            meta["bn_data_yaml"] = kwargs.get("bn_data_yaml") or kwargs.get("data_yaml")

        suffix = ""
        run_label = kwargs.get("run_label")
        if run_label:
            suffix = f"_{run_label}"
        elif max_replaced_layers:
            suffix = f"_top{max_replaced_layers}"
        if selection_strategy == "pareto-gain":
            suffix += "_pareto"
        elif selection_strategy == "sensitivity":
            suffix += "_sens"
        elif selection_strategy == "sensitivity":
            suffix = "_sens"
        out_path = output_dir / f"structural_low_rank_r{rank_ratio:.3f}{suffix}.pt"
        model.save(str(out_path))
        meta.update({
            "status": "ok",
            "ultralytics_compatible": True,
            "output_format": "ultralytics_pt",
            "output_weights": str(out_path),
        })
        meta.update(compression_stats(model, out_path))
        return meta, model

    raise ValueError(f"Unknown compression method: {method}")
