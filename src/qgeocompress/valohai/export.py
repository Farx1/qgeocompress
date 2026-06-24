from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qgeocompress.models.load_model import model_size_mb


def _has_structural_layers(weights: Path) -> bool:
    from ultralytics import YOLO

    from qgeocompress.compression.structural_low_rank import StructuralLowRankConv2d

    model = YOLO(str(weights), task="obb")
    return any(isinstance(m, StructuralLowRankConv2d) for m in model.model.modules())


def _try_onnx_export(weights: Path, onnx_path: Path, imgsz: int = 640) -> dict[str, Any]:
    if _has_structural_layers(weights):
        return {
            "attempted": True,
            "success": False,
            "path": None,
            "reason": "StructuralLowRankConv2d requires no-fuse; ONNX export not supported in MVP",
        }
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights), task="obb")
        model.export(format="onnx", imgsz=imgsz, simplify=True)
        default_onnx = weights.with_suffix(".onnx")
        if default_onnx.exists():
            shutil.copy2(default_onnx, onnx_path)
            return {"attempted": True, "success": True, "path": str(onnx_path), "reason": None}
        return {"attempted": True, "success": False, "path": None, "reason": "ONNX file not produced"}
    except Exception as exc:
        return {"attempted": True, "success": False, "path": None, "reason": str(exc)}


def run_export_if_accepted(
    quality_gate: dict[str, Any],
    compressed_model: Path,
    *,
    compression_summary: dict[str, Any] | None = None,
    final_report_path: Path | None = None,
    output_dir: Path,
    try_onnx: bool = True,
    imgsz: int = 640,
) -> dict[str, Any]:
    """Register or reject compressed checkpoint based on quality gate status."""
    output_dir.mkdir(parents=True, exist_ok=True)
    status = quality_gate.get("valohai_status") or quality_gate.get("status", "rejected")
    timestamp = datetime.now(UTC).isoformat()

    base_manifest: dict[str, Any] = {
        "stage": "export_if_accepted",
        "timestamp": timestamp,
        "valohai_status": status,
        "accepted_for_export": quality_gate.get("accepted_for_export", False),
        "map50_drop": quality_gate.get("map50_drop"),
        "cer08_delta": quality_gate.get("cer08_delta"),
        "real_param_reduction_pct": quality_gate.get("real_param_reduction_pct"),
        "source_weights": str(compressed_model),
        "compression_summary": compression_summary,
        "final_report": str(final_report_path) if final_report_path else None,
    }

    if status == "accepted_for_export":
        export_pt = output_dir / "export_model.pt"
        shutil.copy2(compressed_model, export_pt)
        onnx_result = _try_onnx_export(compressed_model, output_dir / "export_model.onnx", imgsz=imgsz) if try_onnx else {
            "attempted": False,
            "success": False,
            "path": None,
            "reason": "ONNX export disabled",
        }
        if final_report_path and final_report_path.exists():
            shutil.copy2(final_report_path, output_dir / "export_report.md")

        manifest = {
            **base_manifest,
            "exported": True,
            "export_path": str(export_pt),
            "export_size_mb": round(model_size_mb(export_pt), 2),
            "onnx": onnx_result,
            "export_format": "ultralytics_pt",
            "note": "Deployable candidate registered for downstream serving pipelines",
        }
        manifest_name = "export_manifest.json"
    elif status == "accepted_for_research":
        manifest = {
            **base_manifest,
            "exported": False,
            "export_path": None,
            "note": "Methodologically valid; retained for research only (param reduction below deploy threshold)",
        }
        manifest_name = "research_manifest.json"
    else:
        manifest = {
            **base_manifest,
            "exported": False,
            "export_path": None,
            "reasons": quality_gate.get("reasons", []),
            "note": "Candidate rejected; no artifact registered for deployment",
        }
        manifest_name = "rejection_manifest.json"

    return {"manifest": manifest, "manifest_name": manifest_name}
