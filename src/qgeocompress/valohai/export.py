from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qgeocompress.models.export_core import _has_structural_layers, run_multi_format_export
from qgeocompress.models.load_model import model_size_mb


def _try_onnx_export(weights: Path, onnx_path: Path, imgsz: int = 640) -> dict[str, Any]:
    """Backward-compatible ONNX attempt wrapper."""
    from qgeocompress.models.export_core import try_onnx_export

    return try_onnx_export(weights, onnx_path, imgsz=imgsz)


def run_export_if_accepted(
    quality_gate: dict[str, Any],
    compressed_model: Path,
    *,
    compression_summary: dict[str, Any] | None = None,
    final_report_path: Path | None = None,
    output_dir: Path,
    try_onnx: bool = True,
    try_torchscript: bool = True,
    imgsz: int = 640,
    device: str = "cpu",
    collapsed_fallback: bool = True,
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
        formats: list[str] = ["pt"]
        if try_onnx:
            formats.append("onnx")
        if try_torchscript:
            formats.append("torchscript")

        export_manifest = run_multi_format_export(
            compressed_model,
            output_dir,
            formats=formats,
            imgsz=imgsz,
            device=device,
            collapsed_fallback=collapsed_fallback,
            quality_gate=quality_gate,
        )
        if final_report_path and final_report_path.exists():
            shutil.copy2(final_report_path, output_dir / "export_report.md")

        manifest = {
            **base_manifest,
            **export_manifest,
            "exported": True,
            "export_path": export_manifest["export_path"],
            "export_size_mb": round(model_size_mb(Path(export_manifest["export_path"])), 2),
            "export_format": "ultralytics_pt",
            "note": export_manifest.get(
                "note",
                "Deployable candidate registered for downstream serving pipelines",
            ),
        }
        manifest_name = "export_manifest.json"
    elif status == "accepted_for_research":
        manifest = {
            **base_manifest,
            "exported": False,
            "export_path": None,
            "structural": _has_structural_layers(compressed_model),
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
