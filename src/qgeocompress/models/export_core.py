"""Multi-format export orchestration for baseline and compressed checkpoints."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from qgeocompress.compression.structural_low_rank import StructuralLowRankConv2d
from qgeocompress.models.export_onnx import export_onnx
from qgeocompress.models.export_torchscript import export_torchscript
from qgeocompress.models.load_model import model_size_mb

ExportFormat = Literal["pt", "onnx", "torchscript", "all"]
FormatAttempt = dict[str, Any]

_SKIP_STRUCTURAL_DIRECT = (
    "StructuralLowRankConv2d requires no-fuse inference; direct export not supported"
)
_SKIP_TENSORRT = "TensorRT requires NVIDIA GPU, CUDA, and optional tensorrt extra (not run in CI)"


def _attempt(
    *,
    attempted: bool = True,
    success: bool = False,
    path: str | None = None,
    reason: str | None = None,
    **extra: Any,
) -> FormatAttempt:
    return {"attempted": attempted, "success": success, "path": path, "reason": reason, **extra}


def _normalize_formats(formats: ExportFormat | list[str]) -> set[str]:
    if isinstance(formats, str):
        if formats == "all":
            return {"pt", "onnx", "torchscript"}
        return {formats}
    normalized = {f.lower() for f in formats}
    if "all" in normalized:
        return {"pt", "onnx", "torchscript"}
    return normalized


def classify_checkpoint(weights: Path) -> dict[str, Any]:
    """Detect structural layers and infer compression family from checkpoint."""
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights), task="obb")
        structural = any(isinstance(m, StructuralLowRankConv2d) for m in model.model.modules())
        model_type = "structural-low-rank" if structural else "baseline"
        return {
            "structural": structural,
            "model_type": model_type,
            "ultralytics_compatible": not structural,
        }
    except Exception as exc:
        return {
            "structural": False,
            "model_type": "unknown",
            "ultralytics_compatible": True,
            "load_error": str(exc),
        }


def _has_structural_layers(weights: Path) -> bool:
    return classify_checkpoint(weights)["structural"]


def try_onnx_export(
    weights: Path,
    onnx_path: Path,
    *,
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
    device: str = "cpu",
) -> FormatAttempt:
    if _has_structural_layers(weights):
        return _attempt(
            reason=f"{_SKIP_STRUCTURAL_DIRECT}; use collapsed fallback",
        )
    try:
        exported = export_onnx(
            weights,
            output_dir=onnx_path.parent,
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
        )
        target = onnx_path if exported != onnx_path else exported
        if exported.name != onnx_path.name:
            shutil.copy2(exported, onnx_path)
            target = onnx_path
        return _attempt(success=True, path=str(target))
    except Exception as exc:
        return _attempt(reason=str(exc))


def try_torchscript_export(
    weights: Path,
    ts_path: Path,
    *,
    imgsz: int = 640,
    device: str = "cpu",
) -> FormatAttempt:
    if _has_structural_layers(weights):
        return _attempt(
            reason=f"{_SKIP_STRUCTURAL_DIRECT}; YOLO TorchScript trace fails on custom modules",
        )
    try:
        exported = export_torchscript(weights, output_dir=ts_path.parent, imgsz=imgsz, device=device)
        target = ts_path if exported != ts_path else exported
        if exported.name != ts_path.name:
            shutil.copy2(exported, ts_path)
            target = ts_path
        return _attempt(success=True, path=str(target))
    except Exception as exc:
        return _attempt(reason=str(exc))


def try_collapsed_fallback(
    weights: Path,
    output_dir: Path,
    *,
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
    device: str = "cpu",
    try_onnx: bool = True,
    try_torchscript: bool = True,
) -> dict[str, Any]:
    """Collapse structural layers and attempt Ultralytics-compatible exports."""
    from ultralytics import YOLO

    from qgeocompress.compression.structural_collapse import collapse_structural_model

    if not _has_structural_layers(weights):
        return {
            "attempted": False,
            "success": False,
            "collapse_meta": None,
            "collapsed_pt": _attempt(attempted=False, reason="Checkpoint has no structural layers"),
            "collapsed_onnx": _attempt(attempted=False, reason="Not applicable"),
            "collapsed_torchscript": _attempt(attempted=False, reason="Not applicable"),
        }

    try:
        model = YOLO(str(weights), task="obb")
        collapsed, collapse_meta = collapse_structural_model(model)
        collapsed_pt = output_dir / "export_model_collapsed.pt"
        collapsed.save(str(collapsed_pt))
        pt_result = _attempt(success=True, path=str(collapsed_pt), collapse_meta=collapse_meta)

        onnx_result = _attempt(attempted=False, reason="Collapsed ONNX not requested")
        if try_onnx:
            onnx_result = try_onnx_export(
                collapsed_pt,
                output_dir / "export_model_collapsed.onnx",
                imgsz=imgsz,
                opset=opset,
                simplify=simplify,
                device=device,
            )

        ts_result = _attempt(attempted=False, reason="Collapsed TorchScript not requested")
        if try_torchscript:
            ts_result = try_torchscript_export(
                collapsed_pt,
                output_dir / "export_model_collapsed.torchscript",
                imgsz=imgsz,
                device=device,
            )

        success = pt_result["success"] or onnx_result.get("success") or ts_result.get("success")
        return {
            "attempted": True,
            "success": success,
            "collapse_meta": collapse_meta,
            "collapsed_pt": pt_result,
            "collapsed_onnx": onnx_result,
            "collapsed_torchscript": ts_result,
            "note": "Collapsed export folds structural layers to standard Conv2d; loses param savings",
        }
    except Exception as exc:
        return {
            "attempted": True,
            "success": False,
            "collapse_meta": None,
            "collapsed_pt": _attempt(reason=str(exc)),
            "collapsed_onnx": _attempt(attempted=False, reason="Collapse failed"),
            "collapsed_torchscript": _attempt(attempted=False, reason="Collapse failed"),
            "note": "Structural collapse failed",
        }


def run_multi_format_export(
    weights: Path,
    output_dir: Path,
    *,
    formats: ExportFormat | list[str] = "all",
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
    device: str = "cpu",
    collapsed_fallback: bool = True,
    quality_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export checkpoint to requested formats with manifest-friendly results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = _normalize_formats(formats)
    classification = classify_checkpoint(weights)
    structural = classification["structural"]

    export_pt = output_dir / "export_model.pt"
    shutil.copy2(weights, export_pt)

    format_results: dict[str, FormatAttempt] = {
        "pt": _attempt(success=True, path=str(export_pt)),
        "onnx": _attempt(attempted=False, reason="Format not requested"),
        "torchscript": _attempt(attempted=False, reason="Format not requested"),
        "tensorrt": _attempt(
            attempted=False,
            reason=_SKIP_TENSORRT,
            note="Use export_tensorrt.export_tensorrt() on GPU host with tensorrt installed",
        ),
    }

    if "onnx" in requested:
        format_results["onnx"] = try_onnx_export(
            weights,
            output_dir / "export_model.onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            device=device,
        )

    if "torchscript" in requested:
        format_results["torchscript"] = try_torchscript_export(
            weights,
            output_dir / "export_model.torchscript",
            imgsz=imgsz,
            device=device,
        )

    collapsed: dict[str, Any] | None = None
    if structural and collapsed_fallback and ("onnx" in requested or "torchscript" in requested):
        collapsed = try_collapsed_fallback(
            weights,
            output_dir,
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            device=device,
            try_onnx="onnx" in requested,
            try_torchscript="torchscript" in requested,
        )

    manifest: dict[str, Any] = {
        "stage": "export",
        "timestamp": datetime.now(UTC).isoformat(),
        "source_weights": str(weights),
        "export_path": str(export_pt),
        "export_format": "ultralytics_pt",
        "export_size_mb": round(model_size_mb(export_pt), 2),
        "requested_formats": sorted(requested),
        "device": device,
        **classification,
        "formats": format_results,
        "collapsed_fallback": collapsed,
    }

    if structural:
        manifest["note"] = (
            "Structural checkpoint copied as .pt; direct ONNX/TorchScript skipped. "
            "See collapsed_fallback for Ultralytics-compatible export attempt."
        )
    elif format_results["onnx"].get("success"):
        manifest["note"] = "Baseline or in-place compressed model exported with ONNX support"

    if quality_gate is not None:
        manifest["quality_gate"] = quality_gate

    # Backward compatibility for Valohai consumers
    manifest["onnx"] = format_results["onnx"]
    manifest["torchscript"] = format_results["torchscript"]

    return manifest
