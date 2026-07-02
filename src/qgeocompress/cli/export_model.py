from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from qgeocompress.models.export_onnx import export_onnx
from qgeocompress.utils.config import project_root, save_json
from qgeocompress.utils.logging import setup_logging
from qgeocompress.valohai.export import _has_structural_layers, _try_onnx_export


def export_model(
    weights: Path,
    output_dir: Path,
    *,
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
    quality_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export baseline ONNX or copy structural .pt with clear skip reason."""
    output_dir.mkdir(parents=True, exist_ok=True)
    export_pt = output_dir / "export_model.pt"
    shutil.copy2(weights, export_pt)

    result: dict[str, Any] = {
        "source_weights": str(weights),
        "export_path": str(export_pt),
        "export_format": "ultralytics_pt",
        "structural": _has_structural_layers(weights),
    }

    if result["structural"]:
        result["onnx"] = {
            "attempted": False,
            "success": False,
            "path": None,
            "reason": "StructuralLowRankConv2d requires no-fuse; ONNX export not supported in MVP",
        }
        result["note"] = "Structural checkpoint copied as .pt only (no ONNX)"
    else:
        onnx_path = output_dir / "export_model.onnx"
        try:
            exported = export_onnx(weights, output_dir=output_dir, imgsz=imgsz, opset=opset, simplify=simplify)
            result["onnx"] = {
                "attempted": True,
                "success": True,
                "path": str(exported),
                "reason": None,
            }
        except Exception as exc:
            onnx_fallback = _try_onnx_export(weights, onnx_path, imgsz=imgsz)
            result["onnx"] = onnx_fallback
            if not onnx_fallback.get("success"):
                result["onnx"]["reason"] = str(exc)

    if quality_gate is not None:
        result["quality_gate"] = quality_gate

    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export compressed or baseline model for deployment")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--quality-gate", type=Path, default=None, help="Optional gate JSON for manifest metadata")
    args = parser.parse_args(argv)

    logger = setup_logging()
    out_dir = args.output_dir or project_root() / "runs" / "export" / args.weights.stem
    gate = None
    if args.quality_gate is not None:
        import json

        gate = json.loads(args.quality_gate.read_text())

    manifest = export_model(
        args.weights,
        out_dir,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        quality_gate=gate,
    )
    manifest_path = out_dir / "export_manifest.json"
    save_json(manifest, manifest_path)

    if manifest.get("structural"):
        logger.info(
            "Structural model — ONNX skipped (%s). Copied .pt to %s",
            manifest["onnx"]["reason"],
            manifest["export_path"],
        )
    else:
        onnx = manifest.get("onnx", {})
        if onnx.get("success"):
            logger.info("ONNX export succeeded: %s", onnx.get("path"))
        else:
            logger.warning("ONNX export failed: %s", onnx.get("reason"))

    logger.info("Export manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
