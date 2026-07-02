from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qgeocompress.models.export_core import ExportFormat, run_multi_format_export
from qgeocompress.utils.config import project_root, save_json
from qgeocompress.utils.logging import setup_logging


def export_model(
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
    """Export checkpoint to ONNX, TorchScript, and/or .pt with manifest metadata."""
    return run_multi_format_export(
        weights,
        output_dir,
        formats=formats,
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        device=device,
        collapsed_fallback=collapsed_fallback,
        quality_gate=quality_gate,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export compressed or baseline model for deployment")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--format",
        dest="formats",
        choices=["onnx", "torchscript", "pt", "all"],
        default="all",
        help="Export format(s); default exports pt + onnx + torchscript",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for export (cpu or cuda)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--no-collapsed-fallback", action="store_true", help="Skip structural collapse fallback")
    parser.add_argument("--quality-gate", type=Path, default=None, help="Optional gate JSON for manifest metadata")
    args = parser.parse_args(argv)

    logger = setup_logging()
    out_dir = args.output_dir or project_root() / "runs" / "export" / args.weights.stem
    gate = None
    if args.quality_gate is not None:
        gate = json.loads(args.quality_gate.read_text(encoding="utf-8"))

    manifest = export_model(
        args.weights,
        out_dir,
        formats=args.formats,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        device=args.device,
        collapsed_fallback=not args.no_collapsed_fallback,
        quality_gate=gate,
    )
    manifest_path = out_dir / "export_manifest.json"
    save_json(manifest, manifest_path)

    formats = manifest.get("formats", {})
    for name, result in formats.items():
        if not result.get("attempted"):
            continue
        if result.get("success"):
            logger.info("%s export succeeded: %s", name.upper(), result.get("path"))
        elif result.get("reason"):
            logger.warning("%s export skipped/failed: %s", name.upper(), result.get("reason"))

    collapsed = manifest.get("collapsed_fallback") or {}
    if collapsed.get("attempted"):
        if collapsed.get("success"):
            logger.info("Collapsed fallback succeeded (see collapsed_fallback in manifest)")
        else:
            logger.warning("Collapsed fallback did not produce deployable artifacts")

    logger.info("Export manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
