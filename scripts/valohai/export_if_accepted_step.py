#!/usr/bin/env python3
import json
from pathlib import Path

from qgeocompress.valohai.export import run_export_if_accepted
from qgeocompress.valohai.io import copy_output, log_metric, outputs_dir, write_json


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Valohai step: export compressed model if quality gate accepts")
    parser.add_argument("--quality-gate", type=str, required=True)
    parser.add_argument("--compressed-model", type=str, required=True)
    parser.add_argument("--compression-summary", type=str, default=None)
    parser.add_argument("--final-report", type=str, default=None)
    parser.add_argument("--no-onnx", action="store_true", help="Skip ONNX export attempt")
    parser.add_argument("--no-torchscript", action="store_true", help="Skip TorchScript export attempt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    gate = json.loads(Path(args.quality_gate).read_text(encoding="utf-8"))
    compression_summary = None
    if args.compression_summary:
        compression_summary = json.loads(Path(args.compression_summary).read_text(encoding="utf-8"))

    result = run_export_if_accepted(
        gate,
        Path(args.compressed_model),
        compression_summary=compression_summary,
        final_report_path=Path(args.final_report) if args.final_report else None,
        output_dir=outputs_dir(),
        try_onnx=not args.no_onnx,
        try_torchscript=not args.no_torchscript,
        imgsz=args.imgsz,
        device=args.device,
    )
    manifest = result["manifest"]
    write_json(result["manifest_name"], manifest)

    if manifest.get("exported") and manifest.get("export_path"):
        copy_output(manifest["export_path"], "export_model.pt")
        onnx = manifest.get("onnx") or {}
        if onnx.get("success") and onnx.get("path"):
            copy_output(onnx["path"], "export_model.onnx")
        ts = manifest.get("torchscript") or {}
        if ts.get("success") and ts.get("path"):
            copy_output(ts["path"], "export_model.torchscript")
        collapsed = manifest.get("collapsed_fallback") or {}
        collapsed_pt = (collapsed.get("collapsed_pt") or {})
        if collapsed_pt.get("success") and collapsed_pt.get("path"):
            copy_output(collapsed_pt["path"], "export_model_collapsed.pt")

    if args.final_report and Path(args.final_report).exists():
        copy_output(args.final_report, "export_report.md")

    log_metric({
        "stage": "export_if_accepted",
        "valohai_status": manifest.get("valohai_status"),
        "exported": manifest.get("exported", False),
        "export_size_mb": manifest.get("export_size_mb"),
        "onnx_success": (manifest.get("onnx") or {}).get("success"),
    })


if __name__ == "__main__":
    main()
