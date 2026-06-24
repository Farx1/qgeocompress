#!/usr/bin/env python3
import json
from pathlib import Path

from qgeocompress.evaluation.reliability_gate import GateThresholds
from qgeocompress.valohai.gate_report import evaluate_quality_gate, render_final_report
from qgeocompress.valohai.io import log_metric, write_json, write_text


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Valohai step: post-training quality gate")
    parser.add_argument("--comparison-metrics", type=str, required=True)
    parser.add_argument("--compression-summary", type=str, required=True)
    parser.add_argument("--comparison-full", type=str, default=None, help="Optional full compare JSON with calibrations")
    parser.add_argument("--max-map50-drop", type=float, default=0.10)
    parser.add_argument("--max-cer08-delta", type=float, default=0.02)
    parser.add_argument("--min-param-reduction-pct", type=float, default=2.0)
    args = parser.parse_args()

    comparison = json.loads(Path(args.comparison_metrics).read_text(encoding="utf-8"))
    compression_summary = json.loads(Path(args.compression_summary).read_text(encoding="utf-8"))

    if args.comparison_full:
        full = json.loads(Path(args.comparison_full).read_text(encoding="utf-8"))
        comparison = {**full, **comparison}

    thresholds = GateThresholds(
        max_map50_drop=args.max_map50_drop,
        max_cer08_delta=args.max_cer08_delta,
        min_param_reduction_pct=args.min_param_reduction_pct,
    )
    gate = evaluate_quality_gate(comparison, compression_summary, thresholds)
    write_json("quality_gate.json", gate)
    write_text("final_report.md", render_final_report(gate, comparison))
    log_metric({
        "stage": "quality_gate",
        "accepted": gate["accepted"],
        "status": gate["valohai_status"],
        "map50_drop": gate.get("map50_drop"),
        "cer08_delta": gate.get("cer08_delta"),
        "real_param_reduction_pct": gate.get("real_param_reduction_pct"),
    })


if __name__ == "__main__":
    main()
