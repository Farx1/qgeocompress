#!/usr/bin/env python3
from pathlib import Path

from qgeocompress.valohai.io import copy_output, log_metric, outputs_dir, write_json
from qgeocompress.valohai.steps import run_structural_probe


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Valohai step: structural layer probe")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--select-yaml", type=str, required=True)
    parser.add_argument("--dataset", default="dota128")
    parser.add_argument("--rank-ratio", type=float, default=0.84)
    parser.add_argument("--target-layers", nargs="+", default=["backbone", "neck"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    sensitivity_out = outputs_dir() / "structural_layer_sensitivity.json"
    result = run_structural_probe(
        Path(args.model),
        Path(args.select_yaml),
        dataset=args.dataset,
        rank_ratio=args.rank_ratio,
        target_layers=args.target_layers,
        imgsz=args.imgsz,
        device=args.device,
        sensitivity_out=sensitivity_out,
    )
    copy_output(result["sensitivity_report_path"], "structural_layer_sensitivity.json")
    write_json("probe_summary.json", result["summary"])
    log_metric(result["summary"])


if __name__ == "__main__":
    main()
