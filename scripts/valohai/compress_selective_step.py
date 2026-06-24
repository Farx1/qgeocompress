#!/usr/bin/env python3
from pathlib import Path

from qgeocompress.valohai.io import copy_output, log_metric, write_json
from qgeocompress.valohai.steps import run_compress_selective


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Valohai step: selective structural compression")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--sensitivity-json", type=str, required=True)
    parser.add_argument("--train-yaml", type=str, required=True)
    parser.add_argument("--test-yaml", type=str, required=True)
    parser.add_argument("--dataset", default="dota128")
    parser.add_argument("--rank-ratio", type=float, default=0.84)
    parser.add_argument("--max-replaced-layers", type=int, default=5)
    parser.add_argument("--selection-strategy", default="pareto-gain")
    parser.add_argument("--bn-recalibration-batches", type=int, default=20)
    parser.add_argument("--max-map50-drop", type=float, default=0.03)
    parser.add_argument("--target-layers", nargs="+", default=["backbone", "neck"])
    parser.add_argument("--run-label", default="valohai_compress")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    result = run_compress_selective(
        Path(args.model),
        Path(args.sensitivity_json),
        Path(args.train_yaml),
        Path(args.test_yaml),
        dataset=args.dataset,
        rank_ratio=args.rank_ratio,
        max_replaced_layers=args.max_replaced_layers,
        selection_strategy=args.selection_strategy,
        bn_recalibration_batches=args.bn_recalibration_batches,
        max_map50_drop=args.max_map50_drop,
        target_layers=args.target_layers,
        run_label=args.run_label,
        imgsz=args.imgsz,
        device=args.device,
    )
    weights = result["output_weights"]
    if weights is None or not weights.exists():
        raise RuntimeError("Compression did not produce output weights")
    copy_output(weights, "compressed_model.pt")
    write_json("compression_summary.json", result["compression_summary"])
    write_json("selected_layers.json", result["selected_layers"])
    log_metric(result["stdout_metrics"])


if __name__ == "__main__":
    main()
