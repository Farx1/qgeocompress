#!/usr/bin/env python3
from pathlib import Path

from qgeocompress.valohai.inference_compare import run_inference_compare
from qgeocompress.valohai.io import log_metric, write_json


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Valohai step: baseline vs compressed inference compare")
    parser.add_argument("--baseline-model", type=str, required=True)
    parser.add_argument("--compressed-model", type=str, required=True)
    parser.add_argument("--test-yaml", type=str, required=True)
    parser.add_argument("--dataset", default="dota128")
    parser.add_argument("--compression", default="structural-low-rank")
    parser.add_argument("--rank-ratio", type=float, default=0.84)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    result = run_inference_compare(
        Path(args.baseline_model),
        Path(args.compressed_model),
        Path(args.test_yaml),
        dataset=args.dataset,
        compression=args.compression,
        rank_ratio=args.rank_ratio,
        iou_threshold=args.iou_threshold,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
    )

    comparison = {
        "stage": result["stage"],
        "test_yaml": result["test_yaml"],
        "baseline": result["baseline"],
        "compressed": result["compressed"],
        "delta": result["delta"],
    }
    write_json("comparison_metrics.json", comparison)
    write_json("comparison_full.json", result)
    write_json("inference_latency.json", result["inference_latency"])
    write_json("calibration_comparison.json", result["calibration_comparison"])

    log_metric({
        "stage": "inference_compare",
        "compressed_map50": result["compressed"].get("map50"),
        "baseline_map50": result["baseline"].get("map50"),
        "map50_drop": result["delta"].get("map50_drop"),
        "compressed_cer08": result["compressed"].get("cer08"),
        "baseline_cer08": result["baseline"].get("cer08"),
        "cer08_delta": result["delta"].get("cer08_delta"),
        "real_param_reduction_pct": result["delta"].get("param_reduction_pct"),
    })


if __name__ == "__main__":
    main()
