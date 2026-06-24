#!/usr/bin/env python3
from qgeocompress.valohai.steps import run_baseline_eval
from qgeocompress.valohai.io import log_metric, write_json


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Valohai step: baseline evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test-yaml", type=str, required=True)
    parser.add_argument("--dataset", default="dota128")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from pathlib import Path

    result = run_baseline_eval(
        Path(args.model),
        Path(args.test_yaml),
        dataset=args.dataset,
        iou_threshold=args.iou_threshold,
        imgsz=args.imgsz,
        device=args.device,
    )
    write_json("baseline_metrics.json", result["metrics"])
    write_json("baseline_calibration.json", result["calibration"])
    write_json("baseline_predictions.json", result["predictions"])
    log_metric(result["metrics"])


if __name__ == "__main__":
    main()
