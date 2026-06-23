#!/usr/bin/env python3
"""Evaluate robustness under synthetic corruptions."""
import argparse

from qgeocompress.evaluation.robustness_metrics import evaluate_robustness
from qgeocompress.utils.config import project_root, save_json
from qgeocompress.utils.device import resolve_device
from qgeocompress.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", default="dota128")
    parser.add_argument("--corruptions", nargs="+", default=["cloud", "blur", "noise"])
    parser.add_argument("--severity", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    setup_logging()
    results = evaluate_robustness(
        args.weights,
        args.dataset,
        args.corruptions,
        severities=args.severity,
        device=resolve_device(args.device),
    )
    out = project_root() / "results" / "summaries" / f"robustness_{args.dataset}.json"
    save_json(results, out)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
