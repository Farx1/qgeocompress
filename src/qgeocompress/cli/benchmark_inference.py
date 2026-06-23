from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgeocompress.evaluation.system_metrics import benchmark_inference
from qgeocompress.utils.config import project_root, save_json
from qgeocompress.utils.device import resolve_device
from qgeocompress.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference latency and VRAM")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", default="dota128")
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    logger = setup_logging()
    device = resolve_device(args.device)
    results = benchmark_inference(
        args.weights,
        dataset=args.dataset,
        batch_sizes=args.batch_size,
        imgsz=args.imgsz,
        device=device,
    )
    out = args.output or project_root() / "results" / "summaries" / f"benchmark_{args.weights.stem}.json"
    save_json({"weights": str(args.weights), "benchmarks": results}, out)
    logger.info("Benchmark saved to %s", out)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
