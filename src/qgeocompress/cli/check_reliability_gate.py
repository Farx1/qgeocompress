from __future__ import annotations

import argparse
from pathlib import Path

from qgeocompress.evaluation.reliability_gate import GateThresholds, evaluate_reliability_gate
from qgeocompress.utils.config import load_json, project_root, save_json
from qgeocompress.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate post-training quality gate (accept/reject compressed model)",
    )
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline calibration JSON")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate calibration JSON")
    parser.add_argument(
        "--compression-summary",
        type=Path,
        default=None,
        help="Optional compress_model summary JSON (param reduction, layers)",
    )
    parser.add_argument("--max-map50-drop", type=float, default=0.10)
    parser.add_argument("--max-cer08-delta", type=float, default=0.02)
    parser.add_argument("--min-param-reduction-pct", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    logger = setup_logging()
    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    compression_meta = load_json(args.compression_summary) if args.compression_summary else None

    thresholds = GateThresholds(
        max_map50_drop=args.max_map50_drop,
        max_cer08_delta=args.max_cer08_delta,
        min_param_reduction_pct=args.min_param_reduction_pct,
    )
    result = evaluate_reliability_gate(baseline, candidate, compression_meta, thresholds)
    result["baseline_run_id"] = baseline.get("run_id")
    result["candidate_run_id"] = candidate.get("run_id")

    out = args.output or (
        project_root() / "results" / "summaries" / f"reliability_gate_{candidate.get('run_id', 'candidate')}.json"
    )
    save_json(result, out)

    logger.info(
        "Gate %s — status=%s export=%s mAP50 drop=%.4f CER delta=%.4f params=%.2f%% → %s",
        result.get("candidate_run_id"),
        result["gate_status"],
        result["accepted_for_export"],
        result.get("map50_drop_vs_baseline") or 0.0,
        result.get("cer08_delta_vs_baseline") or 0.0,
        result.get("real_param_reduction_pct") or 0.0,
        out,
    )


if __name__ == "__main__":
    main()
