from __future__ import annotations

import argparse
from pathlib import Path

from qgeocompress.evaluation.report import ReportOptions, generate_report
from qgeocompress.utils.config import project_root
from qgeocompress.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate comparison table and Phase 2A figures")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument(
        "--exclude-finetune-failures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude failed fine-tune runs from the main report (default: true)",
    )
    parser.add_argument(
        "--include-unsupported-finetune",
        action="store_true",
        default=False,
        help="Include fine-tune / unsupported_train_api_state_loss runs in the main report",
    )
    args = parser.parse_args(argv)

    logger = setup_logging()
    results_dir = args.results_dir or project_root() / "results" / "summaries"
    options = ReportOptions(
        exclude_finetune_runs=not args.include_unsupported_finetune,
        include_unsupported_finetune=args.include_unsupported_finetune,
        exclude_finetune_failures=args.exclude_finetune_failures,
    )
    report = generate_report(results_dir, options=options)
    logger.info("Report generated: %d included, %d excluded", report["num_included"], report["num_excluded"])
    logger.info("Phase 2A summary: %s", report.get("phase2a_summary_md"))


if __name__ == "__main__":
    main()
