from __future__ import annotations

import argparse
from pathlib import Path

from qgeocompress.evaluation.phase3c_report import (
    build_phase3c_rows,
    render_phase3c_markdown,
    rows_to_csv,
)
from qgeocompress.utils.config import project_root
from qgeocompress.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 3C hold-out comparison report")
    parser.add_argument(
        "--summaries-dir",
        type=Path,
        default=None,
        help="Directory with calibration/compression JSON summaries",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Markdown report path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV table path",
    )
    args = parser.parse_args(argv)

    logger = setup_logging()
    summaries = args.summaries_dir or (project_root() / "results" / "summaries")
    out_md = args.output_md or (project_root() / "results" / "reports" / "qgeocompress_phase3c_comparison.md")
    out_csv = args.output_csv or (project_root() / "results" / "summaries" / "phase3c_comparison_table.csv")

    rows = build_phase3c_rows(summaries)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_phase3c_markdown(rows), encoding="utf-8")
    out_csv.write_text(rows_to_csv(rows), encoding="utf-8")

    logger.info("Phase 3C report — %d rows → %s, %s", len(rows), out_md, out_csv)


if __name__ == "__main__":
    main()
