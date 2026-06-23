from __future__ import annotations

import argparse
from pathlib import Path

from qgeocompress.data.prepare_dota import prepare_dataset_report
from qgeocompress.utils.config import project_root
from qgeocompress.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare and validate a geospatial dataset")
    parser.add_argument("--dataset", default="dota128", choices=["dota128", "dota", "xview"])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    logger = setup_logging()
    output_dir = args.output_dir or project_root() / "results" / "summaries"
    summary = prepare_dataset_report(args.dataset, output_dir)
    logger.info("Dataset report: %s", summary)


if __name__ == "__main__":
    main()
