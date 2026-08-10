#!/usr/bin/env python3
"""Argparse wrapper for run_holdout_pipeline.sh."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run hold-out compression pipeline end-to-end")
    parser.add_argument("--skip-train", action="store_true", help="Use existing baseline_holdout checkpoint")
    parser.add_argument("--skip-probe", action="store_true", help="Reuse existing sensitivity JSON")
    parser.add_argument("--gpu", action="store_true", help="Pass --device cuda to underlying scripts")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args(argv)

    script = Path(__file__).resolve().parent / "run_holdout_pipeline.sh"
    cmd = [str(script)]
    if args.skip_train:
        cmd.append("--skip-train")
    if args.skip_probe:
        cmd.append("--skip-probe")
    if args.gpu:
        cmd.append("--gpu")
    if args.dry_run:
        cmd.append("--dry-run")

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
