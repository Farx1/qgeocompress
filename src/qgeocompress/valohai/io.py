from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def outputs_dir() -> Path:
    """Resolve Valohai outputs directory (or local fallback for ad-hoc runs)."""
    for candidate in (
        os.environ.get("VALOHAI_OUTPUTS_DIR"),
        "/valohai/outputs",
        "valohai_outputs",
    ):
        if candidate:
            path = Path(candidate)
            path.mkdir(parents=True, exist_ok=True)
            return path
    fallback = Path("valohai_outputs")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def log_metric(payload: dict[str, Any]) -> None:
    """Emit JSON metrics to stdout for Valohai experiment tracking."""
    print(json.dumps(payload), flush=True)


def write_json(name: str, data: dict[str, Any]) -> Path:
    dst = outputs_dir() / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return dst


def write_text(name: str, content: str) -> Path:
    dst = outputs_dir() / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content, encoding="utf-8")
    return dst


def copy_output(src: str | Path, name: str) -> Path:
    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(f"Missing output source: {src_path}")
    dst = outputs_dir() / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)
    return dst
