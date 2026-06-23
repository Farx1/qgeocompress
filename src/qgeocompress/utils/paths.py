from __future__ import annotations

from pathlib import Path

from qgeocompress.utils.config import project_root

# Ultralytics default layout for this project's baseline training runs.
DEFAULT_BASELINE_WEIGHTS = (
    project_root() / "runs" / "obb" / "runs" / "baseline" / "train" / "weights" / "best.pt"
)
BASELINE_SEARCH_ROOT = project_root() / "runs" / "obb" / "runs" / "baseline"


def resolve_baseline_weights(weights: Path | str | None = None) -> Path:
    """Resolve baseline checkpoint — never pick a compressed/fine-tuned best.pt by accident."""
    if weights is not None:
        path = Path(weights)
        if path.exists():
            return path
        raise FileNotFoundError(f"Baseline weights not found: {path}")

    if DEFAULT_BASELINE_WEIGHTS.exists():
        return DEFAULT_BASELINE_WEIGHTS

    if BASELINE_SEARCH_ROOT.exists():
        matches = sorted(BASELINE_SEARCH_ROOT.glob("**/weights/best.pt"))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "No baseline best.pt found. Train baseline first or pass --weights explicitly, e.g.\n"
        "  runs/obb/runs/baseline/train/weights/best.pt"
    )
