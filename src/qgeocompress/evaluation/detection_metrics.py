from __future__ import annotations

from typing import Any

from qgeocompress.models.load_model import extract_detection_metrics


def summarize_validation(results: Any) -> dict[str, float]:
    return extract_detection_metrics(results)
