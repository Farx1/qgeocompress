from __future__ import annotations

import numpy as np


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidences.size == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi if i < n_bins - 1 else confidences <= hi)
        if not mask.any():
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)
    return float(ece)


def confident_error_rate(
    confidences: np.ndarray,
    correct: np.ndarray,
    threshold: float = 0.9,
) -> float:
    """Fraction of high-confidence predictions that are wrong."""
    confidences = np.asarray(confidences)
    correct = np.asarray(correct)
    mask = confidences >= threshold
    if not mask.any():
        return 0.0
    return float((1 - correct[mask]).mean())
