from __future__ import annotations

import numpy as np


def coverage_risk_curve(
    confidences: np.ndarray,
    correct: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Compute selective prediction coverage-risk trade-off."""
    confidences = np.asarray(confidences)
    correct = np.asarray(correct)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)

    coverages: list[float] = []
    risks: list[float] = []
    accuracies: list[float] = []

    for t in thresholds:
        mask = confidences >= t
        coverage = float(mask.mean())
        if mask.any():
            acc = float(correct[mask].mean())
            risk = 1.0 - acc
        else:
            acc = 0.0
            risk = 1.0
        coverages.append(coverage)
        risks.append(risk)
        accuracies.append(acc)

    return {"thresholds": thresholds.tolist(), "coverage": coverages, "risk": risks, "accuracy": accuracies}


def selective_accuracy(confidences: np.ndarray, correct: np.ndarray, threshold: float = 0.8) -> float:
    mask = confidences >= threshold
    if not mask.any():
        return 0.0
    return float(correct[mask].mean())
