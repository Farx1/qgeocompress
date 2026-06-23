from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from qgeocompress.calibration.ece import confident_error_rate, expected_calibration_error
from qgeocompress.calibration.selective_prediction import coverage_risk_curve
from qgeocompress.evaluation.gt_matching import MatchingSummary
from qgeocompress.utils.config import project_root


def maximum_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> float:
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidences.size == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi if i < n_bins - 1 else confidences <= hi)
        if not mask.any():
            continue
        gap = abs(correct[mask].mean() - confidences[mask].mean())
        mce = max(mce, gap)
    return float(mce)


def brier_score(confidences: np.ndarray, correct: np.ndarray) -> float:
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidences.size == 0:
        return 0.0
    return float(np.mean((confidences - correct) ** 2))


def reliability_bins(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidences.size == 0:
        return []

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi if i < n_bins - 1 else confidences <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "avg_confidence": float(confidences[mask].mean()),
            "accuracy": float(correct[mask].mean()),
            "count": count,
        })
    return bins


def selective_accuracy_at_coverage(
    confidences: np.ndarray,
    correct: np.ndarray,
    target_coverage: float,
) -> float:
    """Accuracy when keeping the top-confidence fraction of predictions."""
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidences.size == 0:
        return 0.0

    n_keep = max(1, int(round(len(confidences) * target_coverage)))
    order = np.argsort(-confidences)[:n_keep]
    return float(correct[order].mean())


def compute_calibration_metrics(
    matching: MatchingSummary,
    n_bins: int = 15,
) -> dict[str, Any]:
    conf = matching.confidences
    correct = matching.correct

    curve = coverage_risk_curve(conf, correct)
    metrics: dict[str, Any] = {
        **matching.to_counts_dict(),
        "ece": round(expected_calibration_error(conf, correct, n_bins=n_bins), 6),
        "mce": round(maximum_calibration_error(conf, correct, n_bins=n_bins), 6),
        "brier_score": round(brier_score(conf, correct), 6),
        "confident_error_rate_07": round(confident_error_rate(conf, correct, threshold=0.7), 6),
        "confident_error_rate_08": round(confident_error_rate(conf, correct, threshold=0.8), 6),
        "confident_error_rate_09": round(confident_error_rate(conf, correct, threshold=0.9), 6),
        "selective_accuracy_at_80_coverage": round(
            selective_accuracy_at_coverage(conf, correct, 0.80), 6
        ),
        "selective_accuracy_at_60_coverage": round(
            selective_accuracy_at_coverage(conf, correct, 0.60), 6
        ),
        "reliability_bins": reliability_bins(conf, correct, n_bins=10),
        "coverage_curve": curve,
    }
    return metrics


def plot_reliability_diagram(
    bins: list[dict[str, float]],
    label: str,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))

    if bins:
        xs = [b["avg_confidence"] for b in bins]
        ys = [b["accuracy"] for b in bins]
        sizes = [max(b["count"], 1) for b in bins]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
        ax.scatter(xs, ys, s=[min(s * 2, 200) for s in sizes], alpha=0.85, label=label)
        ax.plot(xs, ys, alpha=0.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram — {label}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_coverage_risk_curve(
    curve: dict[str, list[float]],
    label: str,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    coverages = curve.get("coverage", [])
    accuracies = curve.get("accuracy", [])
    risks = curve.get("risk", [])

    if coverages and accuracies:
        ax.plot(coverages, accuracies, "o-", linewidth=2, label=f"{label} — accuracy")
    if coverages and risks:
        ax.plot(coverages, risks, "s--", linewidth=1.5, alpha=0.7, label=f"{label} — error rate")

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy / error rate")
    ax.set_title(f"Selective prediction — {label}")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_reliability_overlay(
    runs: list[dict[str, Any]],
    output_path: Path | None = None,
) -> Path | None:
    """Overlay reliability diagrams for multiple calibration runs."""
    if not runs:
        return None

    output_path = output_path or project_root() / "results" / "figures" / "calibration_reliability_overlay.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")

    for run in runs:
        bins = run.get("reliability_bins") or []
        if not bins:
            continue
        label = _run_label(run)
        xs = [b["avg_confidence"] for b in bins]
        ys = [b["accuracy"] for b in bins]
        ax.plot(xs, ys, "o-", linewidth=1.5, markersize=5, label=label)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Q-GEOCompress — reliability overlay (GT-matched)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _run_label(run: dict[str, Any]) -> str:
    if run.get("compression") in (None, "none", "baseline"):
        return "baseline"
    if run.get("run_label"):
        return str(run["run_label"])
    if run.get("compression") == "structural-low-rank" and run.get("rank_ratio") is not None:
        return f"structural r={run['rank_ratio']}"
    if run.get("rank_ratio") is not None:
        return f"r={run['rank_ratio']}"
    return run.get("run_id", "?")


def calibration_run_label(run: dict[str, Any]) -> str:
    return _run_label(run)
