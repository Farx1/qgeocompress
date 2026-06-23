import numpy as np
import pytest

from qgeocompress.calibration.ece import confident_error_rate, expected_calibration_error
from qgeocompress.evaluation.calibration_real import (
    brier_score,
    compute_calibration_metrics,
    maximum_calibration_error,
    reliability_bins,
    selective_accuracy_at_coverage,
)
from qgeocompress.evaluation.gt_matching import MatchingSummary


def _matching(confidences: list[float], correct: list[int]) -> MatchingSummary:
    conf = np.array(confidences, dtype=np.float64)
    cor = np.array(correct, dtype=np.float64)
    return MatchingSummary(
        outcomes=[],
        num_predictions=len(confidences),
        num_gt=len(confidences),
        num_tp=int(sum(correct)),
        num_fp=int(len(correct) - sum(correct)),
        num_fn=0,
        iou_threshold=0.5,
        iou_mode="axis_aligned",
        confidences=conf,
        correct=cor,
    )


def test_ece_well_calibrated_bins():
    conf = np.array([0.05] * 10 + [0.95] * 10)
    correct = np.array([0] * 10 + [1] * 10)
    ece = expected_calibration_error(conf, correct, n_bins=2)
    assert ece == pytest.approx(0.05, abs=0.06)


def test_ece_miscalibrated():
    conf = np.array([0.9] * 20)
    correct = np.array([0] * 20)
    ece = expected_calibration_error(conf, correct, n_bins=10)
    assert ece > 0.5


def test_cer_at_threshold():
    conf = np.array([0.95, 0.85, 0.75, 0.65])
    correct = np.array([1, 0, 1, 0])
    cer = confident_error_rate(conf, correct, threshold=0.8)
    assert cer == pytest.approx(0.5)


def test_mce_and_brier():
    conf = np.array([0.9, 0.1])
    correct = np.array([0, 1])
    assert maximum_calibration_error(conf, correct, n_bins=2) > 0
    assert brier_score(conf, correct) > 0


def test_reliability_bins_counts():
    conf = np.array([0.15, 0.25, 0.85, 0.95])
    correct = np.array([0, 0, 1, 1])
    bins = reliability_bins(conf, correct, n_bins=2)
    assert len(bins) >= 1
    assert all("avg_confidence" in b for b in bins)


def test_selective_accuracy_at_coverage():
    conf = np.array([0.9, 0.8, 0.3, 0.2])
    correct = np.array([1, 0, 1, 0])
    acc80 = selective_accuracy_at_coverage(conf, correct, 0.5)
    assert 0.0 <= acc80 <= 1.0


def test_compute_calibration_metrics_keys():
    matching = _matching([0.9, 0.8, 0.2], [1, 0, 0])
    metrics = compute_calibration_metrics(matching)
    assert "ece" in metrics
    assert "confident_error_rate_08" in metrics
    assert "selective_accuracy_at_80_coverage" in metrics
    assert metrics["num_predictions"] == 3
