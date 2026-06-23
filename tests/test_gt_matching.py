import numpy as np
import pytest

from qgeocompress.evaluation.gt_matching import (
    OBBDetection,
    axis_aligned_iou,
    compute_pairwise_iou,
    match_predictions_to_gt,
)


def _square(x: float, y: float, size: float) -> np.ndarray:
    return np.array(
        [[x, y], [x + size, y], [x + size, y + size], [x, y + size]],
        dtype=np.float64,
    )


def _det(class_id: int, poly: np.ndarray, conf: float = 0.9) -> OBBDetection:
    return OBBDetection(class_id=class_id, polygon=poly, confidence=conf)


def test_single_prediction_matches_single_gt():
    gt = [_det(0, _square(0, 0, 10), conf=None)]
    pred = [_det(0, _square(1, 1, 8), conf=0.95)]
    summary = match_predictions_to_gt(pred, gt, iou_threshold=0.5, iou_mode="axis_aligned")
    assert summary.num_tp == 1
    assert summary.num_fp == 0
    assert summary.num_fn == 0
    assert summary.correct[0] == 1


def test_gt_not_matched_twice():
    gt = [_det(0, _square(0, 0, 10), conf=None)]
    preds = [
        _det(0, _square(1, 1, 8), conf=0.95),
        _det(0, _square(2, 2, 6), conf=0.80),
    ]
    summary = match_predictions_to_gt(preds, gt, iou_threshold=0.3, iou_mode="axis_aligned")
    assert summary.num_tp == 1
    assert summary.num_fp == 1


def test_false_positive_when_iou_too_low():
    gt = [_det(0, _square(0, 0, 10), conf=None)]
    pred = [_det(0, _square(50, 50, 10), conf=0.9)]
    summary = match_predictions_to_gt(pred, gt, iou_threshold=0.5, iou_mode="axis_aligned")
    assert summary.num_tp == 0
    assert summary.num_fp == 1
    assert summary.num_fn == 1


def test_false_positive_when_class_differs():
    gt = [_det(0, _square(0, 0, 10), conf=None)]
    pred = [_det(1, _square(1, 1, 8), conf=0.9)]
    summary = match_predictions_to_gt(pred, gt, iou_threshold=0.5, iou_mode="axis_aligned")
    assert summary.num_tp == 0
    assert summary.num_fp == 1
    assert summary.num_fn == 1


def test_lower_confidence_can_match_after_high_conf_fp():
    """High-confidence FP does not consume GT; lower-confidence TP can still match."""
    gt = [_det(0, _square(0, 0, 10), conf=None)]
    preds = [
        _det(0, _square(50, 50, 10), conf=0.99),
        _det(0, _square(1, 1, 8), conf=0.70),
    ]
    summary = match_predictions_to_gt(preds, gt, iou_threshold=0.5, iou_mode="axis_aligned")
    assert summary.num_tp == 1
    assert summary.num_fp == 1


def test_axis_aligned_iou_identical_boxes():
    poly = _square(0, 0, 10)
    assert axis_aligned_iou(poly, poly) == pytest.approx(1.0)


def test_compute_pairwise_iou_modes():
    a = _square(0, 0, 10)
    b = _square(1, 1, 8)
    assert compute_pairwise_iou(a, b, iou_mode="axis_aligned") > 0.5
