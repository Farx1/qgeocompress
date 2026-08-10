"""AP50 computed from our own GT matching, with bootstrap confidence intervals.

Ultralytics reports a single mAP50 with no uncertainty, which is useless for
saying whether one compression arm beats another on 24 images. The matching in
``gt_matching`` already produces per-prediction (confidence, is_tp) pairs and
per-image GT counts, so AP is a few lines on top and can be resampled over
images to get an interval.

Absolute values differ slightly from Ultralytics (all-point interpolation, our
IoU matching), but every arm is scored the same way, which is what comparisons
need.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from qgeocompress.evaluation.gt_matching import (
    IoUMode,
    OBBDetection,
    match_predictions_to_gt,
)


def _average_precision(confidences: np.ndarray, correct: np.ndarray, num_gt: int) -> float:
    """All-point-interpolated AP for one class."""
    if num_gt == 0:
        return float("nan")
    if confidences.size == 0:
        return 0.0

    order = np.argsort(-confidences)
    tp = correct[order].astype(np.float64)
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1.0 - tp)

    recall = cum_tp / num_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    # Monotone envelope, then integrate over recall.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([precision[0]], precision))
    return float(np.sum(np.diff(recall) * precision[1:]))


def per_image_class_records(
    pred_by_image: dict[str, list[OBBDetection]],
    gt_by_image: dict[str, list[OBBDetection]],
    iou_threshold: float = 0.5,
    iou_mode: IoUMode = "obb",
) -> dict[str, dict[int, dict[str, Any]]]:
    """Match once and keep per-(image, class) confidences, hit flags and GT counts.

    Matching is the expensive part; keeping the records lets the bootstrap
    resample images without re-running IoU.
    """
    records: dict[str, dict[int, dict[str, Any]]] = {}
    for image_id in sorted(set(pred_by_image) | set(gt_by_image)):
        summary = match_predictions_to_gt(
            pred_by_image.get(image_id, []),
            gt_by_image.get(image_id, []),
            iou_threshold=iou_threshold,
            iou_mode=iou_mode,
        )
        by_class: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"conf": [], "correct": [], "num_gt": 0}
        )
        for outcome in summary.outcomes:
            entry = by_class[outcome.prediction.class_id]
            entry["conf"].append(outcome.prediction.confidence or 0.0)
            entry["correct"].append(1.0 if outcome.is_tp else 0.0)
        for gt in gt_by_image.get(image_id, []):
            by_class[gt.class_id]["num_gt"] += 1
        records[image_id] = dict(by_class)
    return records


def map50_from_records(records: dict[str, dict[int, dict[str, Any]]], image_ids: list[str]) -> float:
    """mAP50 over the classes present in ``image_ids`` (mean of per-class AP)."""
    pooled: dict[int, dict[str, Any]] = defaultdict(lambda: {"conf": [], "correct": [], "num_gt": 0})
    for image_id in image_ids:
        for class_id, entry in records.get(image_id, {}).items():
            pooled[class_id]["conf"].extend(entry["conf"])
            pooled[class_id]["correct"].extend(entry["correct"])
            pooled[class_id]["num_gt"] += entry["num_gt"]

    aps = [
        _average_precision(
            np.asarray(entry["conf"], dtype=np.float64),
            np.asarray(entry["correct"], dtype=np.float64),
            entry["num_gt"],
        )
        for entry in pooled.values()
        if entry["num_gt"] > 0
    ]
    return float(np.mean(aps)) if aps else float("nan")


def bootstrap_map50(
    records: dict[str, dict[int, dict[str, Any]]],
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Point estimate and percentile CI for mAP50, resampling images with replacement."""
    image_ids = sorted(records)
    rng = np.random.default_rng(seed)
    point = map50_from_records(records, image_ids)

    draws = [
        map50_from_records(records, list(rng.choice(image_ids, size=len(image_ids), replace=True)))
        for _ in range(n_resamples)
    ]
    finite = np.asarray([d for d in draws if np.isfinite(d)], dtype=np.float64)
    if finite.size == 0:
        return {"map50": point, "ci_low": float("nan"), "ci_high": float("nan"), "sd": float("nan")}
    return {
        "map50": point,
        "ci_low": float(np.percentile(finite, 2.5)),
        "ci_high": float(np.percentile(finite, 97.5)),
        "sd": float(finite.std(ddof=1)),
    }


def paired_delta_map50(
    records_a: dict[str, dict[int, dict[str, Any]]],
    records_b: dict[str, dict[int, dict[str, Any]]],
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """CI for mAP50(b) - mAP50(a), resampling the SAME images for both arms.

    Pairing removes the image-difficulty variance the two arms share, which is
    most of it — an unpaired comparison on 24 images cannot separate anything.
    """
    image_ids = sorted(set(records_a) & set(records_b))
    rng = np.random.default_rng(seed)
    point = map50_from_records(records_b, image_ids) - map50_from_records(records_a, image_ids)

    draws = []
    for _ in range(n_resamples):
        sample = list(rng.choice(image_ids, size=len(image_ids), replace=True))
        delta = map50_from_records(records_b, sample) - map50_from_records(records_a, sample)
        if np.isfinite(delta):
            draws.append(delta)

    arr = np.asarray(draws, dtype=np.float64)
    if arr.size == 0:
        return {"delta": point, "ci_low": float("nan"), "ci_high": float("nan"), "p_worse": float("nan")}
    return {
        "delta": point,
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "p_worse": float((arr < 0).mean()),
    }
