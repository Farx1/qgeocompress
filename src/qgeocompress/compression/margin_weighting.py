"""Experimental image weighting; NOT a certified nonlinear decision bound."""

from __future__ import annotations

import math

import torch


def margin_priority(scores: torch.Tensor, threshold: float = 0.25, floor: float = 0.05, topk: int = 128) -> float:
    """Average clipped inverse squared threshold margin over top-scoring anchors.

    Scores must be maximum class probabilities before NMS, for one image.
    Clipping stabilizes calibration but removes the exact inverse-margin bound.
    """
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be finite and inside (0, 1)")
    if not math.isfinite(floor) or floor <= 0 or not isinstance(topk, int) or topk < 1:
        raise ValueError("floor must be positive and finite; topk must be a positive integer")
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a nonempty vector")
    if not torch.isfinite(scores).all() or torch.any((scores < 0) | (scores > 1)):
        raise ValueError("scores must be finite probabilities")
    selected = scores.double().topk(min(topk, scores.numel())).values
    return float((selected.sub(threshold).abs().clamp_min(floor).square().reciprocal()).mean())


def normalized_priorities(values: list[float]) -> torch.Tensor:
    """Positive image weights with mean one, accumulated in float64."""
    weights = torch.tensor(values, dtype=torch.float64)
    if weights.ndim != 1 or not weights.numel() or not torch.isfinite(weights).all() or torch.any(weights <= 0):
        raise ValueError("priorities must be a nonempty vector of finite positive values")
    weights = weights / weights.max()
    return weights / weights.mean()
