"""Checks for the bootstrapped AP used to compare compression arms."""

from __future__ import annotations

import numpy as np

from qgeocompress.evaluation.ap import (
    _average_precision,
    bootstrap_map50,
    map50_from_records,
    paired_delta_map50,
)


def _records(per_image: dict[str, list[tuple[float, float]]], num_gt: int = 2):
    return {
        image: {0: {"conf": [c for c, _ in rows], "correct": [k for _, k in rows], "num_gt": num_gt}}
        for image, rows in per_image.items()
    }


def test_average_precision_is_one_for_perfect_ranking():
    conf = np.array([0.9, 0.8, 0.3])
    correct = np.array([1.0, 1.0, 0.0])
    assert _average_precision(conf, correct, num_gt=2) == 1.0


def test_average_precision_penalises_confident_false_positives():
    good = _average_precision(np.array([0.9, 0.4]), np.array([1.0, 0.0]), num_gt=1)
    bad = _average_precision(np.array([0.9, 0.4]), np.array([0.0, 1.0]), num_gt=1)
    assert bad < good


def test_missing_ground_truth_is_not_counted_as_a_perfect_class():
    assert np.isnan(_average_precision(np.array([0.9]), np.array([1.0]), num_gt=0))


def test_bootstrap_ci_brackets_the_point_estimate():
    records = _records({f"img{i}": [(0.9, 1.0), (0.5, 1.0), (0.2, 0.0)] for i in range(12)})
    out = bootstrap_map50(records, n_resamples=200, seed=1)
    assert out["ci_low"] <= out["map50"] <= out["ci_high"]
    assert out["map50"] == map50_from_records(records, sorted(records))


def test_paired_delta_is_tighter_than_the_arms_it_compares():
    """Pairing removes the shared image-difficulty variance — that is the point."""
    rng = np.random.default_rng(0)
    base, worse = {}, {}
    for i in range(20):
        # Image difficulty drives how many of the 4 boxes are found — shared by
        # both arms. The second arm always finds one fewer.
        hits = int(rng.integers(2, 5))
        confs = [0.95, 0.85, 0.75, 0.65]
        base[f"img{i}"] = {
            0: {"conf": confs, "correct": [1.0] * hits + [0.0] * (4 - hits), "num_gt": 4}
        }
        worse[f"img{i}"] = {
            0: {"conf": confs, "correct": [1.0] * (hits - 1) + [0.0] * (5 - hits), "num_gt": 4}
        }

    delta = paired_delta_map50(base, worse, n_resamples=300, seed=2)
    spread = bootstrap_map50(base, n_resamples=300, seed=2)

    assert delta["delta"] < 0
    assert delta["ci_high"] < 0  # the regression is detected despite image variance
    assert (delta["ci_high"] - delta["ci_low"]) < (spread["ci_high"] - spread["ci_low"])
