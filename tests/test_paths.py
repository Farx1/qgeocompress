from pathlib import Path

import pytest

from qgeocompress.compression.finetune import (
    RELOAD_DROP_ABS_THRESHOLD,
    compute_reload_drop,
    reload_is_unstable,
)
from qgeocompress.utils.paths import DEFAULT_BASELINE_WEIGHTS, resolve_baseline_weights


def test_compute_reload_drop():
    assert compute_reload_drop(0.696, 0.015) == pytest.approx(0.681, abs=1e-3)


def test_reload_is_unstable_large_drop():
    assert reload_is_unstable(0.696, 0.015) is True


def test_reload_is_unstable_small_drop():
    assert reload_is_unstable(0.921, 0.908) is False


def test_reload_threshold_constants():
    assert RELOAD_DROP_ABS_THRESHOLD == 0.10


def test_resolve_baseline_weights_explicit(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    assert resolve_baseline_weights(weights) == weights


def test_resolve_baseline_weights_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_baseline_weights(tmp_path / "missing.pt")


def test_default_baseline_path_documents_ultralytics_layout():
    assert "baseline" in str(DEFAULT_BASELINE_WEIGHTS)
    assert DEFAULT_BASELINE_WEIGHTS.name == "best.pt"
