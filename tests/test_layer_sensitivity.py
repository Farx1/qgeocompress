
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from qgeocompress.compression import layer_sensitivity
from qgeocompress.compression.layer_sensitivity import (
    compute_compressibility_score,
    layer_param_gain_pct,
    measure_local_output_error,
    probe_single_layer,
    run_structural_probe,
    select_layers_by_sensitivity,
)
from qgeocompress.compression.structural_low_rank import (
    StructuralLowRankConv2d,
    apply_structural_low_rank,
)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


def test_compressibility_score_orders_layers():
    good = compute_compressibility_score(40.0, 0.02, 0.01)
    bad = compute_compressibility_score(40.0, 0.5, 0.4)
    assert good > bad


def test_select_layers_by_sensitivity():
    rows = [
        {"layer_name": "a", "compressibility_score": 1.0},
        {"layer_name": "b", "compressibility_score": 9.0},
        {"layer_name": "c", "compressibility_score": 5.0},
    ]
    assert select_layers_by_sensitivity(rows, 2) == ["b", "c"]


def test_select_layers_by_pareto_gain_prefers_large_savings():
    from qgeocompress.compression.layer_sensitivity import (
        _param_gain_abs,
        select_layers_by_pareto_gain,
    )

    rows = [
        {"layer_name": "small", "map50_drop": 0.0, "param_gain_abs": 100},
        {"layer_name": "big", "map50_drop": 0.02, "param_gain_abs": 50000},
        {"layer_name": "bad", "map50_drop": 0.10, "param_gain_abs": 90000},
    ]
    assert select_layers_by_pareto_gain(rows, 2) == ["big", "small"]
    legacy = {"layer_name": "legacy", "params_before": 1000, "params_after": 200, "map50_drop": 0.0}
    assert _param_gain_abs(legacy) == 800


def test_layer_param_gain_positive():
    conv = nn.Conv2d(32, 64, 3, padding=1)
    before, after, gain = layer_param_gain_pct(conv, 0.5)
    assert after < before
    assert gain > 0


def test_local_output_error_bounded():
    conv = nn.Conv2d(16, 16, 3, padding=1)
    err = measure_local_output_error(conv, 1.0, [torch.randn(2, 16, 16, 16)])
    assert err is not None
    assert 0.0 <= err < 0.15


def test_capture_layer_inputs_feeds_real_activations(tmp_path, monkeypatch):
    """Deep convs must be scored on their own activations, not on the raw image."""
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.Conv2d(8, 16, 3, padding=1),
    )
    monkeypatch.setattr(
        layer_sensitivity, "_load_bchw_tensor", lambda *_a, **_k: torch.randn(1, 3, 32, 32)
    )
    captured = layer_sensitivity.capture_layer_inputs(model, ["0", "1"], ["fake.jpg"])

    assert captured["0"][0].shape[1] == 3
    assert captured["1"][0].shape[1] == 8  # would be 3 (and crash the conv) with raw images
    err = measure_local_output_error(model[1], 0.5, captured["1"])
    assert err is not None and err > 0.0


def test_apply_structural_max_replaced_layers():
    class _YOLO:
        def __init__(self):
            self.model = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.Conv2d(8, 16, 3, padding=1),
                nn.Conv2d(16, 32, 3, padding=1),
            )

    yolo = _YOLO()
    out, meta = apply_structural_low_rank(
        yolo,
        rank_ratio=0.5,
        target_layers=["0"],
        max_replaced_layers=1,
    )
    assert meta["num_replaced_layers"] == 1


def test_apply_structural_sensitivity_selection():
    class _YOLO:
        def __init__(self):
            self.model = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.Conv2d(8, 16, 3, padding=1),
            )

    report = {
        "layers": [
            {"layer_name": "1", "compressibility_score": 10.0},
            {"layer_name": "0", "compressibility_score": 2.0},
        ]
    }
    yolo = _YOLO()
    _, meta = apply_structural_low_rank(
        yolo,
        rank_ratio=0.5,
        target_layers=["0", "1"],
        selection_strategy="sensitivity",
        sensitivity_report=report,
        max_replaced_layers=1,
    )
    assert meta["num_replaced_layers"] == 1
    assert meta["selected_layer_names"][0] == "1"


def test_probe_single_layer_json_fields():
    conv = nn.Conv2d(16, 32, 3, padding=1)
    row = probe_single_layer(
        layer_name="model.3.conv",
        conv=conv,
        rank_ratio=0.84,
        baseline_map50=0.95,
        map50_after=0.93,
        local_output_error=0.04,
    )
    assert row["layer_name"] == "model.3.conv"
    assert row["param_gain_pct"] > 0
    assert row["map50_drop"] == pytest.approx(0.02, abs=1e-6)
    assert "compressibility_score" in row


def test_run_structural_probe_restores_between_layers(tmp_path):
    class _YOLO:
        def __init__(self):
            self.model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1))
            self.eval_calls = 0

        def predict(self, *args, **kwargs):
            return []

    yolo = _YOLO()
    yolo_factory = [yolo]

    def _factory(*args, **kwargs):
        return yolo_factory[0]

    def _eval(m):
        yolo.eval_calls += 1
        conv = m.model[0]
        if yolo.eval_calls == 1:
            assert isinstance(conv, nn.Conv2d)
            return 0.95
        assert isinstance(conv, StructuralLowRankConv2d)
        return 0.9

    with patch("qgeocompress.compression.layer_sensitivity.YOLO", side_effect=_factory):
        with patch("qgeocompress.compression.layer_sensitivity.list_val_images", return_value=[]):
            report = run_structural_probe(
                tmp_path / "w.pt",
                eval_map50_fn=_eval,
                target_layers=["0"],
                max_probe_layers=1,
            )

    assert report["num_candidates"] == 1
    assert len(report["layers"]) == 1
    assert isinstance(yolo.model[0], nn.Conv2d)
    assert yolo.eval_calls == 2


def test_sensitivity_report_json_roundtrip(tmp_path):
    from qgeocompress.compression.layer_sensitivity import (
        load_sensitivity_report,
        save_sensitivity_report,
    )

    report = {
        "method": "structural-probe",
        "layers": [{"layer_name": "model.0.conv", "compressibility_score": 5.0}],
    }
    path = save_sensitivity_report(report, tmp_path / "probe.json")
    loaded = load_sensitivity_report(path)
    assert loaded["layers"][0]["layer_name"] == "model.0.conv"
    assert "compressibility_score" in loaded["layers"][0]


def test_batchnorm_recalibration_defaults_to_a_light_touch():
    """Each batch decays the pretrained statistics by (1 - momentum).

    Regression: the old default of 20-24 single-image batches left the
    pretrained running statistics 8% of their weight and cost 0.077 mAP50 on a
    model with no compression applied at all.
    """
    import inspect

    default = inspect.signature(layer_sensitivity.recalibrate_batchnorm).parameters["num_batches"]
    assert default.default <= 8
    # Retained weight of the pretrained statistics at the default, momentum 0.1.
    assert (1 - 0.1) ** default.default > 0.5
