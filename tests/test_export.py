from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from qgeocompress.compression.structural_collapse import collapse_structural_conv
from qgeocompress.compression.structural_low_rank import StructuralLowRankConv2d
from qgeocompress.models.export_core import (
    _normalize_formats,
    classify_checkpoint,
    run_multi_format_export,
    try_collapsed_fallback,
)


def test_normalize_formats_all():
    assert _normalize_formats("all") == {"pt", "onnx", "torchscript"}
    assert _normalize_formats(["pt", "onnx"]) == {"pt", "onnx"}


def test_collapse_structural_conv_reconstructs_weight():
    base = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=True)
    structural = StructuralLowRankConv2d(base, rank=2)
    collapsed = collapse_structural_conv(structural)

    x = torch.randn(1, 4, 16, 16)
    assert torch.allclose(structural(x), collapsed(x), atol=1e-5)


@patch("qgeocompress.models.export_core.classify_checkpoint")
def test_run_multi_format_export_pt_only(mock_classify, tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights")
    out = tmp_path / "export"
    mock_classify.return_value = {
        "structural": False,
        "model_type": "baseline",
        "ultralytics_compatible": True,
    }

    manifest = run_multi_format_export(weights, out, formats=["pt"])

    assert manifest["formats"]["pt"]["success"] is True
    assert (out / "export_model.pt").exists()
    assert manifest["formats"]["onnx"]["attempted"] is False
    assert manifest["collapsed_fallback"] is None


@patch("qgeocompress.models.export_core.try_onnx_export")
@patch("qgeocompress.models.export_core.classify_checkpoint")
def test_run_multi_format_export_onnx_attempt(mock_classify, mock_onnx, tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights")
    out = tmp_path / "export"
    mock_classify.return_value = {
        "structural": False,
        "model_type": "baseline",
        "ultralytics_compatible": True,
    }
    mock_onnx.return_value = {
        "attempted": True,
        "success": True,
        "path": str(out / "export_model.onnx"),
        "reason": None,
    }

    manifest = run_multi_format_export(weights, out, formats=["pt", "onnx"])

    assert manifest["formats"]["onnx"]["success"] is True
    assert manifest["onnx"]["success"] is True


@patch("qgeocompress.models.export_core.try_collapsed_fallback")
@patch("qgeocompress.models.export_core.classify_checkpoint")
def test_structural_triggers_collapsed_fallback(mock_classify, mock_collapsed, tmp_path):
    weights = tmp_path / "structural.pt"
    weights.write_bytes(b"weights")
    out = tmp_path / "export"
    mock_classify.return_value = {
        "structural": True,
        "model_type": "structural-low-rank",
        "ultralytics_compatible": False,
    }
    mock_collapsed.return_value = {
        "attempted": True,
        "success": True,
        "collapse_meta": {"num_collapsed_layers": 3},
        "collapsed_pt": {"attempted": True, "success": True, "path": str(out / "export_model_collapsed.pt")},
        "collapsed_onnx": {"attempted": True, "success": False, "path": None, "reason": "mock"},
        "collapsed_torchscript": {"attempted": False, "success": False, "path": None, "reason": "n/a"},
    }

    manifest = run_multi_format_export(weights, out, formats="all")

    assert manifest["structural"] is True
    mock_collapsed.assert_called_once()
    assert manifest["collapsed_fallback"]["success"] is True


def test_classify_checkpoint_handles_invalid_weights(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not-a-checkpoint")
    info = classify_checkpoint(bad)
    assert info["model_type"] == "unknown"
    assert info["structural"] is False


@patch("qgeocompress.models.export_core._has_structural_layers", return_value=True)
@patch("ultralytics.YOLO")
@patch("qgeocompress.compression.structural_collapse.collapse_structural_model")
def test_try_collapsed_fallback_saves_pt(mock_collapse, mock_yolo_cls, _mock_structural, tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"x")
    out = tmp_path / "out"
    out.mkdir()

    mock_model = MagicMock()
    collapsed_pt = out / "export_model_collapsed.pt"
    mock_model.save = MagicMock(side_effect=lambda p: Path(p).write_bytes(b"collapsed"))
    mock_collapse.return_value = (mock_model, {"num_collapsed_layers": 2})
    mock_yolo_cls.return_value = mock_model

    with patch("qgeocompress.models.export_core.try_onnx_export") as mock_onnx:
        mock_onnx.return_value = {"attempted": True, "success": False, "path": None, "reason": "mock"}
        result = try_collapsed_fallback(weights, out, try_torchscript=False)

    assert result["success"] is True
    assert result["collapsed_pt"]["success"] is True
    assert collapsed_pt.exists()
