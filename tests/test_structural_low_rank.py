from unittest.mock import patch

import torch
import torch.nn as nn

from qgeocompress.compression import compress_model
from qgeocompress.compression.structural_low_rank import (
    StructuralLowRankConv2d,
    apply_structural_low_rank,
)


class _DummyYOLO:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.Conv2d(16, 32, 3, padding=1),
        )

    def save(self, path: str) -> None:
        torch.save({"model": self.model.state_dict()}, path)


def test_structural_low_rank_conv_forward_shape():
    conv = nn.Conv2d(8, 16, 3, padding=1)
    block = StructuralLowRankConv2d(conv, rank=4)
    x = torch.randn(2, 8, 32, 32)
    y = block(x)
    assert y.shape == (2, 16, 32, 32)


def test_structural_low_rank_reduces_params():
    conv = nn.Conv2d(64, 128, 3, padding=1)
    original = sum(p.numel() for p in conv.parameters())
    block = StructuralLowRankConv2d(conv, rank=8)
    factored = sum(p.numel() for p in block.parameters())
    assert factored < original


def test_apply_structural_low_rank_replaces_layers():
    dummy = _DummyYOLO()
    model, meta = apply_structural_low_rank(dummy, rank_ratio=0.5, target_layers=["0"])
    assert meta["num_replaced_layers"] >= 1
    assert meta["structural_compression"] is True
    assert meta["compressed_params"] < meta["original_params"]
    assert meta["real_param_reduction_pct"] > 0
    x = torch.randn(1, 3, 32, 32)
    y = model.model(x)
    assert y.shape[1] == 32


def test_compress_model_structural_probe(tmp_path):
    weights = tmp_path / "fake.pt"
    torch.save({"model": nn.Conv2d(3, 8, 3).state_dict()}, weights)

    with patch("qgeocompress.compression.layer_sensitivity.YOLO") as mock_yolo:
        with patch("qgeocompress.compression.layer_sensitivity.list_val_images", return_value=[]):
            mock_yolo.return_value = _DummyYOLO()
            meta, model = compress_model(
                weights,
                "structural-probe",
                tmp_path / "out",
                rank_ratio=0.84,
                target_layers=["0"],
                eval_map50_fn=lambda m: 0.9,
                max_probe_layers=1,
            )

    assert meta["status"] == "ok"
    assert meta["num_candidates"] == 1
    assert model is None
    assert "sensitivity_report" in meta


def test_compress_model_structural_low_rank_json(tmp_path):
    weights = tmp_path / "fake.pt"
    torch.save({"model": nn.Conv2d(3, 8, 3).state_dict()}, weights)

    with patch("qgeocompress.compression.YOLO") as mock_yolo:
        mock_yolo.return_value = _DummyYOLO()
        meta, model = compress_model(
            weights,
            "structural-low-rank",
            tmp_path / "out",
            rank_ratio=0.5,
            target_layers=["0"],
        )

    assert meta["status"] == "ok"
    assert meta["structural_compression"] is True
    assert meta["real_param_reduction_pct"] > 0
    assert meta["ultralytics_compatible"] is True
    assert model is not None
    assert (tmp_path / "out" / "structural_low_rank_r0.500.pt").exists()
