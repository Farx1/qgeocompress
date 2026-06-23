import numpy as np
import torch
import torch.nn as nn
from unittest.mock import patch

from qgeocompress.compression import compress_model
from qgeocompress.compression.low_rank import LowRankLinear, apply_low_rank
from qgeocompress.compression.pruning import magnitude_prune
from qgeocompress.evaluation.report import generate_report, is_evaluable_run, make_comparison_table


class _DummyYOLO:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.Flatten(),
            nn.Linear(8 * 32 * 32, 16),
            nn.Linear(16, 4),
        )

    def save(self, path: str) -> None:
        torch.save({"model": self.model.state_dict()}, path)


def test_low_rank_linear_output_shape():
    linear = nn.Linear(32, 16)
    lr = LowRankLinear(linear, rank=4)
    x = torch.randn(2, 32)
    assert lr(x).shape == (2, 16)


def test_low_rank_replaces_layers():
    dummy = _DummyYOLO()
    out, meta = apply_low_rank(dummy, rank_ratio=0.5, target_layers=["0"])
    assert meta["num_replaced_layers"] >= 1
    assert meta["compression_mode"] == "svd_in_place"
    x = torch.randn(1, 3, 32, 32)
    y = out.model(x)
    assert y.shape[-1] == 4


def test_magnitude_pruning_runs():
    dummy = _DummyYOLO()
    out, meta = magnitude_prune(dummy, sparsity=0.3)
    assert meta["layers"] >= 1


def test_fp16_skipped_without_cuda(tmp_path):
    weights = tmp_path / "fake.pt"
    torch.save({"model": nn.Linear(4, 2).state_dict()}, weights)

    with patch("qgeocompress.compression.torch.cuda.is_available", return_value=False):
        with patch("qgeocompress.compression.YOLO") as mock_yolo:
            mock_yolo.return_value = _DummyYOLO()
            meta, model = compress_model(weights, "fp16", tmp_path)

    assert meta["status"] == "skipped"
    assert meta["ultralytics_compatible"] is False
    assert model is None
    assert "CUDA" in meta["reason"]


def test_int8_unsupported(tmp_path):
    weights = tmp_path / "fake.pt"
    torch.save({"model": nn.Linear(4, 2).state_dict()}, weights)

    with patch("qgeocompress.compression.YOLO") as mock_yolo:
        mock_yolo.return_value = _DummyYOLO()
        meta, model = compress_model(weights, "int8-ptq", tmp_path)

    assert meta["status"] == "unsupported_for_ultralytics_eval"
    assert meta["ultralytics_compatible"] is False
    assert model is None


def test_report_handles_skipped_runs(tmp_path):
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    runs = [
        {
            "run_id": "baseline_x",
            "compression": "baseline",
            "map50": 0.95,
            "map50_95": 0.79,
            "latency_ms_mean": 47.0,
            "throughput_img_s": 21.0,
            "vram_mb_max": 0.0,
            "model_size_mb": 5.6,
        },
        {
            "run_id": "fp16_x",
            "compression": "fp16",
            "status": "skipped",
            "reason": "FP16 meaningful benchmark requires CUDA",
            "ultralytics_compatible": False,
            "map50": None,
            "latency_ms_mean": None,
        },
        {
            "run_id": "lowrank_x",
            "compression": "low-rank",
            "status": "ok",
            "ultralytics_compatible": True,
            "map50": 0.93,
            "map50_95": 0.76,
            "latency_ms_mean": 39.0,
            "throughput_img_s": 25.0,
            "num_replaced_layers": 12,
            "param_reduction_pct": 18.5,
            "model_size_mb": 4.8,
        },
    ]
    for i, run in enumerate(runs):
        (summaries / f"run_{i}.json").write_text(__import__("json").dumps(run))

    report = generate_report(summaries)
    assert report["num_runs"] == 3
    assert report["num_evaluable"] == 2
    assert report["num_skipped"] == 1

    df = make_comparison_table(runs)
    assert "status" in df.columns
    assert not bool(df.loc[df["compression"] == "fp16", "evaluable"].iloc[0])
    assert not is_evaluable_run(runs[1])
