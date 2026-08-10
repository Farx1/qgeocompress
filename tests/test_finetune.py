import json
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from qgeocompress.cli import compress_model as compress_cli
from qgeocompress.compression.finetune import (
    FinetuneConfig,
    compute_map50_recovery,
    compute_reload_drop,
    reload_is_unstable,
    select_finetune_checkpoint,
)
from qgeocompress.evaluation.report import effective_map50, generate_report, run_display_label


def test_compute_map50_recovery():
    recovery = compute_map50_recovery(0.954, 0.001, 0.700)
    assert recovery == pytest.approx(73.34, abs=0.1)


def test_compute_map50_recovery_no_gap():
    assert compute_map50_recovery(0.95, 0.95, 0.94) == 0.0


def test_effective_map50_uses_before_finetune_for_ft_runs():
    run = {
        "map50": 0.1,
        "map50_before_finetune": 0.696,
        "map50_after_finetune": 0.015,
        "finetune_epochs": 1,
    }
    assert effective_map50(run) == 0.696


def test_run_display_label_finetuned():
    label = run_display_label({"compression": "low-rank", "rank_ratio": 0.5, "finetune_epochs": 3})
    assert label == "low-rank r=0.5 +ft3"


def test_finetune_epochs_zero_skips_training(tmp_path):
    args = Namespace(
        method="low-rank",
        weights=tmp_path / "w.pt",
        dataset="dota128",
        model="yolo_obb_small",
        output_dir=tmp_path / "out",
        rank_ratio=0.5,
        target_layers=["backbone", "neck"],
        finetune_epochs=0,
        finetune_lr0=1e-5,
        finetune_lrf=0.1,
        finetune_freeze=None,
        finetune_use_last=False,
        imgsz=640,
        seed=42,
        device="cpu",
    )
    meta = {
        "status": "ok",
        "ultralytics_compatible": True,
        "output_weights": str(tmp_path / "out" / "low_rank_r0.500.pt"),
        "num_replaced_layers": 29,
        "param_reduction_pct": 17.6,
    }
    mock_model = MagicMock()
    metrics = {"map50": 0.001, "map50_95": 0.0001}

    with patch("qgeocompress.cli.compress_model.compress_model", return_value=(meta, mock_model)):
        with patch("qgeocompress.cli.compress_model._evaluate_model", return_value=(metrics, {"latency_ms_mean": 40.0})):
            summary = compress_cli._run_low_rank_pipeline(args, "cpu", {"name": "yolo_obb_small"}, MagicMock())

    assert summary["finetune_epochs"] == 0
    assert summary["map50_before_finetune"] == 0.001
    assert summary["map50_after_finetune"] is None


def test_finetune_epochs_positive_skips_unsupported_train_api(tmp_path):
    args = Namespace(
        method="low-rank",
        weights=tmp_path / "w.pt",
        dataset="dota128",
        model="yolo_obb_small",
        output_dir=tmp_path / "out",
        rank_ratio=0.5,
        target_layers=["backbone", "neck"],
        finetune_epochs=3,
        finetune_lr0=1e-5,
        finetune_lrf=0.1,
        finetune_freeze=None,
        finetune_use_last=False,
        imgsz=640,
        seed=42,
        device="cpu",
    )
    meta = {
        "status": "ok",
        "ultralytics_compatible": True,
        "output_weights": str(tmp_path / "out" / "low_rank_r0.500.pt"),
    }
    mock_model = MagicMock()
    before = {"map50": 0.696, "map50_95": 0.566}

    with patch("qgeocompress.cli.compress_model.compress_model", return_value=(meta, mock_model)):
        with patch(
            "qgeocompress.cli.compress_model._evaluate_model",
            side_effect=[
                (before, {}),
                (before, {"latency_ms_mean": 38.0}),
            ],
        ):
            with patch(
                "qgeocompress.cli.compress_model._evaluate_weights",
                return_value={"map50": 0.696, "map50_95": 0.566},
            ):
                summary = compress_cli._run_low_rank_pipeline(
                    args, "cpu", {"name": "yolo_obb_small"}, MagicMock()
                )

    assert summary["finetune_epochs"] == 3
    assert summary["status"] == "unsupported_train_api_state_loss"
    assert summary["train_state_preserved"] is False
    assert summary["finetune_backend"] == "ultralytics_train_api"
    assert summary["map50_before_finetune"] == 0.696
    assert summary["map50_after_finetune"] is None
    assert summary["reload_drop"] == 0.0


def test_finetune_aborts_on_reload_instability(tmp_path):
    args = Namespace(
        method="low-rank",
        weights=tmp_path / "w.pt",
        dataset="dota128",
        model="yolo_obb_small",
        output_dir=tmp_path / "out",
        rank_ratio=0.75,
        target_layers=["backbone", "neck"],
        finetune_epochs=1,
        finetune_lr0=1e-5,
        finetune_lrf=0.1,
        finetune_freeze=None,
        finetune_use_last=False,
        imgsz=640,
        seed=42,
        device="cpu",
    )
    meta = {
        "status": "ok",
        "ultralytics_compatible": True,
        "output_weights": str(tmp_path / "out" / "low_rank_r0.750.pt"),
    }
    mock_model = MagicMock()
    before = {"map50": 0.696, "map50_95": 0.566}
    after_reload = {"map50": 0.015, "map50_95": 0.001}

    with patch("qgeocompress.cli.compress_model.compress_model", return_value=(meta, mock_model)):
        with patch(
            "qgeocompress.cli.compress_model._evaluate_model",
            side_effect=[(before, {}), (before, {"latency_ms_mean": 40.0})],
        ):
            with patch("qgeocompress.cli.compress_model._evaluate_weights", return_value=after_reload):
                summary = compress_cli._run_low_rank_pipeline(
                    args, "cpu", {"name": "yolo_obb_small"}, MagicMock()
                )

    assert summary["status"] == "failed_reload_instability"
    assert summary["reload_drop"] == pytest.approx(0.681, abs=1e-3)
    assert summary["map50_before_reload"] == 0.696
    assert summary["map50_after_reload_before_training"] == 0.015
    assert summary["fine_tune_aborted_reason"] is not None
    assert summary["map50_after_finetune"] is None


def test_reload_instability_matches_observed_failure():
    assert reload_is_unstable(0.696, 0.015)
    assert compute_reload_drop(0.696, 0.015) > 0.6

def test_select_finetune_checkpoint_prefers_higher_map(tmp_path):
    best = tmp_path / "best.pt"
    last = tmp_path / "last.pt"
    best.write_bytes(b"b")
    last.write_bytes(b"l")
    path, name = select_finetune_checkpoint(
        {"best": best, "last": last},
        {"map50": 0.0},
        {"map50": 0.68},
        use_last=False,
    )
    assert name == "last.pt"
    assert path == last


def test_select_finetune_checkpoint_forced_last(tmp_path):
    best = tmp_path / "best.pt"
    last = tmp_path / "last.pt"
    best.write_bytes(b"b")
    last.write_bytes(b"l")
    path, name = select_finetune_checkpoint(
        {"best": best, "last": last},
        {"map50": 0.9},
        {"map50": 0.1},
        use_last=True,
    )
    assert name == "last.pt"
    assert path == last


def test_finetune_cli_parses_lr0():
    from qgeocompress.cli.compress_model import _add_finetune_args

    parser = __import__("argparse").ArgumentParser()
    _add_finetune_args(parser)
    args = parser.parse_args(["--finetune-epochs", "3", "--finetune-lr0", "5e-5", "--finetune-use-last"])
    assert args.finetune_epochs == 3
    assert args.finetune_lr0 == 5e-5
    assert args.finetune_use_last is True


def test_finetune_config_defaults():
    cfg = FinetuneConfig()
    assert cfg.lr0 == 1e-5
    assert cfg.lrf == 0.1
    assert cfg.optimizer == "AdamW"


def test_report_handles_finetuned_runs(tmp_path):
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    runs = [
        {
            "run_id": "baseline_x",
            "compression": "baseline",
            "status": "ok",
            "map50": 0.954,
            "latency_ms_mean": 47.0,
            "throughput_img_s": 21.0,
        },
        {
            "run_id": "lr_ft",
            "compression": "low-rank",
            "status": "unsupported_train_api_state_loss",
            "rank_ratio": 0.5,
            "finetune_epochs": 3,
            "map50_before_finetune": 0.001,
            "map50_after_finetune": 0.72,
            "latency_ms_mean": 38.0,
            "num_replaced_layers": 29,
        },
    ]
    for i, run in enumerate(runs):
        (summaries / f"run_{i}.json").write_text(json.dumps(run))

    report = generate_report(summaries)
    assert report["num_included"] == 1
    assert report["num_excluded"] == 1
