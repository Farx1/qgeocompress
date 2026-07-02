import json
from pathlib import Path

from qgeocompress.evaluation.phase3c_report import (
    build_phase3c_rows,
    render_phase3c_markdown,
    rows_to_csv,
)


def _write(summaries: Path, name: str, payload: dict) -> None:
    (summaries / name).write_text(json.dumps(payload), encoding="utf-8")


def test_build_phase3c_rows_joins_compress_cal_gate(tmp_path: Path):
    summaries = tmp_path / "summaries"
    summaries.mkdir()

    _write(
        summaries,
        "calibration_baseline_holdout_test_baseline_abcd1234.json",
        {
            "run_id": "calibration_baseline_holdout_test_baseline_abcd1234",
            "compression": "baseline",
            "run_label": "holdout_test_baseline",
            "map50": 0.954,
            "confident_error_rate_08": 0.022,
            "ece": 0.04,
            "status": "ok",
        },
    )
    _write(
        summaries,
        "structural-low-rank_11845535.json",
        {
            "run_id": "structural-low-rank_11845535",
            "run_label": "holdout_top5_pareto",
            "map50": 0.933,
            "real_param_reduction_pct": 5.63,
            "bn_recalibration_batches": 0,
            "status": "ok",
            "output_weights": "runs/compressed/structural_low_rank/top5.pt",
        },
    )
    _write(
        summaries,
        "calibration_structural_r0.840_holdout_top5_pareto_test_f215351d.json",
        {
            "run_id": "calibration_structural_r0.840_holdout_top5_pareto_test_f215351d",
            "compression": "structural-low-rank",
            "run_label": "holdout_top5_pareto_test",
            "map50": 0.933,
            "confident_error_rate_08": 0.005,
            "ece": 0.042,
            "status": "ok",
            "weights": "runs/compressed/structural_low_rank/top5.pt",
        },
    )
    _write(
        summaries,
        "reliability_gate_calibration_structural_r0.840_holdout_top5_pareto_test_f215351d.json",
        {
            "run_id": "reliability_gate_calibration_structural_r0.840_holdout_top5_pareto_test_f215351d",
            "candidate_run_id": "calibration_structural_r0.840_holdout_top5_pareto_test_f215351d",
            "gate_status": "deployable_compression_candidate",
        },
    )

    rows = build_phase3c_rows(summaries)
    methods = {r["method"] for r in rows}
    assert "Baseline (hold-out test)" in methods
    assert "Structural top-5 pareto-gain" in methods

    pareto = next(r for r in rows if r["run_label"] == "holdout_top5_pareto")
    assert pareto["map50"] == 0.933
    assert pareto["cer08"] == 0.005
    assert pareto["param_reduction_pct"] == 5.63
    assert pareto["gate_status"] == "deployable_compression_candidate"


def test_render_phase3c_markdown_includes_pending(tmp_path: Path):
    rows = [
        {
            "method": "Baseline (hold-out test)",
            "run_label": "holdout_test_baseline",
            "map50": 0.954,
            "cer08": 0.022,
            "ece": 0.04,
            "param_reduction_pct": 0.0,
            "bn_batches": None,
            "gate_status": "reference",
            "latency_ms_mean": None,
            "status": "ok",
        },
        {
            "method": "SVD in-place r=0.84 (hold-out test)",
            "run_label": "holdout_low_rank_r084",
            "map50": None,
            "cer08": None,
            "ece": None,
            "param_reduction_pct": None,
            "bn_batches": None,
            "gate_status": "pending",
            "latency_ms_mean": None,
            "status": "pending",
            "reproduce": "python scripts/compress_model.py --method low-rank",
        },
    ]
    md = render_phase3c_markdown(rows)
    assert "Phase 3C Comparison" in md
    assert "Pending experiments" in md
    assert "0.954" in md
    assert "low-rank" in md


def test_rows_to_csv_headers():
    csv = rows_to_csv([{"method": "x", "run_label": "y", "map50": 1.0, "status": "ok"}])
    assert csv.startswith("method,run_label,map50")
    assert "x,y,1.0" in csv
