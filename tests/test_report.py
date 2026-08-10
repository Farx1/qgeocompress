import json
from unittest.mock import patch

import pytest

from qgeocompress.evaluation.report import (
    FINETUNE_EXCLUSION_NOTE,
    ReportOptions,
    detect_rupture_rank,
    effective_map50,
    filter_runs_for_report,
    generate_phase2a_summary,
    generate_report,
    is_finetune_run,
    low_rank_sweep_runs,
    normalize_run_for_report,
    plot_rank_ratio_vs_map50,
    render_phase2a_summary_markdown,
)


def _baseline_run(map50: float = 0.954) -> dict:
    return {"run_id": "baseline_x", "compression": "baseline", "map50": map50, "map50_95": 0.793}


def _low_rank(rank: float, map50: float, finetune_epochs: int = 0, status: str = "ok") -> dict:
    return {
        "run_id": f"lr_{rank}",
        "compression": "low-rank",
        "compression_mode": "svd_in_place",
        "status": status,
        "rank_ratio": rank,
        "finetune_epochs": finetune_epochs,
        "map50": map50,
        "map50_before_finetune": map50 if finetune_epochs else None,
        "map50_95": map50 * 0.8,
        "latency_ms_mean": 40.0,
        "param_reduction_pct": 1.5,
    }


def test_normalize_baseline_status_ok():
    run = normalize_run_for_report({"compression": "baseline", "map50": 0.95})
    assert run["status"] == "ok"


def test_normalize_svd_param_reduction_alias():
    run = normalize_run_for_report(_low_rank(0.88, 0.908))
    assert run["theoretical_rank_reduction_pct"] == 1.5


def test_finetune_runs_excluded_by_default():
    runs = [_baseline_run(), _low_rank(0.75, 0.696, finetune_epochs=1)]
    included, excluded = filter_runs_for_report(runs)
    assert len(included) == 1
    assert len(excluded) == 1
    assert excluded[0]["_exclusion_reason"] == "finetune_run_excluded"


def test_include_unsupported_finetune_flag():
    runs = [_baseline_run(), _low_rank(0.75, 0.696, finetune_epochs=1, status="unsupported_train_api_state_loss")]
    opts = ReportOptions(include_unsupported_finetune=True, exclude_finetune_runs=False)
    included, excluded = filter_runs_for_report(runs, opts)
    assert len(included) == 2
    assert len(excluded) == 0


def test_unsupported_status_excluded_without_flag():
    runs = [_low_rank(0.8, 0.705, status="unsupported_train_api_state_loss")]
    included, excluded = filter_runs_for_report(runs)
    assert len(included) == 0
    assert excluded[0]["status"] == "unsupported_train_api_state_loss"


def test_effective_map50_ignores_post_finetune():
    run = _low_rank(0.75, 0.015, finetune_epochs=1)
    run["map50_after_finetune"] = 0.015
    run["map50_before_finetune"] = 0.696
    assert effective_map50(run) == 0.696


def test_detect_rupture_rank():
    sweep = low_rank_sweep_runs(
        [
            _low_rank(0.95, 0.921),
            _low_rank(0.84, 0.841),
            _low_rank(0.82, 0.840),
            _low_rank(0.80, 0.705),
        ]
    )
    rupture = detect_rupture_rank(sweep, baseline_map50=0.954)
    assert rupture == pytest.approx(0.82, abs=0.01)


def test_phase2a_summary_content():
    runs = [
        _baseline_run(),
        _low_rank(0.90, 0.908),
        _low_rank(0.84, 0.841),
        _low_rank(0.82, 0.840),
        _low_rank(0.80, 0.705),
    ]
    summary = generate_phase2a_summary(runs, excluded=[])
    assert summary["best_safe_rank_ratio"] == 0.90
    assert summary["best_usable_rank_ratio"] == 0.82
    md = render_phase2a_summary_markdown(summary)
    assert "Phase 2A" in md
    assert FINETUNE_EXCLUSION_NOTE.split(".")[0] in md or "Ultralytics train()" in md


def test_plot_rank_ratio_vs_map50(tmp_path):
    runs = [_baseline_run(), _low_rank(0.90, 0.908), _low_rank(0.80, 0.705)]
    out = tmp_path / "rank.png"
    with patch("matplotlib.pyplot.savefig"):
        with patch("matplotlib.pyplot.close"):
            path = plot_rank_ratio_vs_map50(runs, output_path=out)
    assert path == out


def test_generate_report_excludes_finetune_and_writes_summary(tmp_path):
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    runs = [
        _baseline_run(),
        _low_rank(0.88, 0.908),
        _low_rank(0.75, 0.696, finetune_epochs=1),
        {"run_id": "fp16", "compression": "fp16", "status": "skipped", "reason": "cpu"},
    ]
    for i, run in enumerate(runs):
        (summaries / f"run_{i}.json").write_text(json.dumps(run))

    report = generate_report(summaries)
    assert report["num_included"] == 2
    assert report["num_excluded"] == 2
    assert (summaries / "phase2a_summary.md").exists()
    assert report["phase2a_summary"]["num_finetune_runs_excluded"] >= 1

    table = (summaries / "comparison_table.csv").read_text()
    assert "theoretical_rank_reduction_pct" in table
    assert "param_reduction_pct" not in table
    assert "+ft" not in table

    final = json.loads((summaries / "final_report.json").read_text())
    assert final["finetune_exclusion_note"]
    assert any("finetune" in (r.get("reason") or "") for r in final["excluded_runs"])


def test_is_finetune_run():
    assert is_finetune_run({"finetune_epochs": 1})
    assert not is_finetune_run({"finetune_epochs": 0})
