import json
from unittest.mock import patch

from qgeocompress.cli.make_calibration_report import (
    ECE_INTERPRETATION_NOTE,
    filter_phase2b_runs,
    generate_calibration_report,
    make_calibration_table,
    operational_reliability_status,
)


def test_filter_phase2b_runs():
    runs = [
        {"compression": "baseline", "run_id": "calibration_baseline"},
        {"compression": "low-rank", "rank_ratio": 0.88, "run_id": "a"},
        {"compression": "low-rank", "rank_ratio": 0.95, "run_id": "b"},
        {"compression": "low-rank", "rank_ratio": 0.84, "run_id": "c"},
    ]
    selected = filter_phase2b_runs(runs, ranks=[0.88, 0.84])
    assert len(selected) == 3
    assert selected[0]["compression"] == "baseline"


def test_make_calibration_table_columns():
    baseline = {"map50": 0.954, "confident_error_rate_08": 0.0223}
    runs = [
        {"run_id": "x", "compression": "baseline", "map50": 0.954, "ece": 0.04,
         "confident_error_rate_08": 0.0223},
        {"run_id": "y", "compression": "low-rank", "rank_ratio": 0.84, "map50": 0.841, "ece": 0.03,
         "confident_error_rate_08": 0.0256},
    ]
    df = make_calibration_table(runs, baseline=baseline)
    assert "operational_reliability_status" in df.columns
    assert "map50_drop_vs_baseline" in df.columns
    assert "cer08_delta_vs_baseline" in df.columns
    assert df.loc[df["label"] == "r=0.84", "operational_reliability_status"].iloc[0] == "viable"


def test_operational_reliability_status_phase2b_numbers():
    base_cer = 0.0223
    assert operational_reliability_status(0.9538, 0.0223, base_cer) == "safe"
    assert operational_reliability_status(0.9081, 0.0261, base_cer) == "safe"
    assert operational_reliability_status(0.8413, 0.0256, base_cer) == "viable"
    assert operational_reliability_status(0.8397, 0.0292, base_cer) == "viable"
    assert operational_reliability_status(0.7049, 0.0424, base_cer) == "degraded"


def test_operational_status_degraded_when_cer_too_high():
    base_cer = 0.0223
    assert operational_reliability_status(0.90, 0.10, base_cer) == "degraded"


def test_summary_includes_ece_caveat(tmp_path):
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    from qgeocompress.cli.make_calibration_report import render_calibration_summary_markdown

    md = render_calibration_summary_markdown(
        [{"compression": "baseline", "map50": 0.95, "ece": 0.03, "confident_error_rate_08": 0.02}],
        {"confident_error_rate_08": 0.02},
    )
    assert ECE_INTERPRETATION_NOTE.split(".")[0] in md
    assert "operational status" in md


def test_generate_calibration_report(tmp_path):
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    runs = [
        {"run_id": "calibration_baseline_abcd", "compression": "baseline", "map50": 0.954, "ece": 0.04,
         "confident_error_rate_08": 0.03, "reliability_bins": [{"avg_confidence": 0.5, "accuracy": 0.5, "count": 10}]},
        {"run_id": "calibration_low_rank_r0.840_abcd", "compression": "low-rank", "rank_ratio": 0.84,
         "map50": 0.841, "ece": 0.087, "confident_error_rate_08": 0.09,
         "reliability_bins": [{"avg_confidence": 0.6, "accuracy": 0.5, "count": 10}]},
    ]
    for run in runs:
        (summaries / f"{run['run_id']}.json").write_text(json.dumps(run))

    with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
        report = generate_calibration_report(summaries, ranks=[0.84])

    assert report["num_included"] == 2
    assert (summaries / "calibration_comparison_table.csv").exists()
    assert (summaries / "calibration_summary.md").exists()
    assert report.get("map50_vs_cer08_figure")
