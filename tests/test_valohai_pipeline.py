from pathlib import Path

from qgeocompress.valohai.gate_report import (
    build_gate_reasons,
    evaluate_quality_gate,
    render_final_report,
    valohai_gate_status,
)
from qgeocompress.valohai.io import outputs_dir, write_json


def test_valohai_gate_status_mapping():
    assert valohai_gate_status("deployable_compression_candidate") == "accepted_for_export"
    assert valohai_gate_status("methodologically_valid") == "accepted_for_research"
    assert valohai_gate_status("rejected") == "rejected"


def test_gate_from_comparison_metrics_only():
    comparison = {
        "baseline": {"map50": 0.929, "cer08": 0.004},
        "compressed": {"map50": 0.933, "cer08": 0.005},
        "delta": {"map50_drop": 0.0, "cer08_delta": 0.001, "param_reduction_pct": 5.63},
    }
    compression_summary = {"status": "ok", "real_param_reduction_pct": 5.63}
    gate = evaluate_quality_gate(comparison, compression_summary)
    assert gate["valohai_status"] == "accepted_for_export"
    assert gate["accepted_for_export"] is True
    assert "mAP drop acceptable" in gate["reasons"]


def test_gate_research_only_when_params_low():
    comparison = {
        "baseline": {"map50": 0.929, "cer08": 0.004},
        "compressed": {"map50": 0.879, "cer08": 0.005},
        "delta": {"map50_drop": 0.05, "cer08_delta": 0.001, "param_reduction_pct": 0.21},
    }
    gate = evaluate_quality_gate(comparison, {"status": "ok", "real_param_reduction_pct": 0.21})
    assert gate["valohai_status"] == "accepted_for_research"
    assert gate["accepted"] is True
    assert gate["accepted_for_export"] is False


def test_render_final_report_contains_status():
    gate = {"valohai_status": "accepted_for_export", "reasons": ["mAP drop acceptable"], "accepted_for_export": True}
    comparison = {
        "baseline": {"map50": 0.93, "cer08": 0.004, "ece": 0.04, "latency_ms_mean": 60},
        "compressed": {"map50": 0.93, "cer08": 0.005, "ece": 0.04, "latency_ms_mean": 58},
        "delta": {"map50_drop": 0.0, "param_reduction_pct": 5.0, "latency_speedup_pct": 3.3},
    }
    report = render_final_report(gate, comparison)
    assert "accepted_for_export" in report
    assert "mAP50" in report


def test_write_json_to_local_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("VALOHAI_OUTPUTS_DIR", str(tmp_path))
    path = write_json("metrics.json", {"stage": "test", "value": 1})
    assert path == tmp_path / "metrics.json"
    assert path.exists()
    assert outputs_dir() == tmp_path
