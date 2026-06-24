from qgeocompress.evaluation.reliability_gate import GateThresholds, evaluate_reliability_gate


def test_gate_deployable_when_performance_and_compression_met():
    baseline = {"map50": 0.93, "confident_error_rate_08": 0.004, "status": "ok"}
    candidate = {"map50": 0.88, "confident_error_rate_08": 0.005, "status": "ok"}
    meta = {"real_param_reduction_pct": 4.5, "status": "ok"}
    result = evaluate_reliability_gate(baseline, candidate, meta)
    assert result["gate_status"] == "deployable_compression_candidate"
    assert result["accepted_for_export"] is True


def test_gate_methodologically_valid_without_param_gain():
    baseline = {"map50": 0.93, "confident_error_rate_08": 0.004}
    candidate = {"map50": 0.88, "confident_error_rate_08": 0.005, "status": "ok"}
    meta = {"real_param_reduction_pct": 0.2}
    result = evaluate_reliability_gate(baseline, candidate, meta)
    assert result["gate_status"] == "methodologically_valid"
    assert result["accepted_for_export"] is False


def test_gate_rejected_when_map_collapses():
    baseline = {"map50": 0.93, "confident_error_rate_08": 0.004}
    candidate = {"map50": 0.20, "confident_error_rate_08": 0.05, "status": "ok"}
    result = evaluate_reliability_gate(baseline, candidate, {"real_param_reduction_pct": 7.0})
    assert result["gate_status"] == "rejected"


def test_holdout_top5_is_methodologically_valid():
    baseline = {"map50": 0.9287, "confident_error_rate_08": 0.0043}
    candidate = {"map50": 0.8790, "confident_error_rate_08": 0.0047, "status": "ok"}
    meta = {"real_param_reduction_pct": 0.21}
    result = evaluate_reliability_gate(baseline, candidate, meta, GateThresholds())
    assert result["gate_status"] == "methodologically_valid"
