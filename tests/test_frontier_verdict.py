import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "frontier_study", Path(__file__).resolve().parents[1] / "scripts" / "frontier_study.py"
)
frontier_study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frontier_study)


def test_conservative_verdict_uses_ci_low():
    base = {"map50": 0.89, "cer08": 0.015}
    row = {"delta": -0.08, "delta_ci_low": -0.15, "cer08": 0.016, "param_reduction_pct": 17.0}
    point = frontier_study.gate_verdict(base, {"map50": base["map50"] + row["delta"], "cer08": row["cer08"]}, 17.0)
    assert point == "deployable_compression_candidate"
    assert frontier_study.gate_verdict_ci(base, row) == "rejected"
