from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

GateStatus = Literal[
    "deployable_compression_candidate",
    "methodologically_valid",
    "rejected",
]


@dataclass
class GateThresholds:
    """Post-training quality gate thresholds vs baseline reference."""

    max_map50_drop: float = 0.10
    max_cer08_delta: float = 0.02
    min_param_reduction_pct: float = 2.0
    require_status_ok: bool = True


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_reliability_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    compression_meta: dict[str, Any] | None = None,
    thresholds: GateThresholds | None = None,
) -> dict[str, Any]:
    """Accept/reject a compressed model against baseline quality gates."""
    thresholds = thresholds or GateThresholds()
    meta = compression_meta or {}

    baseline_map50 = _f(baseline.get("map50"))
    baseline_cer08 = _f(baseline.get("confident_error_rate_08"))
    candidate_map50 = _f(candidate.get("map50"))
    candidate_cer08 = _f(candidate.get("confident_error_rate_08"))
    candidate_status = candidate.get("status") or meta.get("status")
    param_reduction = _f(meta.get("real_param_reduction_pct") or candidate.get("real_param_reduction_pct"))

    map50_drop = None
    if baseline_map50 is not None and candidate_map50 is not None:
        map50_drop = round(max(0.0, baseline_map50 - candidate_map50), 6)

    cer08_delta = None
    if baseline_cer08 is not None and candidate_cer08 is not None:
        cer08_delta = round(candidate_cer08 - baseline_cer08, 6)

    checks = {
        "status_ok": (candidate_status == "ok") if thresholds.require_status_ok else True,
        "map50_within_tolerance": (
            map50_drop is not None and map50_drop <= thresholds.max_map50_drop
        ),
        "cer08_within_tolerance": (
            cer08_delta is not None and cer08_delta <= thresholds.max_cer08_delta
        ),
        "param_reduction_met": (
            param_reduction is not None and param_reduction >= thresholds.min_param_reduction_pct
        ),
    }

    performance_ok = all([
        checks["status_ok"],
        checks["map50_within_tolerance"],
        checks["cer08_within_tolerance"],
    ])

    if performance_ok and checks["param_reduction_met"]:
        gate_status: GateStatus = "deployable_compression_candidate"
    elif performance_ok:
        gate_status = "methodologically_valid"
    else:
        gate_status = "rejected"

    return {
        "gate_status": gate_status,
        "accepted_for_export": gate_status == "deployable_compression_candidate",
        "checks": checks,
        "thresholds": asdict(thresholds),
        "baseline_map50": baseline_map50,
        "candidate_map50": candidate_map50,
        "map50_drop_vs_baseline": map50_drop,
        "baseline_cer08": baseline_cer08,
        "candidate_cer08": candidate_cer08,
        "cer08_delta_vs_baseline": cer08_delta,
        "real_param_reduction_pct": param_reduction,
        "candidate_weights": candidate.get("weights") or meta.get("output_weights"),
    }
