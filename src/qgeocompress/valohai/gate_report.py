from __future__ import annotations

from typing import Any

from qgeocompress.evaluation.reliability_gate import GateThresholds, evaluate_reliability_gate

VALOHAI_STATUS_MAP = {
    "deployable_compression_candidate": "accepted_for_export",
    "methodologically_valid": "accepted_for_research",
    "rejected": "rejected",
}


def valohai_gate_status(internal_status: str) -> str:
    return VALOHAI_STATUS_MAP.get(internal_status, "rejected")


def build_gate_reasons(result: dict[str, Any]) -> list[str]:
    checks = result.get("checks") or {}
    reasons: list[str] = []
    if checks.get("map50_within_tolerance"):
        reasons.append("mAP drop acceptable")
    else:
        reasons.append("mAP drop exceeds tolerance")
    if checks.get("cer08_within_tolerance"):
        reasons.append("CER stable")
    else:
        reasons.append("CER delta exceeds tolerance")
    if checks.get("param_reduction_met"):
        reasons.append("param reduction meets deployable threshold")
    else:
        reasons.append("param reduction below deployable threshold")
    if not checks.get("status_ok", True):
        reasons.append("compression status not ok")
    return reasons


def evaluate_quality_gate(
    comparison: dict[str, Any],
    compression_summary: dict[str, Any] | None = None,
    thresholds: GateThresholds | None = None,
) -> dict[str, Any]:
    """Run quality gate from inference-compare output + compression summary."""
    baseline_cal = comparison.get("baseline_calibration") or {}
    compressed_cal = comparison.get("compressed_calibration") or {}

    if not baseline_cal and comparison.get("baseline"):
        b = comparison["baseline"]
        baseline_cal = {
            "map50": b.get("map50"),
            "confident_error_rate_08": b.get("cer08"),
            "status": "ok",
        }
    if not compressed_cal and comparison.get("compressed"):
        c = comparison["compressed"]
        compressed_cal = {
            "map50": c.get("map50"),
            "confident_error_rate_08": c.get("cer08"),
            "status": (compression_summary or {}).get("status", "ok"),
            "weights": (compression_summary or {}).get("output_weights"),
        }
    meta = compression_summary or {}

    if meta.get("real_param_reduction_pct") is None and comparison.get("delta", {}).get("param_reduction_pct") is not None:
        meta = {**meta, "real_param_reduction_pct": comparison["delta"]["param_reduction_pct"]}

    gate = evaluate_reliability_gate(baseline_cal, compressed_cal, meta, thresholds)
    valohai_status = valohai_gate_status(gate["gate_status"])
    reasons = build_gate_reasons(gate)

    return {
        "stage": "quality_gate",
        "accepted": valohai_status != "rejected",
        "accepted_for_export": valohai_status == "accepted_for_export",
        "status": valohai_status,
        "gate_status": gate["gate_status"],
        "valohai_status": valohai_status,
        "reasons": reasons,
        "checks": gate["checks"],
        "thresholds": gate["thresholds"],
        "map50_drop": gate.get("map50_drop_vs_baseline") or comparison.get("delta", {}).get("map50_drop"),
        "cer08_delta": gate.get("cer08_delta_vs_baseline") or comparison.get("delta", {}).get("cer08_delta"),
        "real_param_reduction_pct": gate.get("real_param_reduction_pct"),
        "baseline_map50": gate.get("baseline_map50"),
        "compressed_map50": gate.get("candidate_map50"),
        "baseline_cer08": gate.get("baseline_cer08"),
        "compressed_cer08": gate.get("candidate_cer08"),
        "candidate_weights": gate.get("candidate_weights") or compression_summary.get("output_weights"),
    }


def render_final_report(gate: dict[str, Any], comparison: dict[str, Any]) -> str:
    """Markdown summary for Valohai artifact / scientific reporting."""
    b = comparison.get("baseline") or {}
    c = comparison.get("compressed") or {}
    d = comparison.get("delta") or {}
    lines = [
        "# Q-GEOCompress — Post-training Quality Gate",
        "",
        f"**Status:** `{gate.get('valohai_status')}`",
        "",
        "## Detection (test split)",
        "",
        "| Metric | Baseline | Compressed | Delta |",
        "| ------ | -------: | ---------: | ----: |",
        f"| mAP50 | {b.get('map50', '—')} | {c.get('map50', '—')} | {d.get('map50_drop', '—')} |",
        f"| CER@0.8 | {b.get('cer08', '—')} | {c.get('cer08', '—')} | {d.get('cer08_delta', '—')} |",
        f"| ECE | {b.get('ece', '—')} | {c.get('ece', '—')} | — |",
        "",
        "## Compression & inference",
        "",
        f"- Param reduction: **{gate.get('real_param_reduction_pct', d.get('param_reduction_pct', '—'))}%**",
        f"- Latency speedup: **{d.get('latency_speedup_pct', '—')}%**",
        f"- Baseline latency: {b.get('latency_ms_mean', '—')} ms/img",
        f"- Compressed latency: {c.get('latency_ms_mean', '—')} ms/img",
        "",
        "## Gate reasons",
        "",
    ]
    for reason in gate.get("reasons") or []:
        lines.append(f"- {reason}")
    lines.append("")
    if gate.get("accepted_for_export"):
        lines.append("> Candidate **accepted for export** (performance + compression thresholds met).")
    elif gate.get("accepted"):
        lines.append("> Candidate **accepted for research** (performance OK, compression below deploy threshold).")
    else:
        lines.append("> Candidate **rejected**.")
    return "\n".join(lines)
