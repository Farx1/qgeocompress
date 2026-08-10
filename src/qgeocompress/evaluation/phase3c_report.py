from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PENDING_ROWS: list[dict[str, str]] = [
    {
        "method": "Structural top-5 pareto-gain BN=20 (GPU)",
        "run_label": "holdout_top5_pareto_bn20",
        "status": "pending",
        "reproduce": (
            "python scripts/compress_model.py --method structural-low-rank --rank-ratio 0.84 "
            "--weights $BEST --selection-strategy pareto-gain --max-replaced-layers 5 "
            "--sensitivity-file results/summaries/structural_layer_sensitivity_holdout_select.json "
            "--bn-data-yaml $HOLDOUT/dota128_holdout_train.yaml "
            "--eval-data-yaml $HOLDOUT/dota128_holdout_test.yaml "
            "--run-label holdout_top5_pareto_bn20 --device cuda"
        ),
    },
    {
        "method": "Latency benchmark CUDA (GPU)",
        "run_label": "holdout_latency_cuda",
        "status": "pending",
        "reproduce": "See docs/GPU_RUNBOOK.md — benchmark_inference.py --device cuda",
    },
]

CANONICAL_CANDIDATES: list[dict[str, Any]] = [
    {"method": "Baseline (hold-out test)", "run_label": "holdout_test_baseline", "kind": "baseline"},
    {"method": "Structural top-3 sensitivity", "run_label": "holdout_top3_sens"},
    {"method": "Structural top-5 sensitivity", "run_label": "holdout_top5_sens"},
    {"method": "Structural top-3 pareto-gain", "run_label": "holdout_top3_pareto"},
    {"method": "Structural top-5 pareto-gain", "run_label": "holdout_top5_pareto"},
    {"method": "Structural top-5 quantum-qaoa (classical QUBO)", "run_label": "holdout_top5_qaoa"},
    {"method": "Structural top-5 quantum-qaoa (QAOA simulator)", "run_label": "holdout_top5_qaoa_sim"},
    {"method": "SVD in-place r=0.84 (hold-out test)", "run_label": "holdout_low_rank_r084"},
    {"method": "Structural top-8 pareto-gain", "run_label": "holdout_top8_pareto"},
    {"method": "Structural full r=0.84", "run_label": "holdout_full"},
]


def _baseline_latency(summaries_dir: Path) -> float | None:
    """Batch-1 latency from benchmark_inference.py --run-label holdout_test_baseline."""
    path = summaries_dir / "benchmark_holdout_test_baseline.json"
    if not path.exists():
        return None
    benchmarks = json.loads(path.read_text(encoding="utf-8")).get("benchmarks") or []
    for entry in benchmarks:
        if entry.get("batch_size") == 1:
            return entry.get("latency_ms_mean")
    return benchmarks[0].get("latency_ms_mean") if benchmarks else None


def _load_json_dir(summaries_dir: Path, pattern: str) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(summaries_dir.glob(pattern))]


def _find_by_run_label(items: list[dict[str, Any]], run_label: str) -> dict[str, Any] | None:
    for item in items:
        if item.get("run_label") == run_label:
            return item
    return None


def _find_cal(items: list[dict[str, Any]], suffix: str) -> dict[str, Any] | None:
    for item in items:
        label = item.get("run_label") or ""
        if suffix in label or label.endswith(suffix):
            return item
    return None


def _find_gate(gates: list[dict[str, Any]], candidate_run_id: str | None) -> dict[str, Any] | None:
    if not candidate_run_id:
        return None
    for g in gates:
        if g.get("candidate_run_id") == candidate_run_id:
            return g
        if candidate_run_id in (g.get("run_id") or ""):
            return g
    return None


def _gate_status(gate: dict[str, Any] | None, compress: dict[str, Any] | None) -> str:
    if gate:
        return gate.get("gate_status") or gate.get("valohai_status") or gate.get("status") or "—"
    if compress and compress.get("status") == "ok":
        return "no_gate_json"
    return "—"


def _load_holdout_calibrations(summaries_dir: Path) -> list[dict[str, Any]]:
    """Load calibration JSONs whose run_label or data_yaml references hold-out."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(summaries_dir.glob("calibration_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        label = str(data.get("run_label") or "")
        yaml_path = str(data.get("data_yaml") or "")
        if "holdout" not in label and "holdout" not in yaml_path:
            continue
        run_id = str(data.get("run_id") or path.name)
        if run_id in seen:
            continue
        seen.add(run_id)
        rows.append(data)
    return rows


def build_phase3c_rows(summaries_dir: Path) -> list[dict[str, Any]]:
    compress_rows = _load_json_dir(summaries_dir, "structural-low-rank_*.json")
    compress_rows = [c for c in compress_rows if "holdout" in (c.get("run_label") or "")]
    calibrations = _load_holdout_calibrations(summaries_dir)
    gates = _load_json_dir(summaries_dir, "reliability_gate_*holdout*.json")
    baseline_cals = [c for c in calibrations if c.get("compression") == "baseline"]

    rows: list[dict[str, Any]] = []

    baseline_cal = baseline_cals[0] if baseline_cals else None
    if baseline_cal:
        rows.append({
            "method": "Baseline (hold-out test)",
            "run_label": baseline_cal.get("run_label"),
            "map50": baseline_cal.get("map50"),
            "cer08": baseline_cal.get("confident_error_rate_08"),
            "ece": baseline_cal.get("ece"),
            "param_reduction_pct": 0.0,
            "bn_batches": None,
            "gate_status": "reference",
            "latency_ms_mean": _baseline_latency(summaries_dir),
            "status": "ok",
        })

    for spec in CANONICAL_CANDIDATES:
        if spec.get("kind") == "baseline":
            continue
        compress = _find_by_run_label(compress_rows, spec["run_label"])
        # Calibration runs are labelled "<run_label>_test" by evaluate_calibration.py.
        cal = None if spec.get("kind") == "baseline" else _find_cal(calibrations, f"{spec['run_label']}_test")
        gate = _find_gate(gates, cal.get("run_id") if cal else None)

        if compress is None and cal is None:
            continue

        rows.append({
            "method": spec["method"],
            "run_label": spec["run_label"],
            "map50": (cal or compress or {}).get("map50"),
            "cer08": (cal or {}).get("confident_error_rate_08"),
            "ece": (cal or {}).get("ece"),
            "param_reduction_pct": (compress or {}).get("real_param_reduction_pct"),
            "bn_batches": (compress or {}).get("bn_recalibration_batches"),
            "gate_status": _gate_status(gate, compress),
            "latency_ms_mean": (compress or {}).get("latency_ms_mean"),
            "status": (compress or cal or {}).get("status", "ok"),
            "weights": (cal or compress or {}).get("weights") or (compress or {}).get("output_weights"),
        })

    for pending in PENDING_ROWS:
        rows.append({
            "method": pending["method"],
            "run_label": pending["run_label"],
            "map50": None,
            "cer08": None,
            "ece": None,
            "param_reduction_pct": None,
            "bn_batches": None,
            "gate_status": pending["status"],
            "latency_ms_mean": None,
            "status": "pending",
            "reproduce": pending["reproduce"],
        })

    return rows


def render_phase3c_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Q-GEOCompress — Phase 3C Comparison (Hold-out)",
        "",
        "Unified comparison on **test split** (80/24/24 protocol). Generated by `make_phase3c_report.py`.",
        "",
        "| Method | mAP50 | CER@0.8 | ECE | Params Δ | BN batches | Gate | Latency ms |",
        "| ------ | ----: | ------: | ---: | -------: | ---------: | ---- | ---------: |",
    ]
    for r in rows:
        map50 = f"{r['map50']:.3f}" if r.get("map50") is not None else "—"
        cer = f"{r['cer08']:.4f}" if r.get("cer08") is not None else "—"
        ece = f"{r['ece']:.4f}" if r.get("ece") is not None else "—"
        params = f"{r['param_reduction_pct']:.2f}%" if r.get("param_reduction_pct") is not None else "—"
        bn = str(r.get("bn_batches") if r.get("bn_batches") is not None else "—")
        gate = str(r.get("gate_status", "—"))
        lat = f"{r['latency_ms_mean']:.1f}" if r.get("latency_ms_mean") is not None else "—"
        lines.append(f"| {r['method']} | {map50} | {cer} | {ece} | {params} | {bn} | {gate} | {lat} |")

    pending = [r for r in rows if r.get("status") == "pending"]
    if pending:
        lines.extend(["", "## Pending experiments", ""])
        for r in pending:
            lines.append(f"- **{r['method']}**: `{r.get('reproduce', '')}`")

    lines.extend([
        "",
        "## Notes",
        "",
        "- Latency is single-run CPU wall time on a shared host. Two arms that select the "
        "identical layer set (top-5 sensitivity and top-5 quantum-qaoa/classical) differ by "
        "~17%, so treat that as the noise floor and compare only against the baseline row.",
        "- Gate statuses: `deployable_compression_candidate`, `methodologically_valid`, `rejected`.",
    ])
    return "\n".join(lines) + "\n"


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    headers = [
        "method", "run_label", "map50", "cer08", "ece", "param_reduction_pct",
        "bn_batches", "gate_status", "latency_ms_mean", "status", "weights", "reproduce",
    ]
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(
            str(r.get(h, "")).replace(",", ";") for h in headers
        ))
    return "\n".join(lines) + "\n"
