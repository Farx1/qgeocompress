from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from qgeocompress.utils.config import project_root, save_json

_SKIP_STATUSES = frozenset({
    "skipped",
    "unsupported_for_ultralytics_eval",
    "failed_no_layers_replaced",
    "failed_reload_instability",
    "unsupported_train_api_state_loss",
})

FINETUNE_EXCLUSION_NOTE = (
    "Fine-tuning post-compression excluded from main results: Ultralytics train() does not "
    "preserve the compressed checkpoint state (e.g. only 106/541 weights transferred), "
    "making post-fine-tune runs non-comparable. A custom training loop or structural "
    "low-rank modules (Phase 3) are required."
)

# mAP50 zones for rank-sensitivity plot annotation (conceptual bands).
_SAFE_MAP50 = 0.90
_USABLE_MAP50 = 0.82
_BROKEN_MAP50 = 0.75


@dataclass
class ReportOptions:
    exclude_finetune_runs: bool = True
    include_unsupported_finetune: bool = False
    exclude_finetune_failures: bool = True


def load_run_summaries(results_dir: str | Path) -> list[dict[str, Any]]:
    results_dir = Path(results_dir)
    runs: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("dataset_report") or path.name == "final_report.json":
            continue
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def normalize_run_for_report(run: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy with report-friendly fields (does not mutate source JSON)."""
    out = dict(run)

    if out.get("compression") in (None, "none", "baseline"):
        if not out.get("status"):
            out["status"] = "ok"

    is_svd_inplace = (
        out.get("compression_mode") == "svd_in_place"
        or out.get("method") == "low-rank"
        and out.get("compression") == "low-rank"
        and out.get("collapsed_for_ultralytics_compat") is False
    )
    if is_svd_inplace:
        theo = out.get("theoretical_rank_reduction_pct")
        if theo is None and out.get("param_reduction_pct") is not None:
            theo = out["param_reduction_pct"]
        if theo is not None:
            out["theoretical_rank_reduction_pct"] = theo

    return out


def is_finetune_run(run: dict[str, Any]) -> bool:
    return bool(run.get("finetune_epochs", 0))


def is_finetune_failure(run: dict[str, Any]) -> bool:
    if not is_finetune_run(run):
        return False
    if run.get("status") in _SKIP_STATUSES:
        return True
    after = run.get("map50_after_finetune")
    before = run.get("map50_before_finetune") or run.get("map50_before_reload")
    if after is not None and before is not None and before - after > 0.10:
        return True
    return run.get("train_state_preserved") is False


def filter_runs_for_report(
    runs: list[dict[str, Any]],
    options: ReportOptions | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split runs into (included, excluded) for the main Phase 2A report."""
    options = options or ReportOptions()
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for raw in runs:
        run = normalize_run_for_report(raw)
        status = run.get("status")

        if options.exclude_finetune_runs and is_finetune_run(run):
            if not options.include_unsupported_finetune:
                excluded.append({**run, "_exclusion_reason": "finetune_run_excluded"})
                continue

        if status in _SKIP_STATUSES and not (
            options.include_unsupported_finetune and status == "unsupported_train_api_state_loss"
        ):
            excluded.append({**run, "_exclusion_reason": f"status={status}"})
            continue

        if options.exclude_finetune_failures and is_finetune_failure(run):
            if not options.include_unsupported_finetune:
                excluded.append({**run, "_exclusion_reason": "finetune_failure"})
                continue

        included.append(run)

    return included, excluded


def effective_map50(run: dict[str, Any]) -> float | None:
    """Primary mAP50 for reporting: before fine-tune for compressed runs; never post-FT by default."""
    if is_finetune_run(run):
        return run.get("map50_before_finetune") or run.get("map50")
    if run.get("map50_before_finetune") is not None:
        return run.get("map50_before_finetune")
    return run.get("map50")


def run_display_label(run: dict[str, Any]) -> str:
    label = run.get("compression") or run.get("run_id", "?")
    if run.get("rank_ratio") is not None:
        label = f"{label} r={run['rank_ratio']}"
    if run.get("finetune_epochs", 0):
        label = f"{label} +ft{run['finetune_epochs']}"
    return label


def is_evaluable_run(run: dict[str, Any]) -> bool:
    if run.get("status") in _SKIP_STATUSES:
        return False
    if is_finetune_run(run):
        return False
    primary_map = effective_map50(run)
    if run.get("compression") in (None, "none", "baseline"):
        return primary_map is not None
    if run.get("ultralytics_compatible") is False and run.get("method") != "fp16":
        return False
    return primary_map is not None


def compute_global_score(
    row: dict[str, Any],
    baseline: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> float | None:
    """Composite score for evaluable runs only."""
    if not is_evaluable_run(row):
        return None

    w = weights or {"map": 0.35, "throughput": 0.25, "memory": 0.15, "ece": 0.15, "cer": 0.05, "robust": 0.05}

    def norm(val: float | None, ref: float) -> float:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return 0.0
        return val / ref if ref > 0 else 0.0

    map_score = norm(effective_map50(row), baseline.get("map50", 1) or 1)
    thr_score = norm(row.get("throughput_img_s"), baseline.get("throughput_img_s", 1) or 1)
    mem_gain = 1.0 - norm(row.get("vram_mb_max"), baseline.get("vram_mb_max", 1) or 1)
    ece_penalty = norm(row.get("ece"), max(baseline.get("ece", 0.01) or 0.01, 0.01))
    cer_penalty = norm(
        row.get("confident_error_rate"),
        max(baseline.get("confident_error_rate", 0.01) or 0.01, 0.01),
    )
    robust_penalty = row.get("robustness_drop") or 0

    return (
        w["map"] * map_score
        + w["throughput"] * thr_score
        + w["memory"] * mem_gain
        - w["ece"] * ece_penalty
        - w["cer"] * cer_penalty
        - w["robust"] * robust_penalty
    )


def make_comparison_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    if not runs:
        return pd.DataFrame()

    baseline = next(
        (r for r in runs if r.get("compression") in (None, "none", "baseline") and is_evaluable_run(r)),
        next((r for r in runs if is_evaluable_run(r)), runs[0]),
    )

    for r in runs:
        score = compute_global_score(r, baseline)
        r["global_score"] = round(score, 3) if score is not None else None
        r["evaluable"] = is_evaluable_run(r)
        r["map50_reported"] = effective_map50(r)

    df = pd.DataFrame(runs)
    cols = [
        c
        for c in [
            "run_id",
            "model",
            "compression",
            "status",
            "evaluable",
            "rank_ratio",
            "finetune_epochs",
            "map50_reported",
            "map50",
            "map50_before_finetune",
            "map50_95",
            "latency_ms_mean",
            "throughput_img_s",
            "vram_mb_max",
            "model_size_mb",
            "num_replaced_layers",
            "theoretical_rank_reduction_pct",
            "ece",
            "confident_error_rate",
            "reason",
            "global_score",
        ]
        if c in df.columns
    ]
    sort_col = "global_score" if "global_score" in df.columns else cols[0]
    return df[cols].sort_values(sort_col, ascending=False, na_position="last")


def _baseline_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    for r in runs:
        if r.get("compression") in (None, "none", "baseline") and r.get("map50") is not None:
            return r
    return None


def low_rank_sweep_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Low-rank runs without fine-tune, sorted by rank_ratio ascending."""
    sweep = [
        r
        for r in runs
        if r.get("compression") == "low-rank"
        and not is_finetune_run(r)
        and r.get("rank_ratio") is not None
        and effective_map50(r) is not None
    ]
    return sorted(sweep, key=lambda r: float(r["rank_ratio"]))


def detect_rupture_rank(sweep: list[dict[str, Any]], baseline_map50: float) -> float | None:
    """Lowest rank_ratio that still meets the usable mAP50 threshold (stability frontier)."""
    if not sweep:
        return None
    viable = [r for r in sweep if (effective_map50(r) or 0) >= _USABLE_MAP50]
    if not viable:
        return None
    return min(float(r["rank_ratio"]) for r in viable)


def plot_rank_ratio_vs_map50(
    runs: list[dict[str, Any]],
    output_path: Path | None = None,
) -> Path | None:
    """Rank sensitivity plot: rank_ratio vs mAP50 for low-rank runs without fine-tune."""
    sweep = low_rank_sweep_runs(runs)
    if not sweep:
        return None

    baseline = _baseline_run(runs)
    baseline_map = float(baseline["map50"]) if baseline and baseline.get("map50") else None

    output_path = output_path or project_root() / "results" / "figures" / "rank_ratio_vs_map50.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xs = [float(r["rank_ratio"]) for r in sweep]
    ys = [float(effective_map50(r)) for r in sweep]

    fig, ax = plt.subplots(figsize=(9, 6))

    if baseline_map is not None:
        ax.axhspan(_SAFE_MAP50, 1.05, alpha=0.06, color="green", label=f"safe zone (mAP50≥{_SAFE_MAP50})")
        ax.axhspan(_USABLE_MAP50, _SAFE_MAP50, alpha=0.06, color="orange", label=f"usable (≥{_USABLE_MAP50})")
        ax.axhspan(0, _BROKEN_MAP50, alpha=0.06, color="red", label=f"broken (<{_BROKEN_MAP50})")
        ax.axhline(baseline_map, color="gray", linestyle="--", linewidth=1.2, label=f"baseline mAP50={baseline_map:.3f}")

    ax.plot(xs, ys, "o-", color="#2563eb", linewidth=2, markersize=8, label="low-rank (SVD in-place)")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), fontsize=8, xytext=(0, 8), textcoords="offset points", ha="center")

    ax.set_xlabel("Rank ratio")
    ax.set_ylabel("mAP50")
    ax.set_title("Q-GEOCompress — rank sensitivity (SVD in-place, no fine-tune)")
    ax.set_xlim(min(xs) - 0.02, max(xs) + 0.02)
    ax.set_ylim(0, max(ys + ([baseline_map] if baseline_map else [])) * 1.05)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_tradeoff(runs: list[dict[str, Any]], output_path: Path | None = None) -> Path | None:
    """Signature plot: latency vs mAP for evaluable runs only."""
    plottable = [
        r
        for r in runs
        if is_evaluable_run(r)
        and r.get("latency_ms_mean") is not None
        and effective_map50(r) is not None
    ]
    if not plottable:
        return None

    output_path = output_path or project_root() / "results" / "figures" / "compression_tradeoff.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    for run in plottable:
        x = float(run["latency_ms_mean"])
        y = float(effective_map50(run))
        size = max((run.get("vram_mb_max") or 100) / 50, 20)
        ece = run.get("ece") or 0.05
        label = run_display_label(run)
        ax.scatter(x, y, s=size, c=[ece], cmap="coolwarm", vmin=0, vmax=0.15, edgecolors="k", alpha=0.85)
        ax.annotate(label, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Latency (ms / image)")
    ax.set_ylabel("mAP50")
    ax.set_title("Q-GEOCompress — latency vs accuracy (size=VRAM, color=ECE)")
    fig.colorbar(plt.cm.ScalarMappable(cmap="coolwarm"), ax=ax, label="ECE")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def generate_phase2a_summary(
    included: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = _baseline_run(included)
    baseline_map = float(baseline["map50"]) if baseline and baseline.get("map50") else None
    sweep = low_rank_sweep_runs(included)

    best_safe: dict[str, Any] | None = None
    best_usable: dict[str, Any] | None = None
    for r in sweep:
        m = effective_map50(r) or 0
        if m >= _SAFE_MAP50 and (best_safe is None or m > (effective_map50(best_safe) or 0)):
            best_safe = r
        if m >= _USABLE_MAP50 and (best_usable is None or float(r["rank_ratio"]) < float(best_usable["rank_ratio"])):
            best_usable = r

    rupture = detect_rupture_rank(sweep, baseline_map) if baseline_map else None

    finetune_excluded = [r for r in excluded if is_finetune_run(r) or r.get("status") == "unsupported_train_api_state_loss"]

    summary = {
        "baseline_map50": round(baseline_map, 4) if baseline_map else None,
        "best_safe_rank_ratio": best_safe.get("rank_ratio") if best_safe else None,
        "best_safe_map50": round(effective_map50(best_safe), 4) if best_safe else None,
        "best_usable_rank_ratio": best_usable.get("rank_ratio") if best_usable else None,
        "best_usable_map50": round(effective_map50(best_usable), 4) if best_usable else None,
        "rupture_rank_ratio_estimate": rupture,
        "num_low_rank_sweep_points": len(sweep),
        "num_finetune_runs_excluded": len(finetune_excluded),
        "finetune_exclusion_note": FINETUNE_EXCLUSION_NOTE,
    }
    return summary


def render_phase2a_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 2A — Low-rank rank sensitivity summary",
        "",
        "## Key findings",
        "",
    ]
    if summary.get("baseline_map50") is not None:
        lines.append(f"- **Baseline mAP50:** {summary['baseline_map50']:.3f}")
    if summary.get("best_safe_rank_ratio") is not None:
        lines.append(
            f"- **Best safe rank** (mAP50≥{_SAFE_MAP50}): "
            f"`r={summary['best_safe_rank_ratio']}` → mAP50={summary['best_safe_map50']:.3f}"
        )
    if summary.get("best_usable_rank_ratio") is not None:
        lines.append(
            f"- **Most aggressive viable rank** (mAP50≥{_USABLE_MAP50}): "
            f"`r={summary['best_usable_rank_ratio']}` → mAP50={summary['best_usable_map50']:.3f}"
        )
    if summary.get("rupture_rank_ratio_estimate") is not None:
        lines.append(
            f"- **Estimated stability frontier:** viable down to `r≈{summary['rupture_rank_ratio_estimate']:.2f}` "
            f"(lowest rank with mAP50≥{_USABLE_MAP50}); sharp drop expected below."
        )

    lines.extend([
        "",
        "## Fine-tuning (excluded)",
        "",
        summary.get("finetune_exclusion_note", FINETUNE_EXCLUSION_NOTE),
        "",
        f"_{summary.get('num_finetune_runs_excluded', 0)} fine-tune run(s) excluded from this report._",
        "",
        "## Interpretation",
        "",
        "SVD in-place is a **spectral weight approximation** (same param count / file size). "
        "`theoretical_rank_reduction_pct` reflects rank-truncation savings per layer, "
        "not deployable compression. Structural factorization is Phase 3.",
        "",
    ])
    return "\n".join(lines)


def generate_report(
    results_dir: str | Path,
    options: ReportOptions | None = None,
) -> dict[str, Any]:
    options = options or ReportOptions()
    all_runs = load_run_summaries(results_dir)
    included, excluded = filter_runs_for_report(all_runs, options)

    df = make_comparison_table(included)
    table_path = Path(results_dir) / "comparison_table.csv"
    df.to_csv(table_path, index=False)

    fig_tradeoff = plot_tradeoff(included)
    fig_rank = plot_rank_ratio_vs_map50(included)

    phase2a = generate_phase2a_summary(included, excluded)
    summary_md = render_phase2a_summary_markdown(phase2a)
    summary_path = Path(results_dir) / "phase2a_summary.md"
    summary_path.write_text(summary_md)

    evaluable = [r for r in included if is_evaluable_run(r)]
    skipped = [r for r in all_runs if r.get("status") in _SKIP_STATUSES or r in excluded]

    scored = df[df["evaluable"] == True] if "evaluable" in df.columns else df  # noqa: E712
    top_run = scored.iloc[0].to_dict() if not scored.empty else {}

    report = {
        "num_runs": len(all_runs),
        "num_included": len(included),
        "num_excluded": len(excluded),
        "num_evaluable": len(evaluable),
        "num_skipped": len(skipped),
        "table_csv": str(table_path),
        "tradeoff_figure": str(fig_tradeoff) if fig_tradeoff else None,
        "rank_sensitivity_figure": str(fig_rank) if fig_rank else None,
        "phase2a_summary_md": str(summary_path),
        "phase2a_summary": phase2a,
        "finetune_exclusion_note": FINETUNE_EXCLUSION_NOTE,
        "top_run": top_run,
        "excluded_runs": [
            {
                "run_id": r.get("run_id"),
                "compression": r.get("compression"),
                "status": r.get("status"),
                "finetune_epochs": r.get("finetune_epochs"),
                "reason": r.get("_exclusion_reason") or r.get("reason") or r.get("fine_tune_aborted_reason"),
            }
            for r in excluded
        ],
        "skipped_runs": [
            {"run_id": r.get("run_id"), "compression": r.get("compression"), "status": r.get("status"), "reason": r.get("reason")}
            for r in skipped
            if r.get("status") in _SKIP_STATUSES
        ],
    }
    save_json(report, Path(results_dir) / "final_report.json")
    return report
