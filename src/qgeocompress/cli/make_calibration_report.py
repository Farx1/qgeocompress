from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd

from qgeocompress.evaluation.calibration_real import calibration_run_label, plot_reliability_overlay
from qgeocompress.utils.config import project_root, save_json
from qgeocompress.utils.logging import setup_logging
from qgeocompress.utils.paths import DEFAULT_BASELINE_WEIGHTS

DEFAULT_RANKS = [0.88, 0.84, 0.82, 0.80]

SAFE_MAP50 = 0.85
VIABLE_MAP50 = 0.82
SAFE_CER08_DELTA = 0.01
VIABLE_CER08_DELTA = 0.02

OperationalStatus = Literal["safe", "viable", "degraded", "unknown"]

ECE_INTERPRETATION_NOTE = (
    "ECE measures prediction-level calibration only (confidence vs. correctness on emitted "
    "predictions). It does not capture missed detections (false negatives). Interpret ECE "
    "together with mAP50, recall, CER@0.8, and coverage metrics."
)


def load_calibration_summaries(results_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("calibration_*.json")):
        if path.name == "calibration_report.json":
            continue
        runs.append(json.loads(path.read_text()))
    return runs


def filter_phase2b_runs(
    runs: list[dict[str, Any]],
    ranks: list[float] | None = None,
) -> list[dict[str, Any]]:
    ranks = ranks or DEFAULT_RANKS
    rank_set = {round(r, 3) for r in ranks}
    selected: list[dict[str, Any]] = []

    baseline = next((r for r in runs if r.get("compression") in (None, "none", "baseline")), None)
    if baseline:
        selected.append(baseline)

    for rank in sorted(rank_set, reverse=True):
        match = next(
            (
                r
                for r in runs
                if r.get("compression") == "low-rank"
                and r.get("rank_ratio") is not None
                and round(float(r["rank_ratio"]), 3) == round(rank, 3)
            ),
            None,
        )
        if match:
            selected.append(match)
    return selected


def operational_reliability_status(
    map50: float | None,
    cer08: float | None,
    baseline_cer08: float | None,
) -> OperationalStatus:
    """Classify operational reliability relative to baseline CER@0.8."""
    if map50 is None or cer08 is None or baseline_cer08 is None:
        return "unknown"
    if map50 >= SAFE_MAP50 and cer08 <= baseline_cer08 + SAFE_CER08_DELTA:
        return "safe"
    if map50 >= VIABLE_MAP50 and cer08 <= baseline_cer08 + VIABLE_CER08_DELTA:
        return "viable"
    return "degraded"


def make_calibration_table(
    runs: list[dict[str, Any]],
    baseline: dict[str, Any] | None = None,
) -> pd.DataFrame:
    baseline_map = (baseline or {}).get("map50")
    baseline_cer08 = (baseline or {}).get("confident_error_rate_08")

    rows = []
    for run in runs:
        map50 = run.get("map50")
        cer08 = run.get("confident_error_rate_08")
        map_drop = None
        cer_delta = None
        if baseline_map is not None and map50 is not None:
            map_drop = round(float(baseline_map) - float(map50), 6)
        if baseline_cer08 is not None and cer08 is not None:
            cer_delta = round(float(cer08) - float(baseline_cer08), 6)

        rows.append({
            "run_id": run.get("run_id"),
            "label": calibration_run_label(run),
            "compression": run.get("compression"),
            "rank_ratio": run.get("rank_ratio"),
            "map50": map50,
            "map50_drop_vs_baseline": map_drop,
            "ece": run.get("ece"),
            "mce": run.get("mce"),
            "brier_score": run.get("brier_score"),
            "confident_error_rate_07": run.get("confident_error_rate_07"),
            "confident_error_rate_08": cer08,
            "cer08_delta_vs_baseline": cer_delta,
            "confident_error_rate_09": run.get("confident_error_rate_09"),
            "operational_reliability_status": operational_reliability_status(
                map50, cer08, baseline_cer08
            ),
            "selective_accuracy_at_80_coverage": run.get("selective_accuracy_at_80_coverage"),
            "selective_accuracy_at_60_coverage": run.get("selective_accuracy_at_60_coverage"),
            "num_predictions": run.get("num_predictions"),
            "num_tp": run.get("num_tp"),
            "num_fp": run.get("num_fp"),
            "num_fn": run.get("num_fn"),
            "iou_threshold": run.get("iou_threshold"),
        })
    df = pd.DataFrame(rows)
    if not df.empty and "rank_ratio" in df.columns:
        df = df.sort_values(
            by=["rank_ratio"],
            ascending=False,
            na_position="first",
        )
    return df


def plot_map50_vs_cer08(
    runs: list[dict[str, Any]],
    output_path: Path | None = None,
) -> Path | None:
    """Scatter: mAP50 vs CER@0.8 — performance vs confident-error risk."""
    plottable = [
        r for r in runs
        if r.get("map50") is not None and r.get("confident_error_rate_08") is not None
    ]
    if not plottable:
        return None

    output_path = output_path or project_root() / "results" / "figures" / "map50_vs_cer08.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    status_colors = {"safe": "#16a34a", "viable": "#ca8a04", "degraded": "#dc2626", "unknown": "#6b7280"}

    baseline_cer = next(
        (r.get("confident_error_rate_08") for r in plottable if r.get("compression") in (None, "none", "baseline")),
        None,
    )

    for run in plottable:
        x = float(run["map50"])
        y = float(run["confident_error_rate_08"])
        label = calibration_run_label(run)
        status = operational_reliability_status(x, y, baseline_cer)
        color = status_colors.get(status, "#6b7280")
        ax.scatter(x, y, s=120, c=color, edgecolors="k", alpha=0.9, zorder=3)
        ax.annotate(label, (x, y), fontsize=9, xytext=(6, 6), textcoords="offset points")

    if baseline_cer is not None:
        ax.axhline(float(baseline_cer), color="gray", linestyle="--", linewidth=1, label="baseline CER@0.8")
        ax.axhline(float(baseline_cer) + SAFE_CER08_DELTA, color="orange", linestyle=":", linewidth=1, label="safe CER bound")
        ax.axhline(float(baseline_cer) + VIABLE_CER08_DELTA, color="red", linestyle=":", linewidth=1, label="viable CER bound")
    ax.axvline(SAFE_MAP50, color="green", linestyle=":", linewidth=1, alpha=0.5, label=f"mAP50 safe ≥{SAFE_MAP50}")
    ax.axvline(VIABLE_MAP50, color="orange", linestyle=":", linewidth=1, alpha=0.5, label=f"mAP50 viable ≥{VIABLE_MAP50}")

    ax.set_xlabel("mAP50")
    ax.set_ylabel("CER@0.8 (confident error rate)")
    ax.set_title("Q-GEOCompress — performance vs confident-error risk")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_calibration_summary_markdown(runs: list[dict[str, Any]], baseline: dict[str, Any] | None) -> str:
    lines = [
        "# Phase 2B — GT-matched calibration summary",
        "",
        "Reliability metrics computed with greedy GT matching (IoU OBB, one GT per match).",
        "",
    ]
    if not runs:
        lines.append("_No calibration runs found._")
        return "\n".join(lines)

    best_ece = min(runs, key=lambda r: r.get("ece", float("inf")))
    best_cer = min(runs, key=lambda r: r.get("confident_error_rate_08", float("inf")))

    lines.append(f"- **Best ECE:** `{calibration_run_label(best_ece)}` → ECE={best_ece.get('ece', 0):.4f}")
    lines.append(
        f"- **Lowest CER@0.8:** `{calibration_run_label(best_cer)}` "
        f"→ CER@0.8={best_cer.get('confident_error_rate_08', 0):.4f}"
    )

    baseline_cer = (baseline or {}).get("confident_error_rate_08")
    for run in runs:
        status = operational_reliability_status(
            run.get("map50"),
            run.get("confident_error_rate_08"),
            baseline_cer,
        )
        lines.append(
            f"- **{calibration_run_label(run)}:** operational status = `{status}` "
            f"(mAP50={run.get('map50', 0):.4f}, CER@0.8={run.get('confident_error_rate_08', 0):.4f})"
        )

    lines.extend([
        "",
        "## ECE caveat",
        "",
        ECE_INTERPRETATION_NOTE,
        "",
        "A lower ECE on a compressed model does not imply better overall reliability if mAP "
        "or recall dropped sharply — the model may simply emit fewer risky predictions while "
        "missing more objects.",
        "",
    ])

    if baseline:
        base_ece = baseline.get("ece", 0)
        unreliable = [
            r for r in runs
            if r.get("compression") == "low-rank"
            and r.get("ece") is not None
            and base_ece is not None
            and r["ece"] > base_ece * 1.5
        ]
        if unreliable:
            worst = max(unreliable, key=lambda r: r.get("ece", 0))
            lines.append(
                f"- **ECE spike (informational):** `r={worst.get('rank_ratio')}` "
                f"shows ECE={worst.get('ece', 0):.4f} (>1.5× baseline)"
            )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "High mAP alone does not guarantee trustworthy confidence scores. "
        "Compare ECE and CER@0.8 alongside mAP50, recall (via FN rate), and selective "
        "prediction to assess operational reliability.",
        "",
        "Phase 2B finding: rank-based spectral degradation does not immediately produce "
        "confidence miscalibration. Variants r=0.88–0.82 keep ECE near baseline with low "
        "CER@0.8. At r=0.80, mAP drops sharply and CER@0.8 rises — the operational "
        "reliability boundary aligns with the performance cliff.",
        "",
    ])
    return "\n".join(lines)


def generate_calibration_report(
    results_dir: Path | None = None,
    ranks: list[float] | None = None,
) -> dict[str, Any]:
    results_dir = results_dir or project_root() / "results" / "summaries"
    all_runs = load_calibration_summaries(results_dir)
    selected = filter_phase2b_runs(all_runs, ranks=ranks)
    baseline = next((r for r in selected if r.get("compression") in (None, "none", "baseline")), None)

    table = make_calibration_table(selected, baseline=baseline)
    table_path = results_dir / "calibration_comparison_table.csv"
    table.to_csv(table_path, index=False)

    overlay = plot_reliability_overlay(selected)
    map_cer_fig = plot_map50_vs_cer08(selected)
    summary_md = render_calibration_summary_markdown(selected, baseline)
    summary_path = results_dir / "calibration_summary.md"
    summary_path.write_text(summary_md)

    report = {
        "num_calibration_runs": len(all_runs),
        "num_included": len(selected),
        "table_csv": str(table_path),
        "reliability_overlay": str(overlay) if overlay else None,
        "map50_vs_cer08_figure": str(map_cer_fig) if map_cer_fig else None,
        "summary_md": str(summary_path),
        "included_run_ids": [r.get("run_id") for r in selected],
    }
    save_json(report, results_dir / "calibration_report.json")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate Phase 2B calibration results")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--ranks", type=float, nargs="*", default=None)
    args = parser.parse_args(argv)

    logger = setup_logging()
    results_dir = args.results_dir or project_root() / "results" / "summaries"
    report = generate_calibration_report(results_dir, ranks=args.ranks)
    logger.info("Calibration report: %s", report)
    logger.info("Baseline weights ref: %s", DEFAULT_BASELINE_WEIGHTS)


if __name__ == "__main__":
    main()
