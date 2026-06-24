from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from qgeocompress.evaluation.calibration_real import (
    compute_calibration_metrics,
    plot_coverage_risk_curve,
    plot_reliability_diagram,
)
from qgeocompress.evaluation.gt_matching import (
    load_ground_truth_for_dataset,
    match_dataset_predictions,
    run_yolo_predictions,
)
from qgeocompress.models.load_model import extract_detection_metrics
from qgeocompress.utils.config import make_run_id, project_root, save_json
from qgeocompress.utils.device import resolve_device
from qgeocompress.utils.logging import setup_logging
from qgeocompress.data.prepare_dota import resolve_data_yaml
from qgeocompress.utils.paths import resolve_baseline_weights


def _calibration_run_id(
    compression: str,
    rank_ratio: float | None,
    iou_threshold: float,
    run_label: str | None = None,
    weights: Path | None = None,
) -> str:
    if compression in (None, "none", "baseline"):
        prefix = "calibration_baseline"
        if run_label:
            prefix = f"{prefix}_{run_label}"
    elif compression == "structural-low-rank":
        prefix = f"calibration_structural_r{rank_ratio:.3f}" if rank_ratio is not None else "calibration_structural"
        if run_label:
            prefix = f"{prefix}_{run_label}"
    elif compression == "low-rank":
        prefix = f"calibration_low_rank_r{rank_ratio:.3f}" if rank_ratio is not None else "calibration_low_rank"
    else:
        prefix = f"calibration_{compression.replace('-', '_')}"
        if rank_ratio is not None:
            prefix += f"_r{rank_ratio:.3f}"

    digest_input: dict[str, Any] = {"iou": iou_threshold}
    if run_label:
        digest_input["run_label"] = run_label
    if weights is not None:
        digest_input["weights"] = weights.name
    digest = make_run_id(prefix, digest_input).split("_")[-1]
    return f"{prefix}_{digest}"


def _has_structural_layers(model: YOLO) -> bool:
    from qgeocompress.compression.structural_low_rank import StructuralLowRankConv2d

    return any(isinstance(m, StructuralLowRankConv2d) for m in model.model.modules())


def evaluate_calibration(
    weights: Path,
    dataset: str = "dota128",
    data_yaml: str | Path | None = None,
    compression: str = "baseline",
    rank_ratio: float | None = None,
    source_run_id: str | None = None,
    run_label: str | None = None,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.001,
    imgsz: int = 640,
    device: str | None = None,
    iou_mode: str = "obb",
) -> dict[str, Any]:
    device = resolve_device(device)
    model = YOLO(str(weights), task="obb")
    yaml_path = resolve_data_yaml(dataset=dataset, data_yaml=data_yaml)

    gt_by_image = load_ground_truth_for_dataset(dataset=dataset, data_yaml=yaml_path)
    pred_by_image = run_yolo_predictions(
        model,
        dataset=dataset,
        data_yaml=yaml_path,
        conf_threshold=conf_threshold,
        imgsz=imgsz,
        device=device,
    )
    matching = match_dataset_predictions(
        pred_by_image,
        gt_by_image,
        iou_threshold=iou_threshold,
        iou_mode=iou_mode,  # type: ignore[arg-type]
    )
    cal_metrics = compute_calibration_metrics(matching)

    if _has_structural_layers(model):
        from qgeocompress.evaluation.obb_validate import validate_no_fuse

        val_results = validate_no_fuse(model, yaml_path, imgsz=imgsz, device=device)
    else:
        val_results = model.val(data=yaml_path, imgsz=imgsz, device=device, verbose=False)
    det_metrics = extract_detection_metrics(val_results)

    run_id = _calibration_run_id(compression, rank_ratio, iou_threshold, run_label=run_label, weights=weights)
    if compression in (None, "none", "baseline"):
        label = "baseline"
    elif run_label:
        label = run_label
    elif compression == "structural-low-rank":
        label = f"structural r={rank_ratio}"
    else:
        label = f"r={rank_ratio}"

    figures_dir = project_root() / "results" / "figures"
    rel_fig = plot_reliability_diagram(
        cal_metrics["reliability_bins"],
        label,
        figures_dir / f"reliability_{run_id}.png",
    )
    cov_fig = plot_coverage_risk_curve(
        cal_metrics["coverage_curve"],
        label,
        figures_dir / f"coverage_risk_{run_id}.png",
    )

    summary: dict[str, Any] = {
        "run_id": run_id,
        "source_run_id": source_run_id,
        "dataset": dataset,
        "data_yaml": yaml_path,
        "compression": compression,
        "rank_ratio": rank_ratio,
        "run_label": run_label,
        "weights": str(weights),
        "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold,
        "imgsz": imgsz,
        "device": device,
        "status": "ok",
        "map50": det_metrics.get("map50"),
        "map50_95": det_metrics.get("map50_95"),
        "reliability_figure": str(rel_fig),
        "coverage_risk_figure": str(cov_fig),
        **{k: v for k, v in cal_metrics.items() if k != "coverage_curve"},
        "coverage_curve": cal_metrics["coverage_curve"],
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate GT-matched calibration for YOLO-OBB")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", default="dota128")
    parser.add_argument("--data-yaml", type=Path, default=None, help="Ultralytics data YAML (overrides --dataset)")
    parser.add_argument("--compression", default="baseline")
    parser.add_argument("--rank-ratio", type=float, default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument(
        "--run-label",
        default=None,
        help="Distinct label for output JSON (e.g. top5_sens) when compression/rank collide",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--conf-threshold", type=float, default=0.001)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--iou-mode", choices=["obb", "axis_aligned"], default="obb")
    args = parser.parse_args(argv)

    logger = setup_logging()
    weights = resolve_baseline_weights(args.weights) if args.compression == "baseline" else Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    logger.info("Evaluating calibration for %s (%s)", weights, args.compression)
    summary = evaluate_calibration(
        weights=weights,
        dataset=args.dataset,
        data_yaml=args.data_yaml,
        compression=args.compression,
        rank_ratio=args.rank_ratio,
        source_run_id=args.source_run_id,
        run_label=args.run_label,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
        imgsz=args.imgsz,
        device=args.device,
        iou_mode=args.iou_mode,
    )

    out = project_root() / "results" / "summaries" / f"{summary['run_id']}.json"
    save_json(summary, out)
    logger.info(
        "Calibration %s — ECE=%.4f CER@0.8=%.4f mAP50=%.4f → %s",
        summary["run_id"],
        summary.get("ece", 0),
        summary.get("confident_error_rate_08", 0),
        summary.get("map50") or 0,
        out,
    )


if __name__ == "__main__":
    main()
