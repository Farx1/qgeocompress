from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from qgeocompress.compression import compress_model
from qgeocompress.compression.finetune import (
    FinetuneConfig,
    FINETUNE_BACKEND,
    RELOAD_DROP_ABS_THRESHOLD,
    RELOAD_DROP_REL_THRESHOLD,
    UNSUPPORTED_TRAIN_API_REASON,
    compute_reload_drop,
    load_baseline_map50,
    reload_is_unstable,
)
from qgeocompress.data.prepare_dota import get_data_yaml
from qgeocompress.evaluation.system_metrics import benchmark_inference
from qgeocompress.models.load_model import extract_detection_metrics, model_size_mb
from qgeocompress.utils.config import load_model_config, make_run_id, project_root, save_json
from qgeocompress.utils.device import resolve_device
from qgeocompress.utils.logging import setup_logging
from qgeocompress.utils.paths import resolve_baseline_weights
from qgeocompress.utils.seed import set_seed


def _empty_metrics() -> dict[str, Any]:
    return {
        "map50": None,
        "map50_95": None,
        "precision": None,
        "recall": None,
        "latency_ms_mean": None,
        "throughput_img_s": None,
        "vram_mb_max": None,
    }


def _has_structural_layers(model: YOLO) -> bool:
    from qgeocompress.compression.structural_low_rank import StructuralLowRankConv2d

    return any(isinstance(m, StructuralLowRankConv2d) for m in model.model.modules())


def _evaluate_model(
    model: YOLO,
    weights_path: Path | None,
    dataset: str,
    imgsz: int,
    device: str,
    half: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data_yaml = get_data_yaml(dataset)
    if _has_structural_layers(model):
        from qgeocompress.evaluation.obb_validate import validate_no_fuse

        val_results = validate_no_fuse(model, data_yaml, imgsz=imgsz, device=device, half=half)
    else:
        val_results = model.val(data=data_yaml, imgsz=imgsz, device=device, half=half, verbose=False)
    det_metrics = extract_detection_metrics(val_results)

    bench_row: dict[str, Any] = {}
    if weights_path is not None and weights_path.exists():
        if _has_structural_layers(model):
            bench_row = {
                "latency_ms_mean": None,
                "throughput_img_s": None,
                "benchmark_skipped": "structural_checkpoint_reload_incompatible_with_fuse",
            }
        else:
            bench = benchmark_inference(weights_path, dataset=dataset, batch_sizes=[1], imgsz=imgsz, device=device)
            bench_row = bench[0] if bench else {}

    return det_metrics, bench_row


def _map50_only(model: YOLO, dataset: str, imgsz: int, device: str) -> float:
    metrics, _ = _evaluate_model(model, None, dataset, imgsz, device)
    return float(metrics.get("map50") or 0.0)


def _evaluate_weights(
    weights_path: Path,
    dataset: str,
    imgsz: int,
    device: str,
) -> dict[str, Any]:
    if not weights_path.exists():
        return _empty_metrics()
    model = YOLO(str(weights_path), task="obb")
    metrics, _ = _evaluate_model(model, weights_path, dataset, imgsz, device)
    return metrics


def _resolve_baseline_map50(weights: Path, dataset: str, imgsz: int, device: str) -> float:
    cached = load_baseline_map50(dataset)
    if cached is not None:
        return cached
    baseline = YOLO(str(weights), task="obb")
    metrics, _ = _evaluate_model(baseline, weights, dataset, imgsz, device)
    return float(metrics.get("map50") or 0.0)


def _finetune_config_from_args(args: argparse.Namespace) -> FinetuneConfig:
    return FinetuneConfig(
        lr0=args.finetune_lr0,
        lrf=args.finetune_lrf,
        freeze=args.finetune_freeze,
        use_last=args.finetune_use_last,
    )


def _run_low_rank_pipeline(
    args: argparse.Namespace,
    device: str,
    model_cfg: dict[str, Any],
    logger: Any,
) -> dict[str, Any]:
    rank_label = f"low_rank_r{args.rank_ratio:.3f}"
    compress_dir = args.output_dir / rank_label

    meta, model = compress_model(
        args.weights,
        args.method,
        compress_dir,
        rank_ratio=args.rank_ratio,
        target_layers=args.target_layers,
    )

    status = meta.get("status", "ok")
    if status != "ok" or not meta.get("ultralytics_compatible") or model is None:
        logger.info("Low-rank compression status=%s — skipping eval/fine-tune", status)
        return _build_summary(args, model_cfg, meta, _empty_metrics(), {}, None, {})

    before_weights = Path(meta["output_weights"])
    before_metrics, _ = _evaluate_model(model, before_weights, args.dataset, args.imgsz, device)

    ft_config = _finetune_config_from_args(args)
    finetune_meta: dict[str, Any] = {
        "finetune_epochs": args.finetune_epochs,
        "finetune_lr0": ft_config.lr0 if args.finetune_epochs > 0 else None,
        "finetune_lrf": ft_config.lrf if args.finetune_epochs > 0 else None,
        "finetune_freeze": ft_config.freeze,
        "finetune_use_last": ft_config.use_last,
        "map50_before_finetune": before_metrics.get("map50"),
        "map50_95_before_finetune": before_metrics.get("map50_95"),
        "output_weights_before_finetune": str(before_weights),
        "map50_after_finetune": None,
        "map50_95_after_finetune": None,
        "map50_after_finetune_best": None,
        "map50_after_finetune_last": None,
        "map50_95_after_finetune_best": None,
        "map50_95_after_finetune_last": None,
        "output_weights_after_finetune": None,
        "selected_finetune_weights": None,
        "map50_recovery_pct": None,
        "map50_baseline_ref": None,
        "map50_before_reload": None,
        "map50_after_reload_before_training": None,
        "reload_drop": None,
        "fine_tune_aborted_reason": None,
        "finetune_backend": None,
        "train_state_preserved": None,
    }

    eval_metrics = dict(before_metrics)
    bench_row: dict[str, Any] = {}
    model_size = round(model_size_mb(before_weights), 2)

    if args.finetune_epochs > 0:
        set_seed(args.seed)

        map_before_reload = before_metrics.get("map50")
        reload_metrics = _evaluate_weights(before_weights, args.dataset, args.imgsz, device)
        map_after_reload = reload_metrics.get("map50")
        reload_drop = compute_reload_drop(map_before_reload, map_after_reload)

        finetune_meta.update({
            "map50_before_reload": map_before_reload,
            "map50_after_reload_before_training": map_after_reload,
            "reload_drop": reload_drop,
        })

        if reload_is_unstable(map_before_reload, map_after_reload):
            reason = (
                f"reload_drop={reload_drop} exceeds threshold "
                f"(abs>{RELOAD_DROP_ABS_THRESHOLD} or rel>{RELOAD_DROP_REL_THRESHOLD}); "
                "checkpoint reload unstable — fine-tune aborted"
            )
            finetune_meta["fine_tune_aborted_reason"] = reason
            meta["status"] = "failed_reload_instability"
            logger.warning("Fine-tune aborted: %s", reason)
            _, bench_row = _evaluate_model(model, before_weights, args.dataset, args.imgsz, device)
        else:
            # Ultralytics train() does not preserve SVD in-place compressed state (106/541 transfer).
            finetune_meta.update({
                "finetune_backend": FINETUNE_BACKEND,
                "train_state_preserved": False,
                "fine_tune_aborted_reason": UNSUPPORTED_TRAIN_API_REASON,
            })
            meta["status"] = "unsupported_train_api_state_loss"
            logger.warning(
                "Fine-tune skipped: Ultralytics train() API cannot preserve compressed low-rank state. %s",
                UNSUPPORTED_TRAIN_API_REASON,
            )
            _, bench_row = _evaluate_model(model, before_weights, args.dataset, args.imgsz, device)
    else:
        _, bench_row = _evaluate_model(model, before_weights, args.dataset, args.imgsz, device)

    return _build_summary(args, model_cfg, meta, eval_metrics, bench_row, model_size, finetune_meta)


def _build_summary(
    args: argparse.Namespace,
    model_cfg: dict[str, Any],
    meta: dict[str, Any],
    det_metrics: dict[str, Any],
    bench_row: dict[str, Any],
    model_size: float | None,
    finetune_meta: dict[str, Any],
) -> dict[str, Any]:
    run_config = {
        "compression": args.method,
        "weights": str(args.weights),
        "rank_ratio": args.rank_ratio if args.method == "low-rank" else None,
        "finetune_epochs": args.finetune_epochs if args.method == "low-rank" else 0,
        "finetune_lr0": args.finetune_lr0 if args.finetune_epochs > 0 else None,
    }
    return {
        "run_id": make_run_id(args.method, run_config),
        "dataset": args.dataset,
        "model": model_cfg.get("name", args.model),
        "compression": args.method,
        "rank_ratio": args.rank_ratio if args.method == "low-rank" else None,
        **det_metrics,
        **bench_row,
        "model_size_mb": model_size,
        **meta,
        **finetune_meta,
    }


def _add_finetune_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--finetune-epochs", type=int, default=0)
    parser.add_argument("--finetune-lr0", type=float, default=1e-5)
    parser.add_argument("--finetune-lrf", type=float, default=0.1)
    parser.add_argument("--finetune-freeze", type=int, default=None)
    parser.add_argument("--finetune-use-last", action="store_true", default=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compress a trained YOLO model")
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "fp16",
            "int8-ptq",
            "magnitude-pruning",
            "low-rank",
            "structural-low-rank",
            "structural-probe",
        ],
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", default="dota128")
    parser.add_argument("--model", default="yolo_obb_small")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/compressed"))
    parser.add_argument("--sparsity", type=float, default=0.2)
    parser.add_argument("--rank-ratio", type=float, default=0.5)
    parser.add_argument("--target-layers", nargs="+", default=["backbone", "neck"])
    parser.add_argument(
        "--target-scope",
        default="backbone-neck",
        help="Layer scope for structural-low-rank (default: backbone-neck)",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=["all", "sensitivity"],
        default="all",
        help="Layer selection for structural-low-rank (sensitivity requires probe JSON)",
    )
    parser.add_argument(
        "--max-replaced-layers",
        type=int,
        default=None,
        help="Max structural layers to replace (used with all or sensitivity strategy)",
    )
    parser.add_argument(
        "--bn-recalibration-batches",
        type=int,
        default=0,
        help="Forward batches for BN recalibration after structural replacement",
    )
    parser.add_argument(
        "--probe-max-layers",
        type=int,
        default=None,
        help="Limit structural-probe to first N candidate layers (debug/fast)",
    )
    _add_finetune_args(parser)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    logger = setup_logging()
    device = resolve_device(args.device)
    args.weights = resolve_baseline_weights(args.weights)
    model_cfg = load_model_config(args.model)

    if args.method == "low-rank":
        if args.finetune_epochs < 0:
            raise ValueError("--finetune-epochs must be >= 0")
        summary = _run_low_rank_pipeline(args, device, model_cfg, logger)
        out = project_root() / "results" / "summaries" / f"{summary['run_id']}.json"
        save_json(summary, out)
        logger.info(
            "Low-rank [r=%.3f, ft=%d, lr0=%.1e] — mAP50 before=%.4f best=%s last=%s selected=%s (%s)",
            args.rank_ratio,
            args.finetune_epochs,
            args.finetune_lr0,
            summary.get("map50_before_finetune") or 0.0,
            summary.get("map50_after_finetune_best"),
            summary.get("map50_after_finetune_last"),
            summary.get("map50_after_finetune"),
            summary.get("selected_finetune_weights"),
        )
        logger.info("Saved to %s", out)
        return

    if args.method == "structural-probe":
        logger.info(
            "Structural probe r=%.3f on %s (this runs one val per candidate layer)",
            args.rank_ratio,
            args.dataset,
        )

        def _probe_eval(m: YOLO) -> float:
            return _map50_only(m, args.dataset, args.imgsz, device)

        meta, _ = compress_model(
            args.weights,
            "structural-probe",
            args.output_dir / "structural_probe",
            rank_ratio=args.rank_ratio,
            target_layers=args.target_layers,
            dataset=args.dataset,
            imgsz=args.imgsz,
            device=device,
            eval_map50_fn=_probe_eval,
            max_probe_layers=args.probe_max_layers,
        )
        summary = {
            "run_id": make_run_id("structural-probe", {
                "rank_ratio": args.rank_ratio,
                "dataset": args.dataset,
                "weights": str(args.weights),
            }),
            "dataset": args.dataset,
            "compression": "structural-probe",
            "rank_ratio": args.rank_ratio,
            **meta,
        }
        out = project_root() / "results" / "summaries" / f"{summary['run_id']}.json"
        save_json(summary, out)
        logger.info(
            "Structural probe complete — %d candidates, report=%s",
            meta.get("num_candidates", 0),
            meta.get("sensitivity_report"),
        )
        logger.info("Saved summary to %s", out)
        return

    out_subdir = args.output_dir / args.method.replace("-", "_")
    meta, in_memory_model = compress_model(
        args.weights,
        args.method,
        out_subdir,
        sparsity=args.sparsity,
        rank_ratio=args.rank_ratio,
        target_layers=args.target_layers,
        target_scope=args.target_scope,
        selection_strategy=args.selection_strategy,
        max_replaced_layers=args.max_replaced_layers,
        bn_recalibration_batches=args.bn_recalibration_batches,
        dataset=args.dataset,
        imgsz=args.imgsz,
        device=device,
    )

    status = meta.get("status", "ok")
    det_metrics = _empty_metrics()
    bench_row: dict[str, Any] = {}
    model_size: float | None = None

    if status == "ok" and meta.get("ultralytics_compatible"):
        eval_model = in_memory_model if in_memory_model is not None else YOLO(str(meta["output_weights"]))
        det_metrics, bench_row = _evaluate_model(
            eval_model,
            Path(meta["output_weights"]),
            args.dataset,
            args.imgsz,
            device,
        )
        model_size = round(model_size_mb(meta["output_weights"]), 2)
    elif status == "ok" and args.method == "fp16":
        eval_model = YOLO(str(args.weights))
        det_metrics, bench_row = _evaluate_model(
            eval_model,
            args.weights,
            args.dataset,
            args.imgsz,
            device,
            half=True,
        )
        model_size = round(model_size_mb(args.weights), 2)
    else:
        logger.info("Compression [%s] status=%s — skipping Ultralytics eval", args.method, status)

    run_config = {
        "compression": args.method,
        "weights": str(args.weights),
        "rank_ratio": args.rank_ratio if args.method in ("low-rank", "structural-low-rank") else None,
        "finetune_epochs": 0,
        "finetune_lr0": None,
        "selection_strategy": args.selection_strategy if args.method == "structural-low-rank" else None,
        "max_replaced_layers": args.max_replaced_layers,
    }
    summary = {
        "run_id": make_run_id(args.method, run_config),
        "dataset": args.dataset,
        "model": model_cfg.get("name", args.model),
        "compression": args.method,
        "rank_ratio": args.rank_ratio if args.method in ("low-rank", "structural-low-rank") else None,
        "structural_compression": args.method == "structural-low-rank",
        "selection_strategy": args.selection_strategy if args.method == "structural-low-rank" else None,
        "max_replaced_layers": args.max_replaced_layers,
        "bn_recalibration_batches": args.bn_recalibration_batches,
        **det_metrics,
        **bench_row,
        "model_size_mb": model_size,
        **meta,
    }
    out = project_root() / "results" / "summaries" / f"{summary['run_id']}.json"
    save_json(summary, out)
    logger.info("Compression [%s] status=%s — saved to %s", args.method, status, out)


if __name__ == "__main__":
    main()
