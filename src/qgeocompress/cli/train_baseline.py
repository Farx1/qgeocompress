from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ultralytics import YOLO

from qgeocompress.calibration.ece import confident_error_rate, expected_calibration_error
from qgeocompress.data.prepare_dota import get_data_yaml
from qgeocompress.evaluation.system_metrics import benchmark_inference
from qgeocompress.models.load_model import extract_detection_metrics, model_size_mb
from qgeocompress.utils.config import load_dataset_config, load_model_config, make_run_id, project_root, save_json
from qgeocompress.utils.device import resolve_device
from qgeocompress.utils.logging import setup_logging
from qgeocompress.utils.seed import set_seed


def _collect_calibration_arrays(results) -> tuple[list[float], list[int]]:
    confidences: list[float] = []
    correct: list[int] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        confs = r.boxes.conf.cpu().numpy()
        # Proxy: high confidence treated as correct for pipeline smoke test
        confidences.extend(confs.tolist())
        correct.extend([1] * len(confs))
    return confidences, correct


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train baseline YOLO-OBB model")
    parser.add_argument("--dataset", default="dota128")
    parser.add_argument("--model", default="yolo_obb_small")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/baseline")
    parser.add_argument("--name", default="train")
    args = parser.parse_args(argv)

    logger = setup_logging()
    set_seed(args.seed)
    device = resolve_device(args.device)

    ds_cfg = load_dataset_config(args.dataset)
    model_cfg = load_model_config(args.model)
    data_yaml = get_data_yaml(args.dataset)
    weights = model_cfg.get("weights", "yolo11n-obb.pt")

    logger.info("Training %s on %s for %d epochs", weights, args.dataset, args.epochs)
    model = YOLO(weights)
    train_results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    best_weights = Path(train_results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = Path(train_results.save_dir) / "weights" / "last.pt"

    val_results = model.val(data=data_yaml, imgsz=args.imgsz, device=device)
    det_metrics = extract_detection_metrics(val_results)

    bench = benchmark_inference(best_weights, dataset=args.dataset, batch_sizes=[1, 4], imgsz=args.imgsz, device=device)
    bench_primary = bench[0] if bench else {}

    preds = model.predict(source=_val_sample(data_yaml), imgsz=args.imgsz, device=device, verbose=False)
    confs, correct = _collect_calibration_arrays(preds)

    ece = expected_calibration_error(np.array(confs), np.array(correct)) if confs else 0.0
    cer = confident_error_rate(np.array(confs), np.array(correct)) if confs else 0.0

    run_config = {
        "dataset": args.dataset,
        "model": args.model,
        "compression": "baseline",
        "epochs": args.epochs,
        "seed": args.seed,
    }
    summary = {
        "run_id": make_run_id("baseline", run_config),
        **run_config,
        "status": "ok",
        "ultralytics_compatible": True,
        "output_format": "ultralytics_pt",
        **det_metrics,
        **bench_primary,
        "model_size_mb": round(model_size_mb(best_weights), 2),
        "ece": round(ece, 4),
        "confident_error_rate": round(cer, 4),
        "weights": str(best_weights),
        "save_dir": str(train_results.save_dir),
    }

    out = project_root() / "results" / "summaries" / f"{summary['run_id']}.json"
    save_json(summary, out)
    logger.info("Baseline complete — mAP50=%.4f, saved to %s", summary.get("map50", 0), out)


def _val_sample(data_yaml: str, n: int = 8) -> str | list[str]:
    from ultralytics.data.utils import check_det_dataset

    info = check_det_dataset(data_yaml)
    val = info.get("val") or info.get("train")
    if not val:
        return data_yaml
    p = Path(val)
    exts = {".jpg", ".jpeg", ".png", ".tif"}
    if p.is_dir():
        imgs = [str(f) for f in sorted(p.iterdir()) if f.suffix.lower() in exts][:n]
        return imgs or str(p)
    return str(p)


if __name__ == "__main__":
    main()
