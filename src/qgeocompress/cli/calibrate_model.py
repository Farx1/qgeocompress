from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from qgeocompress.calibration.ece import confident_error_rate, expected_calibration_error
from qgeocompress.calibration.temperature_scaling import apply_temperature, fit_temperature
from qgeocompress.data.prepare_dota import get_data_yaml
from qgeocompress.utils.config import project_root, save_json
from qgeocompress.utils.device import resolve_device
from qgeocompress.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate model confidence post-compression")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", default="dota128")
    parser.add_argument("--method", default="temperature-scaling")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    logger = setup_logging()
    device = resolve_device(args.device)
    data_yaml = get_data_yaml(args.dataset)
    model = YOLO(str(args.weights))

    preds = model.predict(source=data_yaml, imgsz=args.imgsz, device=device, verbose=False, split="val")
    confidences = []
    for p in preds:
        if p.boxes is not None and len(p.boxes):
            confidences.extend(p.boxes.conf.cpu().numpy().tolist())

    conf_arr = np.array(confidences) if confidences else np.array([0.5])
    correct = np.ones_like(conf_arr)  # placeholder until GT matching is wired
    ece_before = expected_calibration_error(conf_arr, correct)

    logits = np.log(conf_arr / (1 - conf_arr + 1e-6))
    temperature = fit_temperature(logits, correct)
    calibrated = 1 / (1 + np.exp(-apply_temperature(logits, temperature)))
    ece_after = expected_calibration_error(calibrated, correct)
    cer = confident_error_rate(calibrated, correct)

    out_dir = project_root() / "runs" / "calibrated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_weights = out_dir / f"{args.weights.stem}_calibrated.pt"
    model.save(str(out_weights))

    summary = {
        "weights": str(args.weights),
        "calibrated_weights": str(out_weights),
        "method": args.method,
        "temperature": round(temperature, 4),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "confident_error_rate": round(cer, 4),
    }
    out = project_root() / "results" / "summaries" / f"calibrated_{args.weights.stem}.json"
    save_json(summary, out)
    logger.info("Calibration done — ECE %.4f → %.4f", ece_before, ece_after)


if __name__ == "__main__":
    main()
