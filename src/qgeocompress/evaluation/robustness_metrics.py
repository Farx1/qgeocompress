from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from qgeocompress.data.corruptions import apply_corruption
from qgeocompress.data.prepare_dota import get_data_yaml
from qgeocompress.evaluation.system_metrics import _sample_images
from qgeocompress.models.load_model import extract_detection_metrics


def evaluate_robustness(
    weights: str | Path,
    dataset: str,
    corruptions: list[str],
    severities: list[int] | None = None,
    imgsz: int = 640,
    device: str | None = None,
    max_images: int = 32,
) -> dict[str, Any]:
    """Evaluate mAP drop under synthetic corruptions."""
    severities = severities or [3]
    model = YOLO(str(weights))
    images = _sample_images(dataset, max_images=max_images)
    data_yaml = get_data_yaml(dataset)

    clean_metrics = model.val(data=data_yaml, imgsz=imgsz, device=device, verbose=False)
    clean_map = extract_detection_metrics(clean_metrics).get("map50", 0.0)

    results: dict[str, Any] = {
        "dataset": dataset,
        "weights": str(weights),
        "map50_clean": clean_map,
        "corruptions": {},
    }

    for corruption in corruptions:
        for severity in severities:
            corrupted_dir = Path("results/raw/corrupted") / dataset / corruption / str(severity)
            corrupted_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for img_path in images:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                out = apply_corruption(img, corruption, severity)
                out_path = corrupted_dir / Path(img_path).name
                cv2.imwrite(str(out_path), out)
                paths.append(str(out_path))

            if not paths:
                continue
            list_file = corrupted_dir / "images.txt"
            list_file.write_text("\n".join(paths))
            # Ultralytics val needs a yaml; use simplified image list eval via predict + manual proxy
            preds = model.predict(paths, imgsz=imgsz, device=device, verbose=False)
            confidences = []
            for p in preds:
                if p.boxes is not None and len(p.boxes):
                    confidences.extend(p.boxes.conf.cpu().numpy().tolist())
            mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
            key = f"{corruption}_s{severity}"
            results["corruptions"][key] = {
                "mean_confidence": round(mean_conf, 4),
                "num_images": len(paths),
                "relative_map_drop": None,  # full mAP needs labeled corrupted set
            }

    return results
