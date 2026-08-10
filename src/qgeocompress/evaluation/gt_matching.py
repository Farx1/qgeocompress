from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.ops import xyxyxyxy2xywhr

from qgeocompress.data.prepare_dota import resolve_data_yaml

IoUMode = Literal["obb", "axis_aligned"]


@dataclass
class OBBDetection:
    """Single oriented box with optional confidence (predictions only)."""

    class_id: int
    polygon: np.ndarray  # shape (4, 2), pixel coordinates
    confidence: float | None = None
    image_id: str = ""


@dataclass
class MatchOutcome:
    prediction: OBBDetection
    matched_gt: OBBDetection | None
    is_tp: bool
    iou: float


@dataclass
class MatchingSummary:
    outcomes: list[MatchOutcome]
    num_predictions: int
    num_gt: int
    num_tp: int
    num_fp: int
    num_fn: int
    iou_threshold: float
    iou_mode: IoUMode
    confidences: np.ndarray = field(repr=False)
    correct: np.ndarray = field(repr=False)

    def to_counts_dict(self) -> dict[str, Any]:
        return {
            "num_predictions": self.num_predictions,
            "num_gt": self.num_gt,
            "num_tp": self.num_tp,
            "num_fp": self.num_fp,
            "num_fn": self.num_fn,
            "iou_threshold": self.iou_threshold,
            "iou_mode": self.iou_mode,
        }


def _image_extensions() -> set[str]:
    return {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def list_val_images(
    dataset: str | None = None,
    data_yaml: str | Path | None = None,
) -> list[tuple[str, Path]]:
    """Return (image_id, image_path) pairs for the validation split in a data YAML."""
    yaml_path = resolve_data_yaml(dataset=dataset, data_yaml=data_yaml)
    info = check_det_dataset(yaml_path)
    val_path = info.get("val") or info.get("train")
    if not val_path:
        raise FileNotFoundError(f"No val split found for dataset {dataset}")

    root = Path(val_path)
    exts = _image_extensions()
    images: list[tuple[str, Path]] = []

    if root.is_file():
        with open(root) as f:
            for line in f:
                p = Path(line.strip())
                if p.suffix.lower() in exts and p.exists():
                    images.append((p.stem, p))
        return images

    if root.is_dir():
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in exts:
                images.append((p.stem, p))
        return images

    raise FileNotFoundError(f"Could not resolve validation images from {val_path}")


def label_path_for_image(image_path: Path, dataset_root: Path | None = None) -> Path:
    """Map an image path to its YOLO-OBB label file."""
    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")

    if dataset_root is not None:
        rel = image_path.relative_to(dataset_root)
        return (dataset_root / "labels" / rel).with_suffix(".txt")

    parent = image_path.parent
    if parent.name == "images":
        return parent.parent / "labels" / f"{image_path.stem}.txt"
    return parent / "labels" / f"{image_path.stem}.txt"


def load_yolo_obb_label_file(
    label_path: Path,
    image_id: str,
    img_w: int,
    img_h: int,
) -> list[OBBDetection]:
    """Load normalized YOLO-OBB labels: class x1 y1 x2 y2 x3 y3 x4 y4."""
    if not label_path.exists():
        return []

    detections: list[OBBDetection] = []
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        class_id = int(float(parts[0]))
        coords = np.array([float(x) for x in parts[1:9]], dtype=np.float64)
        coords[0::2] *= img_w
        coords[1::2] *= img_h
        polygon = coords.reshape(4, 2)
        detections.append(OBBDetection(class_id=class_id, polygon=polygon, image_id=image_id))
    return detections


def load_ground_truth_for_dataset(
    dataset: str | None = None,
    data_yaml: str | Path | None = None,
) -> dict[str, list[OBBDetection]]:
    """Load all GT OBB annotations keyed by image_id for the val split."""
    yaml_path = resolve_data_yaml(dataset=dataset, data_yaml=data_yaml)
    info = check_det_dataset(yaml_path)
    dataset_root = Path(info.get("path", "")) if info.get("path") else None

    gt_by_image: dict[str, list[OBBDetection]] = {}
    for image_id, image_path in list_val_images(dataset=dataset, data_yaml=data_yaml):
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = label_path_for_image(image_path, dataset_root)
        gt_by_image[image_id] = load_yolo_obb_label_file(label_path, image_id, w, h)
    return gt_by_image


def polygon_to_xywhr(polygon: np.ndarray) -> np.ndarray:
    flat = np.asarray(polygon, dtype=np.float64).reshape(1, 8)
    return np.asarray(xyxyxyxy2xywhr(flat), dtype=np.float64).reshape(5)


def obb_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Probabilistic IoU between two OBB polygons (pixel coordinates)."""
    a = torch.tensor(polygon_to_xywhr(poly_a), dtype=torch.float32).unsqueeze(0)
    b = torch.tensor(polygon_to_xywhr(poly_b), dtype=torch.float32).unsqueeze(0)
    return float(batch_probiou(a, b)[0, 0].item())


def axis_aligned_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Axis-aligned IoU fallback from polygon extrema."""
    ax1, ay1 = poly_a[:, 0].min(), poly_a[:, 1].min()
    ax2, ay2 = poly_a[:, 0].max(), poly_a[:, 1].max()
    bx1, by1 = poly_b[:, 0].min(), poly_b[:, 1].min()
    bx2, by2 = poly_b[:, 0].max(), poly_b[:, 1].max()

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def compute_pairwise_iou(
    poly_a: np.ndarray,
    poly_b: np.ndarray,
    iou_mode: IoUMode = "obb",
) -> float:
    if iou_mode == "axis_aligned":
        return axis_aligned_iou(poly_a, poly_b)
    try:
        return obb_iou(poly_a, poly_b)
    except Exception:
        return axis_aligned_iou(poly_a, poly_b)


def match_predictions_to_gt(
    predictions: list[OBBDetection],
    ground_truth: list[OBBDetection],
    iou_threshold: float = 0.5,
    iou_mode: IoUMode = "obb",
) -> MatchingSummary:
    """Greedy matching by descending confidence; each GT matched at most once."""
    sorted_preds = sorted(
        predictions,
        key=lambda p: p.confidence if p.confidence is not None else 0.0,
        reverse=True,
    )
    unmatched_gt = list(ground_truth)
    outcomes: list[MatchOutcome] = []
    num_tp = 0
    num_fp = 0

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt: OBBDetection | None = None
        best_idx: int | None = None

        for idx, gt in enumerate(unmatched_gt):
            if gt.class_id != pred.class_id:
                continue
            iou = compute_pairwise_iou(pred.polygon, gt.polygon, iou_mode=iou_mode)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt
                best_idx = idx

        if best_gt is not None and best_iou >= iou_threshold:
            outcomes.append(MatchOutcome(prediction=pred, matched_gt=best_gt, is_tp=True, iou=best_iou))
            unmatched_gt.pop(best_idx)  # type: ignore[arg-type]
            num_tp += 1
        else:
            outcomes.append(MatchOutcome(prediction=pred, matched_gt=None, is_tp=False, iou=best_iou))
            num_fp += 1

    num_fn = len(unmatched_gt)
    confidences = np.array([o.prediction.confidence or 0.0 for o in outcomes], dtype=np.float64)
    correct = np.array([1 if o.is_tp else 0 for o in outcomes], dtype=np.float64)

    return MatchingSummary(
        outcomes=outcomes,
        num_predictions=len(predictions),
        num_gt=len(ground_truth),
        num_tp=num_tp,
        num_fp=num_fp,
        num_fn=num_fn,
        iou_threshold=iou_threshold,
        iou_mode=iou_mode,
        confidences=confidences,
        correct=correct,
    )


def match_dataset_predictions(
    pred_by_image: dict[str, list[OBBDetection]],
    gt_by_image: dict[str, list[OBBDetection]],
    iou_threshold: float = 0.5,
    iou_mode: IoUMode = "obb",
) -> MatchingSummary:
    """Match predictions to GT across all images and aggregate."""
    all_outcomes: list[MatchOutcome] = []
    total_gt = 0
    total_preds = 0
    num_tp = num_fp = num_fn = 0

    image_ids = sorted(set(pred_by_image) | set(gt_by_image))
    for image_id in image_ids:
        preds = pred_by_image.get(image_id, [])
        gts = gt_by_image.get(image_id, [])
        summary = match_predictions_to_gt(preds, gts, iou_threshold=iou_threshold, iou_mode=iou_mode)
        all_outcomes.extend(summary.outcomes)
        total_preds += summary.num_predictions
        total_gt += summary.num_gt
        num_tp += summary.num_tp
        num_fp += summary.num_fp
        num_fn += summary.num_fn

    confidences = np.array([o.prediction.confidence or 0.0 for o in all_outcomes], dtype=np.float64)
    correct = np.array([1 if o.is_tp else 0 for o in all_outcomes], dtype=np.float64)

    return MatchingSummary(
        outcomes=all_outcomes,
        num_predictions=total_preds,
        num_gt=total_gt,
        num_tp=num_tp,
        num_fp=num_fp,
        num_fn=num_fn,
        iou_threshold=iou_threshold,
        iou_mode=iou_mode,
        confidences=confidences,
        correct=correct,
    )


def predictions_from_yolo_result(result, image_id: str, conf_threshold: float) -> list[OBBDetection]:
    """Extract OBB predictions from a single Ultralytics Result."""
    if result.obb is None or len(result.obb) == 0:
        return []

    detections: list[OBBDetection] = []
    polys = result.obb.xyxyxyxy.cpu().numpy()
    confs = result.obb.conf.cpu().numpy()
    classes = result.obb.cls.cpu().numpy().astype(int)

    for poly, conf, cls_id in zip(polys, confs, classes, strict=True):
        if float(conf) < conf_threshold:
            continue
        detections.append(
            OBBDetection(
                class_id=int(cls_id),
                polygon=np.asarray(poly, dtype=np.float64),
                confidence=float(conf),
                image_id=image_id,
            )
        )
    return detections


def run_yolo_predictions(
    model,
    dataset: str | None = None,
    data_yaml: str | Path | None = None,
    conf_threshold: float = 0.001,
    imgsz: int = 640,
    device: str | None = None,
) -> dict[str, list[OBBDetection]]:
    """Run YOLO-OBB inference on the validation split."""
    pred_by_image: dict[str, list[OBBDetection]] = {}
    images = list_val_images(dataset=dataset, data_yaml=data_yaml)
    paths = [str(p) for _, p in images]
    if not paths:
        return pred_by_image

    from qgeocompress.compression.structural_low_rank import StructuralLowRankConv2d

    predict_kwargs = {
        "source": paths,
        "imgsz": imgsz,
        "device": device,
        "conf": conf_threshold,
        "verbose": False,
    }
    if any(isinstance(m, StructuralLowRankConv2d) for m in model.model.modules()):
        from qgeocompress.evaluation.obb_validate import predict_no_fuse

        results = predict_no_fuse(model, **predict_kwargs)
    else:
        results = model.predict(**predict_kwargs)
    # Do not key on result.path: for a list source Ultralytics labels results
    # "image0", "image1", ... , which silently matched every prediction against
    # an empty GT list (TP=0, CER@0.8=1.0). Predictions come back in input order.
    for (image_id, _), result in zip(images, results, strict=True):
        pred_by_image[image_id] = predictions_from_yolo_result(result, image_id, conf_threshold)
    return pred_by_image
