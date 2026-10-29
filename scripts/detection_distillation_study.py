"""Ablate second/fourth rectangle moments and per-anchor margins after low-rank compression."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from margin_weighting_study import PROTOCOL, ROOT, digest, evaluate, paired_bootstrap, split_scenes
from ultralytics import YOLO
from ultralytics.utils.ops import xywhr2xyxyxyxy

from qgeocompress.compression.data_aware import collect_patch_gram, data_aware_factors
from qgeocompress.compression.detection_moments import geometry_error
from qgeocompress.compression.layer_sensitivity import _load_bchw_tensor, capture_layer_inputs
from qgeocompress.compression.structural_low_rank import StructuralLowRankConv2d
from qgeocompress.evaluation.gt_matching import label_path_for_image, load_yolo_obb_label_file

CONFIG = {
    "seeds": [2, 3],
    "layers": ["model.7.conv", "model.20.conv"],
    "rank_ratio": 0.5,
    "steps": 64,
    "learning_rate": 1e-5,
    "score_coefficient": 10.0,
    "min_teacher_score": 0.05,
    "topk": 128,
    "margin_floor": 0.05,
    "arms": ["uniform", "covariance", "fourth", "fourth_margin"],
}


def decoded(model, tensor):
    output = model.model(tensor)
    while isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim != 3 or output.shape[1] != len(model.names) + 5:
        raise ValueError("Unexpected decoded OBB output")
    return output[0].T


def box_columns(output):
    return torch.cat((output[:, :4], output[:, -1:]), dim=-1)


def polygon_agreement(model, paths, teacher_outputs):
    """Teacher fidelity on aligned anchors; NOT ground-truth AP or a certificate."""
    rows = []
    with torch.no_grad():
        for path, target in zip(paths, teacher_outputs, strict=True):
            output = decoded(model, _load_bchw_tensor(str(path), 640, "cpu"))
            scores = target[:, 4:-1].amax(-1)
            indices = scores.topk(min(128, len(scores))).indices
            indices = indices[scores[indices] >= 0.25]
            original, compressed = box_columns(target[indices]), box_columns(output[indices])
            polygons_t, polygons_s = [xywhr2xyxyxyxy(b).numpy().astype(np.float32) for b in (original, compressed)]
            values, square_values = [], []
            for i, (a, b) in enumerate(zip(polygons_t, polygons_s, strict=True)):
                intersection = max(0.0, float(cv2.intersectConvexConvex(a, b)[0]))
                union = abs(cv2.contourArea(a)) + abs(cv2.contourArea(b)) - intersection
                iou = min(1.0, intersection / union) if union > 0 else 0.0
                values.append(iou)
                width, height = original[i, 2:4]
                if float(max(width, height) / min(width, height).clamp_min(1e-9)) <= 1.2:
                    square_values.append(iou)
            rows.append({"image": path.stem, "polygon_ious": values, "near_square_polygon_ious": square_values})
    return rows


def objective(output, teacher, indices, arm):
    output, teacher = output[indices], teacher[indices]
    score = (output[:, 4:-1] - teacher[:, 4:-1]).square().mean(-1)
    geometry = geometry_error(box_columns(output), box_columns(teacher), fourth=arm != "covariance")
    weights = torch.ones_like(score)
    if arm == "fourth_margin":
        weights = (teacher[:, 4:-1].amax(-1) - PROTOCOL["threshold"]).abs().clamp_min(CONFIG["margin_floor"]).pow(-2)
        weights = weights / weights.mean()
    return (weights * (CONFIG["score_coefficient"] * score + geometry)).mean()


def refine(initial, tensors, targets, selections, arm, seed):
    variant = copy.deepcopy(initial)
    variant.model.eval()
    for parameter in variant.model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name in CONFIG["layers"]:
        for parameter in variant.model.get_submodule(name).parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    optimizer = torch.optim.Adam(trainable, lr=CONFIG["learning_rate"])
    order = np.random.default_rng(seed).permutation(len(tensors))
    losses = []
    start = time.perf_counter()
    for step in range(CONFIG["steps"]):
        i = int(order[step % len(order)])
        optimizer.zero_grad(set_to_none=True)
        loss = objective(decoded(variant, tensors[i]), targets[i], selections[i], arm)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite distillation loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    return variant, {"losses": losses, "training_seconds": time.perf_counter() - start}


def run():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    checkpoint = ROOT / "yolo11n-obb.pt"
    output_path = ROOT / "results/research/detection_distillation_pilot.json"
    report = {
        "config": CONFIG,
        "inherited_protocol": PROTOCOL,
        "checkpoint_sha256": digest(checkpoint),
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                Path(__file__),
                ROOT / "src/qgeocompress/compression/detection_moments.py",
                ROOT / "scripts/margin_weighting_study.py",
            ]
        },
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "runs": [],
    }
    for seed in CONFIG["seeds"]:
        cal, evaluation = split_scenes(sorted((ROOT / "datasets/dota128/images/train").glob("*.jpg")), seed)
        teacher = YOLO(str(checkpoint))
        teacher.model.eval()
        tensors = [_load_bchw_tensor(str(p), PROTOCOL["imgsz"], "cpu") for p in cal]
        with torch.no_grad():
            targets = [decoded(teacher, tensor).detach().clone() for tensor in tensors]
        selections = []
        for target in targets:
            scores = target[:, 4:-1].amax(-1)
            indices = scores.topk(min(CONFIG["topk"], len(scores))).indices
            selected = indices[scores[indices] >= CONFIG["min_teacher_score"]]
            selections.append(selected if len(selected) else indices[:1])
        initial = YOLO(str(checkpoint))
        captured = capture_layer_inputs(teacher.model, CONFIG["layers"], [str(p) for p in cal], 640, "cpu")
        for name in CONFIG["layers"]:
            conv = initial.model.get_submodule(name)
            gram = collect_patch_gram(conv, captured[name], max_patches=2048, seed=seed)
            rank = int(conv.out_channels * CONFIG["rank_ratio"])
            down, up = data_aware_factors(conv, rank, gram)
            parent, leaf = name.rsplit(".", 1)
            setattr(initial.model.get_submodule(parent), leaf, StructuralLowRankConv2d.from_factors(conv, down, up))
        del captured
        gt = {}
        for path in evaluation:
            image = cv2.imread(str(path))
            if image is None or not label_path_for_image(path).exists():
                raise ValueError(f"Missing image or labels: {path}")
            gt[path.stem] = load_yolo_obb_label_file(label_path_for_image(path), path.stem, image.shape[1], image.shape[0])
        with torch.no_grad():
            teacher_outputs = [decoded(teacher, _load_bchw_tensor(str(p), 640, "cpu")).detach().clone() for p in evaluation]
        baseline, baseline_records, reference = evaluate(teacher, evaluation, gt)
        run_result = {
            "seed": seed,
            "baseline": baseline,
            "baseline_records": baseline_records,
            "calibration": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in cal],
            "evaluation": [
                {"path": str(p.relative_to(ROOT)), "sha256": digest(p), "label_sha256": digest(label_path_for_image(p))}
                for p in evaluation
            ],
            "arms": [],
            "comparisons": [],
        }
        records_by_arm = {}
        for arm in CONFIG["arms"]:
            if arm == "uniform":
                variant, training = copy.deepcopy(initial), None
            else:
                variant, training = refine(initial, tensors, targets, selections, arm, seed)
            agreement = polygon_agreement(variant, evaluation, teacher_outputs)
            metrics, records, _ = evaluate(variant, evaluation, gt, reference)
            metrics["teacher_polygon_agreement"] = agreement
            records_by_arm[arm] = records
            run_result["arms"].append({"method": arm, "metrics": metrics, "records": records, "training": training})
            print(f"seed={seed} arm={arm} AP={metrics['map50_repo']:.8f}", flush=True)
        for control, treatment in [("covariance", "fourth"), ("fourth", "fourth_margin"), ("uniform", "fourth")]:
            contrast = paired_bootstrap(records_by_arm[control], records_by_arm[treatment], [p.stem for p in evaluation], seed)
            contrast["delta_map50"] = contrast.pop("delta_map50_margin_minus_uniform")
            run_result["comparisons"].append({"control": control, "treatment": treatment, **contrast})
        report["runs"].append(run_result)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    run()
