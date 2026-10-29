"""Fixed-protocol exploratory DOTA128 pilot. See docs/research/margin_weighting.md."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from qgeocompress.compression.data_aware import collect_patch_gram, data_aware_factors, param_count
from qgeocompress.compression.layer_sensitivity import _load_bchw_tensor, capture_layer_inputs
from qgeocompress.compression.margin_weighting import margin_priority, normalized_priorities
from qgeocompress.compression.structural_low_rank import StructuralLowRankConv2d
from qgeocompress.evaluation.ap import map50_from_records, per_image_class_records
from qgeocompress.evaluation.gt_matching import (
    label_path_for_image,
    load_yolo_obb_label_file,
    predictions_from_yolo_result,
)
from qgeocompress.evaluation.obb_validate import predict_no_fuse

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    "seeds": [0, 1],
    "rank_ratios": [0.5, 0.7],
    "calibration_scenes": 16,
    "evaluation_scenes": 32,
    "imgsz": 640,
    "threshold": 0.25,
    "margin_floor": 0.05,
    "topk": 128,
    "max_patches": 2048,
    "num_layers": 2,
    "bootstrap_draws": 500,
    "prediction_conf": 0.001,
    "nms_iou": 0.7,
    "max_det": 300,
    "torch_threads": 2,
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_scenes(paths, seed):
    groups = {}
    for path in sorted(paths):
        groups.setdefault(path.stem.split("__")[0], []).append(path)
    scenes = sorted(groups)
    np.random.default_rng(seed).shuffle(scenes)
    ncal, neval = PROTOCOL["calibration_scenes"], PROTOCOL["evaluation_scenes"]
    if len(scenes) < ncal + neval:
        raise ValueError("Not enough distinct scenes for the fixed pilot protocol")
    # One deterministic tile per scene avoids within-arm correlated tiles.
    return [groups[s][0] for s in scenes[:ncal]], [groups[s][0] for s in scenes[ncal : ncal + neval]]


def raw_scores(model, paths):
    scores = []
    model.model.eval()
    nc = len(model.names)
    with torch.no_grad():
        for path in paths:
            out = model.model(_load_bchw_tensor(str(path), PROTOCOL["imgsz"], "cpu"))
            while isinstance(out, (tuple, list)):
                out = out[0]
            if out.ndim != 3 or out.shape[1] != 4 + nc + 1:
                raise ValueError(f"Unexpected OBB output shape {out.shape}")
            scores.append(out[0, 4 : 4 + nc].amax(dim=0).detach().cpu())
    return scores


def evaluate(model, paths, gt, reference_scores=None):
    # Read raw outputs before the predictor is initialized; disable fusion for EVERY arm.
    scores = raw_scores(model, paths)
    results = predict_no_fuse(
        model,
        source=[str(p) for p in paths],
        imgsz=PROTOCOL["imgsz"],
        device="cpu",
        conf=PROTOCOL["prediction_conf"],
        iou=PROTOCOL["nms_iou"],
        max_det=PROTOCOL["max_det"],
        batch=1,
        rect=False,
        verbose=False,
        save=False,
    )
    if len(results) != len(paths):
        raise RuntimeError("Prediction count differs from evaluation manifest")
    preds = {
        p.stem: predictions_from_yolo_result(r, p.stem, PROTOCOL["prediction_conf"]) for p, r in zip(paths, results, strict=True)
    }
    records = per_image_class_records(preds, gt, iou_threshold=0.5, iou_mode="obb")
    threshold = PROTOCOL["threshold"]
    selected = {key: [p for p in values if p.confidence >= threshold] for key, values in preds.items()}
    operating = per_image_class_records(selected, gt, iou_threshold=0.5, iou_mode="obb")
    tp = sum(sum(row["correct"]) for per_class in operating.values() for row in per_class.values())
    ngt = sum(row["num_gt"] for per_class in operating.values() for row in per_class.values())
    npred = sum(len(row["correct"]) for per_class in operating.values() for row in per_class.values())
    metrics = {
        "map50_repo": map50_from_records(records, [p.stem for p in paths]),
        "recall_at_025": tp / ngt if ngt else 0.0,
        "precision_at_025": tp / npred if npred else 0.0,
        "num_gt": ngt,
        "num_predictions_at_025": npred,
        "parameters": sum(p.numel() for p in model.model.parameters()),
    }
    if reference_scores is not None:
        flips, active_flips, mse = [], [], []
        for reference, score in zip(reference_scores, scores, strict=True):
            changed = (reference >= threshold) != (score >= threshold)
            active = (reference >= threshold) | (score >= threshold)
            flips.append(int(changed.sum()))
            active_flips.append(float(changed.sum() / active.sum()) if active.any() else 0.0)
            mse.append(float((reference - score).square().mean()))
        metrics.update(threshold_flips_per_image=flips, active_flip_fraction_per_image=active_flips, score_mse_per_image=mse)
    return metrics, records, scores


def paired_bootstrap(uniform, margin, ids, seed):
    rng = np.random.default_rng(seed + 10000)
    diffs = []
    for _ in range(PROTOCOL["bootstrap_draws"]):
        sample = rng.choice(ids, size=len(ids), replace=True).tolist()
        diffs.append(map50_from_records(margin, sample) - map50_from_records(uniform, sample))
    return {
        "delta_map50_margin_minus_uniform": map50_from_records(margin, ids) - map50_from_records(uniform, ids),
        "paired_scene_bootstrap_95_percentile": np.quantile(diffs, [0.025, 0.975]).tolist(),
    }


def run(checkpoint, output):
    torch.set_num_threads(PROTOCOL["torch_threads"])
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    paths = sorted((ROOT / "datasets/dota128/images/train").glob("*.jpg"))
    # Download checkpoint if absent via Ultralytics' standard loader.
    initial = YOLO(str(checkpoint))
    head_prefix = f"model.{len(initial.model.model) - 1}."
    candidates = [
        (n, c)
        for n, c in initial.model.named_modules()
        if isinstance(c, torch.nn.Conv2d)
        and c.groups == 1
        and c.kernel_size == (3, 3)
        and c.in_channels <= 128
        and not n.startswith(head_prefix)
    ]
    layers = [n for n, _ in sorted(candidates, key=lambda pair: (-pair[1].weight.numel(), pair[0]))[: PROTOCOL["num_layers"]]]
    if len(layers) != PROTOCOL["num_layers"]:
        raise ValueError("Too few eligible layers")
    report = {
        "protocol": PROTOCOL,
        "layers": layers,
        "checkpoint_sha256": digest(checkpoint),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "script_sha256": digest(Path(__file__)),
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "ultralytics", "numpy", "opencv-python"]},
        "runs": [],
    }
    for seed in PROTOCOL["seeds"]:
        cal, evaluation = split_scenes(paths, seed)
        gt = {}
        for p in evaluation:
            image = cv2.imread(str(p))
            if image is None or not label_path_for_image(p).exists():
                raise ValueError(f"Missing image or annotation: {p}")
            h, w = image.shape[:2]
            gt[p.stem] = load_yolo_obb_label_file(label_path_for_image(p), p.stem, w, h)
        model = YOLO(str(checkpoint))
        weights = normalized_priorities(
            [
                margin_priority(s, PROTOCOL["threshold"], PROTOCOL["margin_floor"], PROTOCOL["topk"])
                for s in raw_scores(model, cal)
            ]
        )
        captures = capture_layer_inputs(model.model, layers, [str(p) for p in cal], PROTOCOL["imgsz"], "cpu")
        grams = {"uniform": {}, "margin": {}}
        for name in layers:
            conv = model.model.get_submodule(name)
            image_grams = [
                collect_patch_gram(conv, [x], max_patches=PROTOCOL["max_patches"], seed=seed * 100 + i)
                for i, x in enumerate(captures[name])
            ]
            grams["uniform"][name] = sum(image_grams) / len(image_grams)
            grams["margin"][name] = sum(w * g for w, g in zip(weights, image_grams, strict=True)) / len(image_grams)
        del captures
        baseline, baseline_records, reference = evaluate(model, evaluation, gt)
        def manifest(items):
            return [
                {
                    "image": str(p.relative_to(ROOT)),
                    "sha256": digest(p),
                    "label_sha256": digest(label_path_for_image(p)),
                }
                for p in items
            ]
        run_result = {
            "seed": seed,
            "calibration": manifest(cal),
            "evaluation": manifest(evaluation),
            "weights": weights.tolist(),
            "effective_calibration_size": float(weights.sum().square() / weights.square().sum()),
            "baseline": baseline,
            "baseline_records": baseline_records,
            "arms": [],
            "comparisons": [],
        }
        print(f"seed={seed} baseline={baseline}", flush=True)
        for ratio in PROTOCOL["rank_ratios"]:
            pair = {}
            for method in ["plain", "uniform", "margin"]:
                variant = YOLO(str(checkpoint))
                ranks = {}
                for name in layers:
                    conv = variant.model.get_submodule(name)
                    rank = max(1, int(min(conv.out_channels, conv.weight[0].numel()) * ratio))
                    if param_count(conv, rank) >= sum(p.numel() for p in conv.parameters()):
                        raise ValueError("Protocol rank does not reduce parameters")
                    down, up = data_aware_factors(conv, rank, None if method == "plain" else grams[method][name])
                    parent, leaf = name.rsplit(".", 1)
                    setattr(variant.model.get_submodule(parent), leaf, StructuralLowRankConv2d.from_factors(conv, down, up))
                    ranks[name] = rank
                metrics, records, _ = evaluate(variant, evaluation, gt, reference)
                arm = {"method": method, "ratio": ratio, "ranks": ranks, "metrics": metrics, "records": records}
                run_result["arms"].append(arm)
                pair[method] = records
                print(f"seed={seed} ratio={ratio} method={method} AP={metrics['map50_repo']:.6f}", flush=True)
            run_result["comparisons"].append(
                {"ratio": ratio, **paired_bootstrap(pair["uniform"], pair["margin"], [p.stem for p in evaluation], seed)}
            )
        report["runs"].append(run_result)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "yolo11n-obb.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "results/research/margin_weighting_pilot.json")
    args = parser.parse_args()
    run(args.checkpoint, args.output)
