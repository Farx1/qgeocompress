#!/usr/bin/env python
"""Sweep the compression frontier and test which mathematical choices actually pay.

Four hypotheses, each isolated so the comparison is one variable at a time:

  H1  rank ratio        r=0.84 caps total savings at 7.3% of the model. Sweep lower.
  H2  layer coverage    1x1 convs hold 39% of the weights and were excluded outright.
  H3  data-aware SVD    minimize ||(W-W')X|| over real activations, not ||W-W'||_F.
  H4  rank allocation   per-layer rank from the spectrum (energy / budget), not a
                        single global ratio.

Every arm is scored on the same held-out images with a paired bootstrap against
the baseline, because on this dataset an unpaired mAP50 has a +/-0.06 spread and
cannot separate anything.

Usage: python scripts/frontier_study.py [--split-dir DIR] [--device cpu]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from ultralytics import YOLO

from qgeocompress.compression.data_aware import (
    allocate_ranks_by_budget,
    collect_cross_grams,
    collect_patch_gram,
    data_aware_factors,
    drift_corrected_weight,
    layer_importance,
    param_count,
    rank_for_energy,
    transfer_importance,
)
from qgeocompress.compression.factorized import (
    FactorizedConv2d,
    allocate_from_curves,
    importance_from_curves,
    measure_scheme_curves,
    refit_second_factor,
    scheme_params,
)
from qgeocompress.compression.layer_sensitivity import (
    _load_bchw_tensor,
    capture_layer_inputs,
    recalibrate_batchnorm,
)
from qgeocompress.compression.structural_low_rank import (
    StructuralLowRankConv2d,
    _set_module,
    max_useful_rank,
)
from qgeocompress.evaluation.ap import (
    bootstrap_map50,
    paired_delta_map50,
    per_image_class_records,
)
from qgeocompress.evaluation.calibration_real import compute_calibration_metrics
from qgeocompress.evaluation.gt_matching import (
    list_val_images,
    load_ground_truth_for_dataset,
    match_dataset_predictions,
    run_yolo_predictions,
)
from qgeocompress.utils.config import project_root, save_json


def eligible_convs(
    model: nn.Module, include_1x1: bool, include_head: bool = False
) -> list[tuple[str, nn.Conv2d]]:
    """Factorizable convs outside the detection head.

    Grouped convs are skipped (the factorization assumes dense channel mixing)
    and so are very narrow layers, where the two factors cost more than the
    original.
    """
    out: list[tuple[str, nn.Conv2d]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d) or module.groups != 1:
            continue
        if name.startswith("model.23") and not include_head:  # detection head
            continue
        k = module.kernel_size[0]
        if k not in (1, 3) or (k == 1 and not include_1x1):
            continue
        if min(module.in_channels, module.out_channels) < 8:
            continue
        out.append((name, module))
    return out


def build_variant(
    weights: str,
    ranks: dict[str, int],
    grams: dict[str, torch.Tensor] | None,
) -> tuple[YOLO, dict[str, Any]]:
    """Replace every named conv with a rank-r block, data-aware when grams given.

    Loads a fresh checkpoint per arm: a YOLO wrapper holds thread locks and
    cannot be deep-copied.
    """
    model = YOLO(weights, task="obb")
    original = sum(p.numel() for p in model.model.parameters())
    saved = 0

    for name, rank in ranks.items():
        conv = model.model.get_submodule(name)
        gram = (grams or {}).get(name)
        before = sum(p.numel() for p in conv.parameters())
        after = param_count(conv, rank)
        if after >= before:  # factorization would cost more than it saves
            continue
        down, up = data_aware_factors(conv, rank, gram)
        _set_module(model.model, name, StructuralLowRankConv2d.from_factors(conv, down, up))
        saved += before - after

    compressed = sum(p.numel() for p in model.model.parameters())
    return model, {
        "original_params": original,
        "compressed_params": compressed,
        "param_reduction_pct": round(100.0 * (original - compressed) / original, 3),
        "layers_replaced": len(ranks),
    }



def _final_output(model: YOLO, x: torch.Tensor) -> torch.Tensor:
    out = model.model(x)
    while isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _capture_one(module: nn.Module, model: YOLO, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Inputs seen by ``module`` when ``model`` runs over ``tensors``."""
    seen: list[torch.Tensor] = []
    handle = module.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach()))
    try:
        model.model.eval()
        with torch.no_grad():
            for x in tensors:
                model.model(x)
    finally:
        handle.remove()
    return seen


def build_variant_sequential(
    weights: str,
    ranks: dict[str, int],
    probe_tensors: list[torch.Tensor],
) -> tuple[YOLO, dict[str, Any]]:
    """Factorize in forward order, re-fitting each layer against the drifted input."""
    reference = YOLO(weights, task="obb")
    model = YOLO(weights, task="obb")
    original = sum(p.numel() for p in model.model.parameters())

    for name, rank in ranks.items():
        conv = model.model.get_submodule(name)
        ref_conv = reference.model.get_submodule(name)
        if param_count(conv, rank) >= sum(p.numel() for p in conv.parameters()):
            continue
        clean = _capture_one(ref_conv, reference, probe_tensors)
        drifted = _capture_one(conv, model, probe_tensors)
        cross, gram = collect_cross_grams(ref_conv, clean, drifted)
        fitted = drift_corrected_weight(ref_conv, cross, gram)
        down, up = data_aware_factors(ref_conv, rank, gram, weight=fitted)
        _set_module(model.model, name, StructuralLowRankConv2d.from_factors(conv, down, up))

    compressed = sum(p.numel() for p in model.model.parameters())
    return model, {
        "original_params": original,
        "compressed_params": compressed,
        "param_reduction_pct": round(100.0 * (original - compressed) / original, 3),
        "layers_replaced": len(ranks),
    }


def measure_importance(
    weights: str,
    convs: dict[str, nn.Conv2d],
    grams: dict[str, torch.Tensor],
    probe_tensors: list[torch.Tensor],
    probe_ratio: float = 0.5,
) -> dict[str, float]:
    """Relative change in the model's final output when one layer is factorized.

    One forward per layer. Spectral tail energy says how much of a layer's own
    output is lost; this says how much of the *network's* output moves, which is
    what the allocation should be pricing.
    """
    model = YOLO(weights, task="obb")
    model.model.eval()
    with torch.no_grad():
        reference = [_final_output(model, x).clone() for x in probe_tensors]

    sensitivity: dict[str, float] = {}
    for name, conv in convs.items():
        original = model.model.get_submodule(name)
        rank = max(1, int(max_useful_rank(conv) * probe_ratio))
        down, up = data_aware_factors(original, rank, grams.get(name))
        _set_module(model.model, name, StructuralLowRankConv2d.from_factors(original, down, up))
        num = den = 0.0
        with torch.no_grad():
            for x, ref in zip(probe_tensors, reference, strict=True):
                out = _final_output(model, x)
                num += float(((out - ref) ** 2).sum())
                den += float((ref**2).sum())
        _set_module(model.model, name, original)
        sensitivity[name] = (num / den) ** 0.5 if den > 0 else 0.0
    return sensitivity


def build_variant_multi_scheme(
    weights: str,
    allocation: dict[str, tuple[str, int]],
    probe_tensors: list[torch.Tensor],
    refit: bool = True,
    grams_for_init: dict[str, torch.Tensor] | None = None,
) -> tuple[YOLO, dict[str, Any]]:
    """Replace each conv with its own best scheme, then refit the second factor.

    Inputs are captured from the ORIGINAL model: the refit already absorbs the
    per-layer approximation error, and chaining it on drifted inputs was
    measured to lose against the parallel pass once a budget allocation is in
    play.
    """
    reference = YOLO(weights, task="obb")
    model = YOLO(weights, task="obb")
    original = sum(p.numel() for p in model.model.parameters())

    inputs_by_layer = _capture_many(reference, list(allocation), probe_tensors)

    scheme_counts: dict[str, int] = {}
    for name, (scheme, rank) in allocation.items():
        if scheme == "none":
            continue
        conv = model.model.get_submodule(name)
        if scheme_params(conv, scheme, rank) >= sum(p.numel() for p in conv.parameters()):
            continue
        block = FactorizedConv2d(conv, scheme, rank)
        if scheme == "spatial-first" and grams_for_init and name in grams_for_init:
            # spatial-first is the one scheme whose whitened optimum is a plain
            # SVD, so start it from the data-aware factors rather than handing it
            # a worse initialisation than the single-scheme arm gets.
            down, up = data_aware_factors(conv, rank, grams_for_init[name])
            with torch.no_grad():
                block.down.weight.copy_(down)
                block.up.weight.copy_(up)
        if refit and inputs_by_layer.get(name):
            refit_second_factor(block, conv, inputs_by_layer[name])
        _set_module(model.model, name, block)
        scheme_counts[scheme] = scheme_counts.get(scheme, 0) + 1

    compressed = sum(p.numel() for p in model.model.parameters())
    return model, {
        "original_params": original,
        "compressed_params": compressed,
        "param_reduction_pct": round(100.0 * (original - compressed) / original, 3),
        "layers_replaced": sum(scheme_counts.values()),
        "schemes": scheme_counts,
    }


def _capture_many(
    model: YOLO, names: list[str], tensors: list[torch.Tensor]
) -> dict[str, list[torch.Tensor]]:
    """Inputs every named layer sees, in one forward per image."""
    captured: dict[str, list[torch.Tensor]] = {name: [] for name in names}

    def hook(name: str):
        return lambda _m, args: captured[name].append(args[0].detach())

    handles = [model.model.get_submodule(n).register_forward_pre_hook(hook(n)) for n in names]
    try:
        model.model.eval()
        with torch.no_grad():
            for x in tensors:
                model.model(x)
    finally:
        for h in handles:
            h.remove()
    return captured


def evaluate(
    model: YOLO, test_yaml: str, gt: dict, device: str, imgsz: int = 640
) -> tuple[dict[str, Any], dict[str, float]]:
    """One prediction pass, returning both the AP records and the gate metrics.

    Predictions dominate the cost of an arm, so scoring accuracy and reliability
    from a single pass halves the study's runtime.
    """
    preds = run_yolo_predictions(
        model, data_yaml=test_yaml, conf_threshold=0.001, imgsz=imgsz, device=device
    )
    records = per_image_class_records(preds, gt)
    metrics = compute_calibration_metrics(match_dataset_predictions(preds, gt, iou_threshold=0.5))
    return records, {"ece": metrics["ece"], "cer08": metrics["confident_error_rate_08"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", default="datasets/dota128_power")
    parser.add_argument("--weights", default="yolo11n-obb.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--budgets", type=float, nargs="+", default=None,
                        help="Override the data-aware budget sweep (percent of model params).")
    parser.add_argument("--old-ratios", type=float, nargs="+", default=None,
                        help="Override the plain-SVD rank ratios compared against.")
    parser.add_argument("--gram-images", type=int, default=16)
    parser.add_argument("--seq-images", type=int, default=16,
                        help="Probe images for the drift-corrected pass. Deep layers see only "
                             "~400 patch positions per image against d up to 2304, so too few "
                             "images make the refit underdetermined and it loses to the parallel pass.")
    parser.add_argument("--bn-batches", type=int, default=4,
                    help="Kept small on purpose: see recalibrate_batchnorm. Large values "
                         "overwrite the pretrained statistics with a 24-image estimate and "
                         "cost 0.077 mAP50 even with no compression at all.")
    parser.add_argument("--resamples", type=int, default=400)
    parser.add_argument("--preset", choices=["hypotheses", "frontier", "multi"], default="hypotheses",
                        help="hypotheses: isolate H1-H6. frontier: fine sweep of the old method "
                             "against the winning one, for an equal-cost comparison.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    split = Path(args.split_dir)

    def yaml_for(role: str) -> str:
        # create_dota128_holdout.py keeps its own stem regardless of --out.
        hits = sorted(split.glob(f"*_{role}.yaml"))
        if not hits:
            raise SystemExit(f"no *_{role}.yaml under {split}")
        return str(hits[0])

    test_yaml, select_yaml, train_yaml = yaml_for("test"), yaml_for("select"), yaml_for("train")

    baseline = YOLO(args.weights, task="obb")
    gt = load_ground_truth_for_dataset(data_yaml=test_yaml)
    print(f"test images: {len(gt)}  gt boxes: {sum(len(v) for v in gt.values())}")

    # Activation statistics come from the select split — never from test.
    probe_images = [str(p) for _, p in list_val_images(data_yaml=select_yaml)[: args.gram_images]]
    all_convs = dict(eligible_convs(baseline.model, include_1x1=True, include_head=True))
    print(f"capturing activations for {len(all_convs)} convs on {len(probe_images)} images")
    captured = capture_layer_inputs(baseline.model, list(all_convs), probe_images, args.imgsz, args.device)

    grams: dict[str, torch.Tensor] = {}
    for name, conv in all_convs.items():
        grams[name] = collect_patch_gram(conv, captured.pop(name), max_patches=200_000)
    del captured

    convs_3x3 = dict(eligible_convs(baseline.model, include_1x1=False))
    convs_all = dict(eligible_convs(baseline.model, include_1x1=True))  # head excluded

    def ratio_ranks(convs: dict[str, nn.Conv2d], ratio: float) -> dict[str, int]:
        return {n: max(1, int(max_useful_rank(c) * ratio)) for n, c in convs.items()}

    def energy_ranks(convs: dict[str, nn.Conv2d], energy: float) -> dict[str, int]:
        return {n: rank_for_energy(c, energy, grams[n]) for n, c in convs.items()}

    probe_tensors = [_load_bchw_tensor(p, args.imgsz, args.device) for p in probe_images[: args.seq_images]]
    print("measuring end-to-end layer sensitivity")
    sensitivity = measure_importance(args.weights, convs_all, grams, probe_tensors[:3])
    importance = transfer_importance(
        sensitivity, {n: {"conv": c, "gram": grams[n]} for n, c in convs_all.items()}
    )
    importance_raw = layer_importance(sensitivity)
    ranked = sorted(sensitivity.items(), key=lambda kv: -kv[1])[:5]
    print("  most sensitive:", ", ".join(f"{n}={v:.3f}" for n, v in ranked))

    def budget_ranks(
        convs: dict[str, nn.Conv2d], pct: float, weighted: str | bool = False
    ) -> dict[str, int]:
        total = sum(p.numel() for p in baseline.model.parameters())
        return allocate_ranks_by_budget(
            {n: {"conv": c, "gram": grams[n]} for n, c in convs.items()},
            target_saved_params=int(total * pct / 100.0),
            importance={"transfer": importance, "raw": importance_raw}.get(weighted)
            if weighted else None,
        )

    def run_arms(arms: list[dict[str, Any]]) -> None:
        print("scoring baseline")
        base_records, base_reliability = evaluate(baseline, test_yaml, gt, args.device, args.imgsz)
        base_boot = bootstrap_map50(base_records, n_resamples=args.resamples)
        rows: list[dict[str, Any]] = [{
            "arm": "baseline",
            "param_reduction_pct": 0.0,
            "map50": round(base_boot["map50"], 4),
            "ci_low": round(base_boot["ci_low"], 4),
            "ci_high": round(base_boot["ci_high"], 4),
            "delta": 0.0,
            "delta_ci_low": 0.0,
            "delta_ci_high": 0.0,
            **base_reliability,
        }]

        for arm in arms:
            started = time.time()
            if arm.get("mode") == "multi-scheme":
                model, meta = build_variant_multi_scheme(
                    args.weights,
                    arm["allocation"],
                    probe_tensors[:4],
                    refit=arm.get("refit", True),
                    grams_for_init=grams,
                )
            elif arm.get("mode") == "sequential":
                model, meta = build_variant_sequential(args.weights, arm["ranks"], probe_tensors)
            else:
                model, meta = build_variant(args.weights, arm["ranks"], grams if arm["gram"] else None)
            recalibrate_batchnorm(
                model,
                data_yaml=train_yaml,
                num_batches=args.bn_batches,
                device=args.device,
                imgsz=args.imgsz,
            )
            records, arm_reliability = evaluate(model, test_yaml, gt, args.device, args.imgsz)
            boot = bootstrap_map50(records, n_resamples=args.resamples)
            delta = paired_delta_map50(base_records, records, n_resamples=args.resamples)
            rows.append({
                "arm": arm["name"],
                "param_reduction_pct": meta["param_reduction_pct"],
                "map50": round(boot["map50"], 4),
                "ci_low": round(boot["ci_low"], 4),
                "ci_high": round(boot["ci_high"], 4),
                "delta": round(delta["delta"], 4),
                "delta_ci_low": round(delta["ci_low"], 4),
                "delta_ci_high": round(delta["ci_high"], 4),
                "layers_replaced": meta["layers_replaced"],
                **arm_reliability,
            })
            print(
                f"{arm['name']:<34} -{meta['param_reduction_pct']:5.2f}% params  "
                f"mAP50 {boot['map50']:.4f}  delta {delta['delta']:+.4f} "
                f"[{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]  "
                f"CER@0.8 {rows[-1]['cer08']:.4f}  ({time.time() - started:.0f}s)",
                flush=True,
            )

        out = args.out or project_root() / "results" / "summaries" / f"frontier_{args.preset}.json"
        save_json(
            {"split": str(split), "weights": args.weights, "preset": args.preset, "rows": rows}, out
        )
        print(f"\nsaved {out}")

        report = project_root() / "results" / "reports" / f"qgeocompress_frontier_{args.preset}.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Q-GEOCompress — compression frontier",
            "",
            f"Split `{split.name}` ({len(gt)} test images, "
            f"{sum(len(v) for v in gt.values())} GT boxes). Generated by `scripts/frontier_study.py "
            f"--preset {args.preset}`.",
            "",
            "mAP50 is computed from this repo's own GT matching so it can be bootstrapped; it "
            "runs lower than the Ultralytics figure because per-class AP on a small split "
            "punishes rare classes. Every arm is scored identically, and `delta` is a **paired** "
            "bootstrap over images, which is the only comparison this sample size supports.",
            "",
            "| Arm | Params Δ | mAP50 | delta vs baseline (95% CI) | ECE | CER@0.8 |",
            "| --- | -------: | ----: | -------------------------- | --: | ------: |",
        ]
        for row in rows:
            ci = (
                "reference" if row["arm"] == "baseline"
                else f"{row['delta']:+.4f} [{row['delta_ci_low']:+.4f}, {row['delta_ci_high']:+.4f}]"
            )
            lines.append(
                f"| {row['arm']} | {row['param_reduction_pct']:.2f}% | {row['map50']:.4f} | "
                f"{ci} | {row['ece']:.4f} | {row['cer08']:.4f} |"
            )
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"saved {report}")

    arms: list[dict[str, Any]] = []
    if args.preset == "multi":
        total_params = sum(p.numel() for p in baseline.model.parameters())
        # Including the head was measured at mAP50 0.15-0.18 at every budget:
        # its layers only save parameters at ranks that destroy them.
        for label, convs in (("", convs_all),):
            print(f"measuring scheme curves{label} ({len(convs)} convs)")
            curves = measure_scheme_curves(
                convs, _capture_many(baseline, list(convs), probe_tensors[:3])
            )
            curve_importance = importance_from_curves(curves, sensitivity)
            for pct in (20.0, 30.0, 35.0, 40.0):
                arms.append({
                    "name": f"multi-scheme{label} budget {pct:.0f}%",
                    "allocation": allocate_from_curves(
                        curves, int(total_params * pct / 100.0), importance=curve_importance
                    ),
                    "mode": "multi-scheme",
                    "gram": False,
                })
        run_arms(arms)
        return

    if args.preset == "frontier":
        # Old method: plain SVD, uniform ratio, 3x3 only — what the project shipped.
        for r in args.old_ratios or (0.95, 0.9, 0.84, 0.8, 0.75, 0.7):
            arms.append({
                "name": f"old plain-svd r={r}",
                "ranks": ratio_ranks(convs_3x3, r),
                "gram": False,
            })
        # New method: data-aware factors, all convs, importance-weighted budget.
        for pct in args.budgets or (5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0):
            arms.append({
                "name": f"new data-aware budget {pct:.0f}%",
                "ranks": budget_ranks(convs_all, pct, weighted="transfer"),
                "gram": True,
            })
        run_arms(arms)
        return

    for r in (0.7, 0.5, 0.3):
        arms.append({"name": f"H1 plain-svd 3x3 r={r}", "ranks": ratio_ranks(convs_3x3, r), "gram": False})
    for r in (0.7, 0.5, 0.3):
        arms.append({"name": f"H3 data-aware 3x3 r={r}", "ranks": ratio_ranks(convs_3x3, r), "gram": True})
    for r in (0.7, 0.5, 0.3):
        arms.append({"name": f"H2+H3 data-aware 3x3+1x1 r={r}", "ranks": ratio_ranks(convs_all, r), "gram": True})
    for e in (0.99, 0.95, 0.90):
        arms.append({"name": f"H4 energy tau={e}", "ranks": energy_ranks(convs_all, e), "gram": True})
    for pct in (10.0, 20.0, 30.0):
        arms.append({"name": f"H4 budget {pct:.0f}%", "ranks": budget_ranks(convs_all, pct), "gram": True})
    for pct in (10.0, 20.0, 30.0):
        arms.append({
            "name": f"H5 budget {pct:.0f}% + importance",
            "ranks": budget_ranks(convs_all, pct, weighted="transfer"),
            "gram": True,
        })
    arms.append({
        "name": "H6 sequential 3x3 r=0.5",
        "ranks": ratio_ranks(convs_3x3, 0.5),
        "gram": True,
        "mode": "sequential",
    })
    for pct in (10.0, 20.0, 30.0):
        arms.append({
            "name": f"H6 sequential budget {pct:.0f}% + importance",
            "ranks": budget_ranks(convs_all, pct, weighted="transfer"),
            "gram": True,
            "mode": "sequential",
        })

    run_arms(arms)


if __name__ == "__main__":
    main()
