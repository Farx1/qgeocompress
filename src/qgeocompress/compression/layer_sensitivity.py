from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from ultralytics import YOLO

from qgeocompress.compression.low_rank import count_params
from qgeocompress.compression.structural_low_rank import (
    StructuralLowRankConv2d,
    _rank_for_conv,
    _set_module,
    list_structural_candidates,
)
from qgeocompress.data.prepare_dota import get_data_yaml
from qgeocompress.evaluation.gt_matching import list_val_images
from qgeocompress.utils.config import project_root, save_json


def layer_param_gain_pct(conv: nn.Conv2d, rank_ratio: float) -> tuple[int, int, float]:
    before = count_params(conv)
    rank = _rank_for_conv(conv, rank_ratio)
    after = StructuralLowRankConv2d.param_count(conv, rank)
    gain = 100.0 * (before - after) / before if before else 0.0
    return before, after, round(gain, 4)


def compute_compressibility_score(
    param_gain_pct: float,
    local_output_error: float | None,
    map50_drop: float | None,
    eps: float = 1e-6,
) -> float:
    """Higher is better: large param gain with low activation/output impact."""
    err = (local_output_error or 0.0) + (map50_drop or 0.0)
    return round(param_gain_pct / (err + eps), 4)


def measure_local_output_error(
    baseline_model: nn.Module,
    layer_name: str,
    rank_ratio: float,
    sample_inputs: list[torch.Tensor],
) -> float:
    """Relative ||orig_out - factored_out|| / ||orig_out|| averaged over sample inputs."""
    module = baseline_model.get_submodule(layer_name)
    if not isinstance(module, nn.Conv2d):
        return 0.0

    factored = StructuralLowRankConv2d(module, _rank_for_conv(module, rank_ratio))
    errors: list[float] = []

    for x in sample_inputs:
        with torch.no_grad():
            orig = module(x)
            approx = factored(x)
            denom = orig.norm().item()
            if denom < 1e-9:
                continue
            rel = (orig - approx).norm().item() / denom
            errors.append(rel)

    if not errors:
        return 0.0
    return round(float(sum(errors) / len(errors)), 6)


def _collect_probe_inputs(
    model: YOLO,
    dataset: str | None,
    imgsz: int,
    device: str,
    max_batches: int = 4,
    data_yaml: str | Path | None = None,
) -> list[torch.Tensor]:
    """Capture intermediate-ready tensors by running the first backbone conv."""
    images = [str(p) for _, p in list_val_images(dataset=dataset, data_yaml=data_yaml)[:max_batches]]
    if not images:
        return []

    tensors: list[torch.Tensor] = []
    model.model.eval()
    for path in images:
        results = model.predict(path, imgsz=imgsz, device=device, verbose=False)
        if not results:
            continue
        im = results[0].orig_img
        if im is None:
            continue
        t = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensors.append(t.to(device))
    return tensors


def _replace_single_layer(
    pytorch_model: nn.Module,
    layer_name: str,
    rank_ratio: float,
) -> tuple[int, int]:
    conv = pytorch_model.get_submodule(layer_name)
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f"{layer_name} is not Conv2d")
    before = count_params(conv)
    rank = _rank_for_conv(conv, rank_ratio)
    after = StructuralLowRankConv2d.param_count(conv, rank)
    _set_module(pytorch_model, layer_name, StructuralLowRankConv2d(conv, rank))
    return before, after


def probe_single_layer(
    baseline_model: YOLO,
    layer_name: str,
    conv: nn.Conv2d,
    rank_ratio: float,
    baseline_map50: float,
    map50_after: float | None,
    local_output_error: float | None,
) -> dict[str, Any]:
    before, after, gain = layer_param_gain_pct(conv, rank_ratio)
    gain_abs = before - after
    drop = None if map50_after is None else round(max(0.0, baseline_map50 - map50_after), 6)
    score = compute_compressibility_score(gain, local_output_error, drop)
    return {
        "layer_name": layer_name,
        "params_before": before,
        "params_after": after,
        "param_gain_abs": gain_abs,
        "param_gain_pct": gain,
        "local_output_error": local_output_error,
        "map50_single_layer": map50_after,
        "map50_drop": drop,
        "compressibility_score": score,
    }


def select_layers_by_sensitivity(
    probe_results: list[dict[str, Any]],
    max_layers: int,
) -> list[str]:
    ranked = sorted(
        probe_results,
        key=lambda r: r.get("compressibility_score") or 0.0,
        reverse=True,
    )
    return [r["layer_name"] for r in ranked[:max_layers]]


def _param_gain_abs(row: dict[str, Any]) -> int:
    if row.get("param_gain_abs") is not None:
        return int(row["param_gain_abs"])
    before = row.get("params_before")
    after = row.get("params_after")
    if before is not None and after is not None:
        return int(before) - int(after)
    return 0


def select_layers_by_pareto_gain(
    probe_results: list[dict[str, Any]],
    max_layers: int,
    max_map50_drop: float = 0.03,
) -> list[str]:
    """Filter low-impact layers, then rank by absolute parameter savings."""
    eligible = [
        r for r in probe_results
        if (r.get("map50_drop") is None) or (r.get("map50_drop") or 0.0) <= max_map50_drop
    ]
    ranked = sorted(
        eligible,
        key=_param_gain_abs,
        reverse=True,
    )
    return [r["layer_name"] for r in ranked[:max_layers]]


def select_layers(
    probe_results: list[dict[str, Any]],
    max_layers: int,
    strategy: str = "sensitivity",
    max_map50_drop: float = 0.03,
    quantum_backend: str = "auto",
) -> list[str]:
    if strategy == "pareto-gain":
        return select_layers_by_pareto_gain(probe_results, max_layers, max_map50_drop=max_map50_drop)
    if strategy == "quantum-qaoa":
        from qgeocompress.quantum import select_layers_by_qaoa

        return select_layers_by_qaoa(
            probe_results, max_layers, backend=quantum_backend
        )
    return select_layers_by_sensitivity(probe_results, max_layers)


def _restore_single_layer(
    pytorch_model: nn.Module,
    layer_name: str,
    module: nn.Module,
) -> None:
    _set_module(pytorch_model, layer_name, module)


def run_structural_probe(
    weights: Path | str,
    dataset: str | None = "dota128",
    rank_ratio: float = 0.84,
    target_layers: list[str] | None = None,
    imgsz: int = 640,
    device: str = "cpu",
    eval_map50_fn: Any | None = None,
    max_probe_layers: int | None = None,
    activation_batches: int = 4,
    data_yaml: str | Path | None = None,
) -> dict[str, Any]:
    """Probe each candidate layer individually (replace → eval → discard)."""
    baseline = YOLO(str(weights), task="obb")
    baseline_map50: float | None = None
    if eval_map50_fn is not None:
        baseline_map50 = float(eval_map50_fn(baseline))

    candidates = list_structural_candidates(baseline.model, target_layers)
    if max_probe_layers is not None:
        candidates = candidates[:max_probe_layers]

    sample_inputs = _collect_probe_inputs(
        baseline, dataset, imgsz, device, activation_batches, data_yaml=data_yaml
    )
    layer_results: list[dict[str, Any]] = []

    for layer_name, conv in candidates:
        local_err = None
        if sample_inputs:
            try:
                local_err = measure_local_output_error(
                    baseline.model, layer_name, rank_ratio, sample_inputs
                )
            except Exception:
                local_err = None

        original_module = baseline.model.get_submodule(layer_name)
        saved_module = copy.deepcopy(original_module)
        _replace_single_layer(baseline.model, layer_name, rank_ratio)

        map50_after = None
        if eval_map50_fn is not None:
            map50_after = float(eval_map50_fn(baseline))

        _restore_single_layer(baseline.model, layer_name, saved_module)

        layer_results.append(
            probe_single_layer(
                baseline,
                layer_name,
                conv,
                rank_ratio,
                baseline_map50 or 0.0,
                map50_after,
                local_err,
            )
        )

    ranked = sorted(layer_results, key=lambda r: r["compressibility_score"], reverse=True)
    return {
        "method": "structural-probe",
        "rank_ratio": rank_ratio,
        "dataset": dataset,
        "data_yaml": str(data_yaml) if data_yaml else None,
        "baseline_map50": baseline_map50,
        "num_candidates": len(candidates),
        "layers": ranked,
        "top_layers": [r["layer_name"] for r in ranked[:10]],
    }


def save_sensitivity_report(
    report: dict[str, Any],
    path: Path | None = None,
    run_label: str | None = None,
) -> Path:
    if path is None:
        stem = "structural_layer_sensitivity"
        if run_label:
            stem = f"{stem}_{run_label}"
        path = project_root() / "results" / "summaries" / f"{stem}.json"
    save_json(report, path)
    return path


def load_sensitivity_report(path: Path | None = None) -> dict[str, Any]:
    path = path or project_root() / "results" / "summaries" / "structural_layer_sensitivity.json"
    import json

    return json.loads(path.read_text())


def _load_bchw_tensor(path: str, imgsz: int, device: str) -> torch.Tensor:
    import cv2
    import numpy as np
    from ultralytics.data.augment import LetterBox

    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(path)
    im = LetterBox(imgsz, auto=False, stride=32)(image=im)
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    tensor = torch.from_numpy(im).float().unsqueeze(0) / 255.0
    return tensor.to(device)


def recalibrate_batchnorm(
    model: YOLO,
    dataset: str | None = None,
    imgsz: int = 640,
    device: str = "cpu",
    num_batches: int = 20,
    data_yaml: str | Path | None = None,
) -> int:
    """Run train-mode forward passes to refresh BatchNorm running statistics."""
    images = [str(p) for _, p in list_val_images(dataset=dataset, data_yaml=data_yaml)[:num_batches]]
    if not images:
        return 0

    pytorch_model = model.model
    pytorch_model.train()
    ran = 0
    with torch.no_grad():
        for path in images:
            x = _load_bchw_tensor(path, imgsz, device)
            pytorch_model(x)
            ran += 1
    pytorch_model.eval()
    return ran
