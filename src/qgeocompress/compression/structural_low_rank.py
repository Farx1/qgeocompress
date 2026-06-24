from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from ultralytics import YOLO

from qgeocompress.compression.low_rank import _should_replace_layer, count_params


class StructuralLowRankConv2d(nn.Module):
    """Factorize Conv2d(in, out, k, k) as Conv2d(in, r, k, k) + Conv2d(r, out, 1, 1)."""

    def __init__(self, conv: nn.Conv2d, rank: int):
        super().__init__()
        if conv.kernel_size[0] != conv.kernel_size[1]:
            raise ValueError("Structural low-rank currently supports square kernels only")

        c_in, c_out = conv.in_channels, conv.out_channels
        k = conv.kernel_size[0]
        rank = max(1, min(rank, c_in, c_out))

        self.rank = rank
        self.in_channels = c_in
        self.out_channels = c_out
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.down = nn.Conv2d(
            c_in,
            rank,
            kernel_size=k,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=False,
        )
        self.up = nn.Conv2d(
            rank,
            c_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=conv.bias is not None,
        )
        self._init_from_svd(conv, rank)

    def _init_from_svd(self, conv: nn.Conv2d, rank: int) -> None:
        w = conv.weight.data
        c_out, c_in, kh, kw = w.shape
        w2d = w.reshape(c_out, -1)
        u, s, vh = torch.linalg.svd(w2d, full_matrices=False)
        rank = min(rank, s.shape[0])
        self.rank = rank

        down_w = vh[:rank].reshape(rank, c_in, kh, kw)
        up_w = (u[:, :rank] * s[:rank]).reshape(c_out, rank, 1, 1)

        self.down.weight.data.copy_(down_w)
        self.up.weight.data.copy_(up_w)
        if conv.bias is not None and self.up.bias is not None:
            self.up.bias.data.copy_(conv.bias.data)

    @classmethod
    def param_count(cls, conv: nn.Conv2d, rank: int) -> int:
        c_in, c_out = conv.in_channels, conv.out_channels
        k = conv.kernel_size[0]
        rank = max(1, min(rank, c_in, c_out))
        n = c_in * rank * k * k + rank * c_out
        if conv.bias is not None:
            n += c_out
        return n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


def _rank_for_conv(conv: nn.Conv2d, rank_ratio: float) -> int:
    c_out, c_in = conv.out_channels, conv.in_channels
    k = conv.kernel_size[0]
    flat = c_out * c_in * k * k
    w2d = conv.weight.data.reshape(c_out, -1)
    max_rank = min(w2d.shape[0], w2d.shape[1])
    return max(1, int(max_rank * rank_ratio))


def _set_module(root: nn.Module, dotted_name: str, module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def list_structural_candidates(
    pytorch_model: nn.Module,
    target_layers: list[str] | None = None,
) -> list[tuple[str, nn.Conv2d]]:
    """Return (layer_name, conv_module) pairs eligible for structural factorization."""
    candidates: list[tuple[str, nn.Conv2d]] = []
    for name, module in pytorch_model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if module.groups != 1:
            continue
        if not _should_replace_layer(name, module, target_layers):
            continue
        candidates.append((name, module))
    return candidates


def apply_structural_low_rank(
    model: YOLO,
    rank_ratio: float = 0.84,
    target_layers: list[str] | None = None,
    target_scope: str = "backbone-neck",
    selected_layer_names: list[str] | None = None,
    max_replaced_layers: int | None = None,
    selection_strategy: str = "all",
    sensitivity_report: dict[str, Any] | None = None,
    max_map50_drop: float = 0.03,
) -> tuple[YOLO, dict[str, Any]]:
    """Replace selected Conv2d layers with structural low-rank blocks."""
    pytorch_model = copy.deepcopy(model.model)
    original_params = count_params(pytorch_model)
    rank_ratio = max(0.05, min(1.0, rank_ratio))

    layers = target_layers
    if layers is None and target_scope == "backbone-neck":
        layers = ["backbone", "neck"]

    candidates = [
        name
        for name, module in list_structural_candidates(pytorch_model, layers)
    ]

    if selected_layer_names is not None:
        replace_names = [n for n in selected_layer_names if n in candidates]
    elif selection_strategy in ("sensitivity", "pareto-gain") and sensitivity_report:
        from qgeocompress.compression.layer_sensitivity import select_layers

        limit = max_replaced_layers or len(candidates)
        replace_names = select_layers(
            sensitivity_report.get("layers", []),
            limit,
            strategy=selection_strategy,
            max_map50_drop=max_map50_drop,
        )
        replace_names = [n for n in replace_names if n in candidates]
    elif max_replaced_layers is not None:
        replace_names = candidates[:max_replaced_layers]
    else:
        replace_names = candidates

    replace_set = set(replace_names)
    replaced = 0
    replaced_param_before = 0
    replaced_param_after = 0
    layer_names: list[str] = []

    for name, module in list(pytorch_model.named_modules()):
        if name not in replace_set:
            continue
        if not isinstance(module, nn.Conv2d):
            continue

        rank = _rank_for_conv(module, rank_ratio)
        before = count_params(module)
        after = StructuralLowRankConv2d.param_count(module, rank)
        structural = StructuralLowRankConv2d(module, rank)
        _set_module(pytorch_model, name, structural)

        replaced += 1
        replaced_param_before += before
        replaced_param_after += after
        layer_names.append(name)

    compressed_params = count_params(pytorch_model)
    layer_savings = replaced_param_before - replaced_param_after
    real_reduction = 100.0 * layer_savings / original_params if original_params else 0.0

    meta: dict[str, Any] = {
        "method": "structural-low-rank",
        "compression_mode": "structural_factorization",
        "structural_compression": True,
        "rank_ratio": rank_ratio,
        "target_scope": target_scope,
        "target_layers": layers or ["backbone", "neck"],
        "selection_strategy": selection_strategy,
        "max_map50_drop": max_map50_drop if selection_strategy == "pareto-gain" else None,
        "max_replaced_layers": max_replaced_layers,
        "selected_layer_names": layer_names,
        "num_replaced_layers": replaced,
        "layers_replaced": replaced,
        "replaced_layer_names": layer_names[:10],
        "original_params": original_params,
        "compressed_params": compressed_params,
        "replaced_params_before": replaced_param_before,
        "replaced_params_after": replaced_param_after,
        "real_param_reduction_pct": round(real_reduction, 2),
        "param_reduction_pct": round(real_reduction, 2),
        "collapsed_for_ultralytics_compat": False,
    }
    model.model = pytorch_model
    return model, meta
