"""Collapse StructuralLowRankConv2d layers back to standard Conv2d for export compatibility."""

from __future__ import annotations

import copy
from typing import Any

import torch.nn as nn
from ultralytics import YOLO

from qgeocompress.compression.low_rank import count_params
from qgeocompress.compression.structural_low_rank import StructuralLowRankConv2d, _set_module


def collapse_structural_conv(module: StructuralLowRankConv2d) -> nn.Conv2d:
    """Fold down+up factorization into a single equivalent Conv2d."""
    down = module.down.weight.data
    up = module.up.weight.data.squeeze(-1).squeeze(-1)
    rank, c_in, kh, kw = down.shape
    c_out = up.shape[0]

    down_flat = down.reshape(rank, -1)
    w2d = up @ down_flat
    w = w2d.reshape(c_out, c_in, kh, kw)

    conv = nn.Conv2d(
        c_in,
        c_out,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.up.bias is not None,
    )
    conv.weight.data.copy_(w)
    if module.up.bias is not None and conv.bias is not None:
        conv.bias.data.copy_(module.up.bias.data)
    return conv


def collapse_structural_model(model: YOLO) -> tuple[YOLO, dict[str, Any]]:
    """Replace all StructuralLowRankConv2d modules with equivalent Conv2d layers."""
    collapsed = copy.deepcopy(model)
    pytorch_model = collapsed.model
    original_params = count_params(pytorch_model)
    replaced = 0
    layer_names: list[str] = []

    for name, module in list(pytorch_model.named_modules()):
        if not isinstance(module, StructuralLowRankConv2d):
            continue
        _set_module(pytorch_model, name, collapse_structural_conv(module))
        replaced += 1
        layer_names.append(name)

    collapsed_params = count_params(pytorch_model)
    meta: dict[str, Any] = {
        "collapsed_for_ultralytics_compat": True,
        "num_collapsed_layers": replaced,
        "collapsed_layer_names": layer_names[:20],
        "original_params": original_params,
        "collapsed_params": collapsed_params,
        "param_delta_pct": round(
            100.0 * (collapsed_params - original_params) / original_params if original_params else 0.0,
            2,
        ),
        "note": "Structural layers folded to standard Conv2d; param count may increase vs compressed checkpoint",
    }
    return collapsed, meta
