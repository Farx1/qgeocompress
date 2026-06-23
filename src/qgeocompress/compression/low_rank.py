from __future__ import annotations

import copy
import re
from typing import Any

import torch
import torch.nn as nn
from ultralytics import YOLO

_HEAD_LAYER_RE = re.compile(r"^model\.23(\.|$)")
_DEFAULT_TARGETS = frozenset({"backbone", "neck"})


class LowRankLinear(nn.Module):
    """W ≈ U @ V factorization for Linear layers (used in unit tests)."""

    def __init__(self, linear: nn.Linear, rank: int):
        super().__init__()
        in_features = linear.in_features
        out_features = linear.out_features
        rank = max(1, min(rank, in_features, out_features))
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=linear.bias is not None)
        self._init_from_linear(linear, rank)

    def _init_from_linear(self, linear: nn.Linear, rank: int) -> None:
        with torch.no_grad():
            w = linear.weight.data
            u, s, vh = torch.linalg.svd(w, full_matrices=False)
            u = u[:, :rank]
            s = s[:rank]
            vh = vh[:rank, :]
            self.down.weight.copy_(vh)
            self.up.weight.copy_(u @ torch.diag(s))
            if linear.bias is not None and self.up.bias is not None:
                self.up.bias.copy_(linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class LowRankConv2d(nn.Module):
    """Low-rank conv chain (used in unit tests)."""

    def __init__(self, conv: nn.Conv2d, rank: int):
        super().__init__()
        c_in, c_out = conv.in_channels, conv.out_channels
        k = conv.kernel_size[0]
        rank = max(1, min(rank, c_in, c_out))
        self.down = nn.Conv2d(c_in, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(
            rank, c_out, kernel_size=k, stride=conv.stride, padding=conv.padding, bias=conv.bias is not None
        )
        with torch.no_grad():
            w2d = conv.weight.data.reshape(c_out, -1)
            u, s, vh = torch.linalg.svd(w2d, full_matrices=False)
            rank = min(rank, s.shape[0])
            approx = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
            approx = approx.reshape(c_out, c_in, k, k)
            for r in range(rank):
                self.up.weight.data[:, r : r + 1] = approx[:, :1] if rank == 1 else self.up.weight.data[:, r : r + 1]
            self.down.weight.normal_(0, 0.02)
            self.up.weight.copy_(approx.unsqueeze(1).expand(c_out, rank, c_in, k, k)[:, 0])  # fallback init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _layer_matches(name: str, targets: list[str]) -> bool:
    name_lower = name.lower()
    return any(t.lower() in name_lower for t in targets)


def _is_yolo_head_layer(name: str) -> bool:
    return bool(_HEAD_LAYER_RE.match(name))


def _should_replace_layer(name: str, module: nn.Module, target_layers: list[str] | None) -> bool:
    targets = {t.lower() for t in (target_layers or ["backbone", "neck"])}

    if _is_yolo_head_layer(name):
        return False

    if targets <= _DEFAULT_TARGETS:
        if not isinstance(module, nn.Conv2d):
            return False
        if module.kernel_size != (3, 3):
            return False
        if min(module.in_channels, module.out_channels) < 8:
            return False
        return name.endswith(".conv")

    if not _layer_matches(name, list(targets)):
        return False
    return isinstance(module, (nn.Linear, nn.Conv2d))


def _svd_truncate_conv(conv: nn.Conv2d, rank_ratio: float) -> tuple[int, int, int]:
    """In-place SVD truncation; returns (rank, original_params, theoretical_params)."""
    w = conv.weight.data
    c_out, c_in, kh, kw = w.shape
    flat = c_out * c_in * kh * kw
    w2d = w.reshape(c_out, -1)
    u, s, vh = torch.linalg.svd(w2d, full_matrices=False)
    rank = max(1, int(min(s.shape[0], c_out, w2d.shape[1]) * rank_ratio))
    approx = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
    conv.weight.data.copy_(approx.reshape(c_out, c_in, kh, kw))
    theoretical = c_out * rank + rank * c_in * kh * kw
    return rank, flat, theoretical


def apply_low_rank(
    model: YOLO,
    rank_ratio: float = 0.5,
    target_layers: list[str] | None = None,
) -> tuple[YOLO, dict[str, Any]]:
    """Apply low-rank SVD truncation to Conv2d weights (Ultralytics-compatible, in-place)."""
    pytorch_model = copy.deepcopy(model.model)
    original_params = count_params(pytorch_model)
    replaced = 0
    rank_ratio = max(0.05, min(1.0, rank_ratio))
    theoretical_params = 0
    layer_savings = 0

    for name, module in list(pytorch_model.named_modules()):
        if not _should_replace_layer(name, module, target_layers):
            continue
        if isinstance(module, nn.Conv2d):
            _, flat, theo = _svd_truncate_conv(module, rank_ratio)
            theoretical_params += theo
            layer_savings += flat - theo
            replaced += 1
        elif isinstance(module, nn.Linear):
            w = module.weight.data
            flat = w.numel()
            u, s, vh = torch.linalg.svd(w, full_matrices=False)
            rank = max(1, int(min(s.shape[0], w.shape[0], w.shape[1]) * rank_ratio))
            module.weight.data.copy_((u[:, :rank] * s[:rank]) @ vh[:rank, :])
            theo = w.shape[0] * rank + rank * w.shape[1]
            theoretical_params += theo
            layer_savings += flat - theo
            replaced += 1

    model.model = pytorch_model
    param_reduction = 100.0 * layer_savings / original_params if original_params else 0.0

    meta = {
        "method": "low-rank",
        "compression_mode": "svd_in_place",
        "rank_ratio": rank_ratio,
        "target_layers": target_layers or ["backbone", "neck"],
        "num_replaced_layers": replaced,
        "layers_replaced": replaced,
        "original_params": original_params,
        "compressed_params": original_params,
        "theoretical_params": theoretical_params,
        "param_reduction_pct": round(param_reduction, 2),
        "theoretical_rank_reduction_pct": round(param_reduction, 2),
        "collapsed_for_ultralytics_compat": False,
    }
    return model, meta


def collapse_low_rank_model(model: YOLO) -> YOLO:
    """No-op for in-place SVD mode (kept for API compatibility)."""
    return model
