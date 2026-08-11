#!/usr/bin/env python
"""Screen factorization schemes analytically, before spending any evaluation budget.

A rank-r factorization of a k x k conv can be built from three different
matricizations of the same kernel, each with its own parameter cost:

  spatial-first   W -> (c_out, c_in*kh*kw)      r*(c_in*kh*kw + c_out)
  channel-first   W -> (c_out*kh*kw, c_in)      r*(c_in + c_out*kh*kw)
  separable       W -> (c_in*kh, c_out*kw)      r*(c_in*kh + c_out*kw)

The project only ever used spatial-first. Which one wins is a per-layer
question: it depends on whether c_in*k^2 or c_out*k^2 is the larger term, and
on how fast that particular matricization's spectrum decays.

This script asks the decisive question directly — for a target relative output
error, how many parameters does each scheme need? — on the layers' real
activations, without running detection once.

Usage: python scripts/scheme_screen.py [--split-dir DIR] [--include-head]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

from qgeocompress.compression.layer_sensitivity import capture_layer_inputs
from qgeocompress.compression.tensor_train import approximate as tt_approximate
from qgeocompress.evaluation.gt_matching import list_val_images
from qgeocompress.utils.config import project_root, save_json

SCHEMES = ("spatial-first", "channel-first", "separable")
TT_LAYOUTS = ("tt-chw", "tt-split", "tt-interleaved")


def matricize(weight: torch.Tensor, scheme: str) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Return (2-D view, permutation) for the requested scheme."""
    c_out, c_in, kh, kw = weight.shape
    if scheme == "spatial-first":
        return weight.reshape(c_out, c_in * kh * kw), ()
    if scheme == "channel-first":
        # (c_out, kh, kw, c_in) -> rows carry the spatial extent
        return weight.permute(0, 2, 3, 1).reshape(c_out * kh * kw, c_in), (0, 2, 3, 1)
    if scheme == "separable":
        # (c_in, kh, c_out, kw): a rank-r fit is r vertical k x 1 filters
        # followed by r horizontal 1 x k filters.
        return weight.permute(1, 2, 0, 3).reshape(c_in * kh, c_out * kw), (1, 2, 0, 3)
    raise ValueError(scheme)


def unmatricize(flat: torch.Tensor, weight: torch.Tensor, scheme: str) -> torch.Tensor:
    c_out, c_in, kh, kw = weight.shape
    if scheme == "spatial-first":
        return flat.reshape(c_out, c_in, kh, kw)
    if scheme == "channel-first":
        return flat.reshape(c_out, kh, kw, c_in).permute(0, 3, 1, 2).contiguous()
    if scheme == "separable":
        return flat.reshape(c_in, kh, c_out, kw).permute(2, 0, 1, 3).contiguous()
    raise ValueError(scheme)


def scheme_params(conv: nn.Conv2d, scheme: str, rank: int) -> int:
    c_in, c_out = conv.in_channels, conv.out_channels
    kh, kw = conv.kernel_size
    bias = c_out if conv.bias is not None else 0
    if scheme == "spatial-first":
        return rank * (c_in * kh * kw + c_out) + bias
    if scheme == "channel-first":
        return rank * (c_in + c_out * kh * kw) + bias
    if scheme == "separable":
        return rank * (c_in * kh + c_out * kw) + bias
    raise ValueError(scheme)


def truncated_weight(weight: torch.Tensor, scheme: str, rank: int) -> torch.Tensor:
    flat, _ = matricize(weight.to(torch.float64), scheme)
    u, s, vh = torch.linalg.svd(flat, full_matrices=False)
    rank = max(1, min(rank, s.numel()))
    approx = (u[:, :rank] * s[:rank]) @ vh[:rank]
    return unmatricize(approx, weight, scheme).to(weight.dtype)


def output_error(conv: nn.Conv2d, approx_weight: torch.Tensor, inputs: list[torch.Tensor]) -> float:
    num = den = 0.0
    with torch.no_grad():
        for x in inputs:
            ref = F.conv2d(x, conv.weight, None, conv.stride, conv.padding, conv.dilation)
            got = F.conv2d(x, approx_weight, None, conv.stride, conv.padding, conv.dilation)
            num += float(((ref - got) ** 2).sum())
            den += float((ref**2).sum())
    return (num / den) ** 0.5 if den > 0 else 0.0


def candidates(conv: nn.Conv2d, method: str, probes: int = 12):
    """Yield ``(parameter cost, reconstructed weight)`` over a rank sweep.

    Matricization schemes cost ``r * (something)``, linear in the rank. A tensor
    train costs ``sum_k r_{k-1} * n_k * r_k``, quadratic in the bond dimension on
    interior bonds, so the two families are only comparable through the actual
    parameter count — which is why this yields cost alongside each candidate
    rather than a rank.
    """
    if method in SCHEMES:
        flat, _ = matricize(conv.weight.data, method)
        top = min(flat.shape)
        for i in range(1, probes + 1):
            rank = max(1, round(top * i / probes))
            yield scheme_params(conv, method, rank), truncated_weight(conv.weight.data, method, rank)
        return

    layout = method.removeprefix("tt-")
    top = max(conv.out_channels, conv.in_channels)
    for i in range(1, probes + 1):
        rank = max(1, round(top * i / probes))
        approx, cost, _ = tt_approximate(conv.weight.data, layout, rank)
        yield cost, approx


def params_for_error(
    conv: nn.Conv2d,
    method: str,
    inputs: list[torch.Tensor],
    target_error: float,
    probes: int = 12,
) -> int | None:
    """Cheapest parameter count reaching ``target_error``, or None if unreachable."""
    full = int(sum(p.numel() for p in conv.parameters()))
    best: int | None = None
    for cost, approx in candidates(conv, method, probes):
        if cost >= full or (best is not None and cost >= best):
            continue
        if output_error(conv, approx, inputs) <= target_error:
            best = cost
    return best


def eligible(model: nn.Module, include_head: bool) -> list[tuple[str, nn.Conv2d]]:
    out = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d) or module.groups != 1:
            continue
        if not include_head and name.startswith("model.23"):
            continue
        if module.kernel_size[0] not in (1, 3):
            continue
        if min(module.in_channels, module.out_channels) < 8:
            continue
        out.append((name, module))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", default="datasets/dota128_power")
    parser.add_argument("--weights", default="yolo11n-obb.pt")
    parser.add_argument("--images", type=int, default=3)
    parser.add_argument("--errors", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--include-head", action="store_true")
    args = parser.parse_args(argv)

    select_yaml = str(sorted(Path(args.split_dir).glob("*_select.yaml"))[0])
    model = YOLO(args.weights, task="obb")
    total = sum(p.numel() for p in model.model.parameters())

    convs = eligible(model.model, include_head=args.include_head)
    head_convs = [n for n, _ in eligible(model.model, include_head=True) if n.startswith("model.23")]
    print(f"total params {total}  candidates {len(convs)}  (head convs available: {len(head_convs)})")

    images = [str(p) for _, p in list_val_images(data_yaml=select_yaml)[: args.images]]
    captured = capture_layer_inputs(model.model, [n for n, _ in convs], images)

    rows = []
    for name, conv in convs:
        inputs = captured.pop(name)
        full = int(sum(p.numel() for p in conv.parameters()))
        entry = {"layer": name, "params": full, "kernel": conv.kernel_size[0],
                 "c_in": conv.in_channels, "c_out": conv.out_channels}
        for target in args.errors:
            for method in SCHEMES + TT_LAYOUTS:
                if conv.kernel_size[0] == 1 and method in ("channel-first", "separable"):
                    continue  # 1x1: those matricizations coincide with spatial-first
                entry[f"{method}@{target}"] = params_for_error(conv, method, inputs, target)
        rows.append(entry)

    print(f"\n{'target':>7} {'method':>16} {'model params kept':>18} {'reduction':>10}")
    summary = {}
    for target in args.errors:
        per_scheme = defaultdict(int)
        best_mix = 0
        for entry in rows:
            options = {
                m: entry.get(f"{m}@{target}")
                for m in SCHEMES + TT_LAYOUTS
                if entry.get(f"{m}@{target}") is not None
            }
            for method in SCHEMES + TT_LAYOUTS:
                per_scheme[method] += options.get(method, entry["params"])
            best_mix += min(options.values()) if options else entry["params"]

        uncompressed = total - sum(e["params"] for e in rows)
        for method in SCHEMES + TT_LAYOUTS:
            kept = uncompressed + per_scheme[method]
            print(f"{target:>7} {method:>16} {kept:>18} {100 * (1 - kept / total):>9.2f}%")
        kept = uncompressed + best_mix
        print(f"{target:>7} {'BEST-PER-LAYER':>16} {kept:>18} {100 * (1 - kept / total):>9.2f}%")
        summary[str(target)] = {
            **{
                m: round(100 * (1 - (uncompressed + per_scheme[m]) / total), 3)
                for m in SCHEMES + TT_LAYOUTS
            },
            "best_per_layer": round(100 * (1 - kept / total), 3),
        }

    out = project_root() / "results" / "summaries" / "scheme_screen.json"
    save_json({"weights": args.weights, "include_head": args.include_head,
               "reduction_pct": summary, "layers": rows}, out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
