"""Activation-weighted low-rank factorization and rank allocation.

Plain SVD truncation minimizes ``||W - W_hat||_F``. That is the wrong objective:
the network only cares about ``||(W - W_hat) X||_F``, where ``X`` holds the
im2col patches the layer actually sees. Directions of weight space that the data
never excites can be discarded for free, and directions the data hammers must be
kept even when their singular value is small.

With ``M = X X^T = L L^T`` (Cholesky),

    ||(W - W_hat) X||_F = ||(W - W_hat) L||_F

so the optimal rank-r factor is the plain truncated SVD of ``W L`` mapped back
through ``L^-1``:

    W L = U S V^T,  W_hat = U_r S_r V_r^T L^-1

which factors into ``down = S_r V_r^T L^-1`` (a k x k conv with r outputs) and
``up = U_r`` (a 1x1 conv), exactly the shape ``StructuralLowRankConv2d`` uses.

Setting ``L = I`` recovers plain SVD, so both paths share one code path.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def collect_patch_gram(
    conv: nn.Conv2d,
    layer_inputs: list[torch.Tensor],
    max_patches: int = 200_000,
    seed: int = 0,
) -> torch.Tensor:
    """Return ``X X^T`` over im2col patches of this conv's real inputs (d x d)."""
    d = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
    gram = torch.zeros(d, d, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)

    for x in layer_inputs:
        patches = F.unfold(
            x.to(torch.float32),
            kernel_size=conv.kernel_size,
            dilation=conv.dilation,
            padding=conv.padding,
            stride=conv.stride,
        )  # (batch, d, positions)
        patches = patches.permute(1, 0, 2).reshape(d, -1)
        if patches.shape[1] > max_patches:
            idx = torch.randperm(patches.shape[1], generator=generator)[:max_patches]
            patches = patches[:, idx]
        gram += (patches.to(torch.float64) @ patches.to(torch.float64).T)

    return gram


def _whitening_factor(gram: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    """Cholesky factor of a ridge-stabilised Gram matrix."""
    d = gram.shape[0]
    scale = float(torch.diagonal(gram).mean().clamp(min=1e-12))
    eye = torch.eye(d, dtype=gram.dtype)
    for attempt in range(6):
        try:
            return torch.linalg.cholesky(gram + (ridge * (10**attempt)) * scale * eye)
        except RuntimeError:
            continue
    # Fall back to plain SVD if the Gram matrix refuses to factor.
    return eye


def factor_spectrum(
    conv: nn.Conv2d,
    gram: torch.Tensor | None,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Singular values of ``W L`` (or of ``W`` when ``gram`` is None)."""
    src = conv.weight.data if weight is None else weight
    w2d = src.reshape(conv.out_channels, -1).to(torch.float64)
    if gram is None:
        return torch.linalg.svdvals(w2d)
    return torch.linalg.svdvals(w2d @ _whitening_factor(gram))


def bn_output_scale(model: nn.Module, conv_name: str) -> torch.Tensor | None:
    """``gamma / sqrt(running_var + eps)`` of the BatchNorm following this conv.

    Nothing reads a conv's output directly: in Ultralytics every conv is
    immediately scaled by its BatchNorm, so an error of the same size in two
    different output channels reaches the next layer scaled very differently.
    Weighting the rows by that factor puts the truncation in the metric the
    network actually sees.
    """
    parent_name = conv_name.rsplit(".", 1)[0] if "." in conv_name else ""
    try:
        parent = model.get_submodule(parent_name) if parent_name else model
    except AttributeError:
        return None
    bn = getattr(parent, "bn", None)
    if not isinstance(bn, nn.BatchNorm2d) or bn.weight is None:
        return None
    return (bn.weight.data / torch.sqrt(bn.running_var + bn.eps)).to(torch.float64)


def data_aware_factors(
    conv: nn.Conv2d,
    rank: int,
    gram: torch.Tensor | None,
    weight: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(down_weight, up_weight)`` for a rank-``r`` factorization.

    ``down_weight`` has shape ``(r, c_in, kh, kw)`` and ``up_weight``
    ``(c_out, r, 1, 1)``, matching ``StructuralLowRankConv2d``. ``weight``
    overrides ``conv.weight`` (used by the drift-corrected pass, which factors a
    regression solution rather than the original kernel).
    """
    source = conv.weight.data if weight is None else weight
    c_out, c_in, kh, kw = source.shape
    w2d = source.reshape(c_out, -1).to(torch.float64)
    rank = max(1, min(rank, min(w2d.shape)))

    # Two-sided metric: rows scaled by what BatchNorm will apply, columns
    # whitened by what the data excites. Undone on the way out.
    scale = None
    if output_scale is not None:
        scale = output_scale.reshape(-1, 1).clamp(min=1e-8)
        w2d = scale * w2d

    if gram is None:
        u, s, vh = torch.linalg.svd(w2d, full_matrices=False)
        down = vh[:rank]
        up = u[:, :rank] * s[:rank]
    else:
        chol = _whitening_factor(gram)
        u, s, vh = torch.linalg.svd(w2d @ chol, full_matrices=False)
        # down = (S_r V_r^T) L^-1. solve_triangular(A, B, left=False) gives B @ A^-1,
        # so pass the lower-triangular L itself, not its transpose.
        down = torch.linalg.solve_triangular(
            chol, (s[:rank, None] * vh[:rank]), upper=False, left=False
        )
        up = u[:, :rank]

    if scale is not None:
        up = up / scale

    return (
        down.reshape(rank, c_in, kh, kw).to(source.dtype),
        up.reshape(c_out, rank, 1, 1).to(source.dtype),
    )


def relative_error(conv: nn.Conv2d, rank: int, gram: torch.Tensor | None) -> float:
    """Relative reconstruction error in the metric the factorization optimizes."""
    spectrum = factor_spectrum(conv, gram)
    total = float((spectrum**2).sum())
    if total <= 0:
        return 0.0
    kept = float((spectrum[:rank] ** 2).sum())
    return float(max(0.0, 1.0 - kept / total) ** 0.5)


def rank_for_energy(conv: nn.Conv2d, energy: float, gram: torch.Tensor | None) -> int:
    """Smallest rank keeping ``energy`` of the squared spectrum."""
    spectrum = factor_spectrum(conv, gram)
    cumulative = torch.cumsum(spectrum**2, dim=0)
    total = float(cumulative[-1])
    if total <= 0:
        return 1
    idx = int(torch.searchsorted(cumulative, torch.tensor(energy * total)).item())
    return max(1, min(idx + 1, spectrum.numel()))


def param_count(conv: nn.Conv2d, rank: int) -> int:
    c_in, c_out = conv.in_channels, conv.out_channels
    kh, kw = conv.kernel_size
    n = c_in * rank * kh * kw + rank * c_out
    if conv.bias is not None:
        n += c_out
    return n


def allocate_ranks_by_budget(
    layers: dict[str, dict[str, Any]],
    target_saved_params: int,
    lambda_steps: int = 240,
    importance: dict[str, float] | None = None,
) -> dict[str, int]:
    """Pick per-layer ranks minimizing total error under a parameter budget.

    The problem is separable across layers and each ``error(rank)`` curve is
    decreasing, so a Lagrangian sweep solves it: for a price ``lambda`` per saved
    parameter, every layer independently takes the rank minimizing
    ``error + lambda * params``. Sweeping ``lambda`` traces the whole frontier;
    keep the cheapest allocation that still meets the budget. The sweep must be
    fine: at 60 steps, 12% and 16% targets landed on the same allocation.

    ``importance`` scales each layer's error term. Spectral tail energy treats
    all layers as equally valuable, which they are not: the same relative error
    in an early backbone conv propagates through everything downstream. Pass
    measured end-to-end sensitivities to weight them.

    ``layers`` maps name -> {"conv": Conv2d, "gram": Tensor | None}.
    """
    curves: dict[str, list[tuple[int, int, float]]] = {}
    for name, entry in layers.items():
        conv, gram = entry["conv"], entry.get("gram")
        spectrum = factor_spectrum(conv, gram)
        squared = spectrum**2
        total = float(squared.sum()) or 1.0
        tail = torch.flip(torch.cumsum(torch.flip(squared, [0]), dim=0), [0])
        full = int(sum(p.numel() for p in conv.parameters()))
        ranks = range(1, spectrum.numel() + 1)
        weight = (importance or {}).get(name, 1.0)
        curves[name] = [
            (
                r,
                full - param_count(conv, r),
                weight * (float(tail[r].item() / total) if r < spectrum.numel() else 0.0),
            )
            for r in ranks
        ]

    best: dict[str, int] | None = None
    best_saved = -1
    # error is normalised to [0, 1]; lambda spans "free" to "parameters are worthless".
    for step in range(lambda_steps + 1):
        lam = 10 ** (-8 + 10 * step / lambda_steps)
        allocation: dict[str, int] = {}
        saved_total = 0
        for name, curve in curves.items():
            rank, saved, _ = min(curve, key=lambda row: row[2] - lam * row[1])
            allocation[name] = rank
            saved_total += saved
        if saved_total >= target_saved_params and (best is None or saved_total < best_saved):
            best, best_saved = allocation, saved_total

    if best is None:  # budget unreachable — take the most aggressive allocation
        best = {name: 1 for name in curves}
    return best


def collect_cross_grams(
    conv: nn.Conv2d,
    clean_inputs: list[torch.Tensor],
    drifted_inputs: list[torch.Tensor],
    max_patches: int = 60_000,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(X X'^T, X' X'^T)`` for clean vs already-compressed inputs."""
    d = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
    cross = torch.zeros(d, d, dtype=torch.float64)
    gram = torch.zeros(d, d, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)

    def unfold(x: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(
            x.to(torch.float32),
            kernel_size=conv.kernel_size,
            dilation=conv.dilation,
            padding=conv.padding,
            stride=conv.stride,
        )
        return patches.permute(1, 0, 2).reshape(d, -1)

    for clean, drifted in zip(clean_inputs, drifted_inputs, strict=True):
        a, b = unfold(clean), unfold(drifted)
        if a.shape[1] > max_patches:
            idx = torch.randperm(a.shape[1], generator=generator)[:max_patches]
            a, b = a[:, idx], b[:, idx]
        a64, b64 = a.to(torch.float64), b.to(torch.float64)
        cross += a64 @ b64.T
        gram += b64 @ b64.T

    return cross, gram


def drift_corrected_weight(
    conv: nn.Conv2d,
    cross: torch.Tensor,
    gram: torch.Tensor,
    ridge: float = 1e-2,
) -> torch.Tensor:
    """Least-squares weight mapping drifted inputs onto the ORIGINAL outputs.

    Factorizing layer by layer moves every downstream layer's input off the
    distribution its kernel was trained on, and that drift compounds. Matching
    the drifted input to the drifted output (what plain data-aware SVD does)
    locks the error in; the right target is the output the *uncompressed* layer
    produced.

    Minimizing ``||W X - W' X'||_F`` over ``W'`` gives
    ``W' = W (X X'^T) (X' X'^T)^-1``, which is then factored to rank r under the
    ``X'`` metric. If the drift is zero this returns ``W`` exactly.
    """
    w2d = conv.weight.data.reshape(conv.out_channels, -1).to(torch.float64)
    # The regression needs a real ridge, not just enough jitter to make Cholesky
    # succeed: a deep layer at stride 32 sees ~400 patch positions per image
    # against d up to 2304, so G' is rank-deficient and its inverse amplifies
    # sampling noise straight into the refitted kernel.
    d = gram.shape[0]
    scale = float(torch.diagonal(gram).mean().clamp(min=1e-12))
    chol = torch.linalg.cholesky(gram + ridge * scale * torch.eye(d, dtype=gram.dtype))
    # cholesky_solve(B, L) returns (L L^T)^-1 B, so this is G'^-1 C^T; the factor
    # we want is C G'^-1, its transpose (G' is symmetric). Both collapse to the
    # identity when there is no drift, so only a drifted case tells them apart.
    rhs = torch.cholesky_solve(cross.T, chol).T
    return (w2d @ rhs).reshape(conv.weight.shape).to(conv.weight.dtype)


def layer_importance(
    reference_output_error: dict[str, float],
    floor: float = 1e-3,
) -> dict[str, float]:
    """Normalize measured end-to-end sensitivities into allocation weights."""
    if not reference_output_error:
        return {}
    scale = max(reference_output_error.values()) or 1.0
    return {name: max(floor, value / scale) for name, value in reference_output_error.items()}


def transfer_importance(
    reference_output_error: dict[str, float],
    layers: dict[str, dict[str, Any]],
    probe_ratio: float = 0.5,
    floor: float = 1e-3,
) -> dict[str, float]:
    """Network error produced per unit of a layer's OWN reconstruction error.

    The allocation prices layers in units of relative spectral tail energy, so
    the weight it needs is a transfer coefficient, not a raw sensitivity. A layer
    measured as "insensitive" may simply have had very little tail energy to
    discard at the probe rank; dividing by that error recovers how much the
    network actually moves per unit of damage, which is what the Lagrangian
    trade-off is comparing against saved parameters.

    Measured against raw sensitivity at a -31% budget: mAP50 0.767 vs 0.719.
    """
    ratios: dict[str, float] = {}
    for name, entry in layers.items():
        conv, gram = entry["conv"], entry.get("gram")
        rank = max(1, int(min(conv.out_channels, conv.in_channels * conv.kernel_size[0] ** 2) * probe_ratio))
        own_error = max(1e-6, relative_error(conv, rank, gram))
        ratios[name] = reference_output_error.get(name, 0.0) / own_error
    return layer_importance(ratios, floor=floor)


def measured_marginals(
    model: nn.Module,
    reference_outputs: list[torch.Tensor],
    probe_inputs: list[torch.Tensor],
    bump: dict[str, tuple[nn.Module, int]],
    forward_output: Any,
) -> dict[str, float]:
    """Error removed per extra parameter, measured on the COMPOSED model.

    The Lagrangian prices layers with their error measured in isolation, but a
    layer's real cost depends on everything else that was compressed: the
    surrogate and the objective disagree, and that disagreement is what caps the
    frontier once 61 layers are factorized together.

    Here each layer is bumped one rank step up *in the assembled model* and the
    drop in end-to-end output error is measured. That is the true discrete
    gradient of the objective the allocation is supposed to be minimizing.

    ``bump`` maps layer name -> (replacement block, extra parameters).
    """
    def total_error() -> float:
        num = den = 0.0
        with torch.no_grad():
            for x, ref in zip(probe_inputs, reference_outputs, strict=True):
                out = forward_output(model, x)
                num += float(((out - ref) ** 2).sum())
                den += float((ref**2).sum())
        return (num / den) ** 0.5 if den else 0.0

    base_error = total_error()
    marginals: dict[str, float] = {}

    for name, (replacement, extra_params) in bump.items():
        if extra_params <= 0:
            continue
        parent = model
        *path, leaf = name.split(".")
        for part in path:
            parent = getattr(parent, part)
        original = getattr(parent, leaf)
        setattr(parent, leaf, replacement)
        marginals[name] = max(0.0, base_error - total_error()) / extra_params
        setattr(parent, leaf, original)

    return marginals


def calibrate_importance(
    importance: dict[str, float],
    measured: dict[str, float],
    predicted: dict[str, float],
    floor: float = 1e-3,
    clip: float = 10.0,
) -> dict[str, float]:
    """Rescale each layer's weight by how wrong its surrogate marginal was.

    A layer whose measured marginal exceeds what the curve predicted is being
    under-protected, and vice versa. Multiplying the weight by the ratio makes
    the next allocation agree with the composed model where it currently does
    not, without discarding the curve's shape.
    """
    corrected: dict[str, float] = {}
    for name, weight in importance.items():
        ratio = measured.get(name, 0.0) / max(predicted.get(name, 0.0), 1e-12)
        corrected[name] = weight * min(max(ratio, 1.0 / clip), clip)

    scale = max(corrected.values()) if corrected else 1.0
    return {name: max(floor, value / (scale or 1.0)) for name, value in corrected.items()}
