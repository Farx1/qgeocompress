"""Tensor-train (matrix product state) factorization of conv kernels.

A rank-r SVD cuts the weight tensor once, into two blocks. A tensor train cuts
it at every mode, giving a chain of cores linked by bond dimensions — the matrix
product state of quantum many-body physics, where the bond dimension needed at a
cut is governed by the entanglement entropy across it. SVD is the two-site case.

For ``W[c_out, c_in, kh, kw]`` the useful tensorizations split the channel
dimensions, since that is the structure a single matricization cannot see:

    (c_out, c_in, k*k)                 3 cores, closest to plain SVD
    (o1, o2, i1, i2, k*k)              5 cores, output modes then input modes
    (o1, i1, o2, i2, k*k)              5 cores, interleaved (classic TT-conv)

with ``c_out = o1*o2`` and ``c_in = i1*i2`` balanced.

The parameter count is the crux. A rank-r SVD costs ``r*(c_in*k^2 + c_out)``,
linear in r. A tensor train with uniform bond dimension r costs
``sum_k r_{k-1} * n_k * r_k``, which is **quadratic** in r on the interior
bonds. So a tensor train wins only where the required bonds are genuinely small,
and loses badly where they are not. That is an empirical question about the
weights, answered here by reconstruction rather than assumed.

Nothing in this module builds a runtime block: it answers whether the
factorization is worth having before anything is built around it.
"""

from __future__ import annotations

import math

import torch


def balanced_factors(n: int) -> tuple[int, int]:
    """Split ``n`` into the most balanced pair of factors."""
    best = 1
    for candidate in range(1, int(math.isqrt(n)) + 1):
        if n % candidate == 0:
            best = candidate
    return best, n // best


def tensorize(weight: torch.Tensor, layout: str) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Reshape a conv kernel into the tensor a train will be built over."""
    c_out, c_in, kh, kw = weight.shape
    spatial = kh * kw
    if layout == "chw":
        shape = (c_out, c_in, spatial)
        return weight.reshape(shape), shape

    o1, o2 = balanced_factors(c_out)
    i1, i2 = balanced_factors(c_in)
    grouped = weight.reshape(o1, o2, i1, i2, spatial)
    if layout == "split":
        shape = (o1, o2, i1, i2, spatial)
        return grouped, shape
    if layout == "interleaved":
        shape = (o1, i1, o2, i2, spatial)
        return grouped.permute(0, 2, 1, 3, 4).contiguous(), shape
    raise ValueError(f"unknown layout: {layout}")


def tt_decompose(
    tensor: torch.Tensor,
    max_rank: int,
) -> tuple[list[torch.Tensor], list[float]]:
    """TT-SVD (Oseledets). Returns the cores and the entanglement entropy per bond.

    The entropy at a bond is ``-sum p_i log p_i`` with ``p_i = sigma_i^2 / sum
    sigma^2``: the von Neumann entropy of the Schmidt spectrum across that cut.
    ``exp(entropy)`` is the effective bond dimension, which is what a physicist
    would use to pick a truncation instead of a hand-set rank.
    """
    shape = tensor.shape
    cores: list[torch.Tensor] = []
    entropies: list[float] = []

    remainder = tensor.to(torch.float64)
    left = 1
    for mode in range(len(shape) - 1):
        matrix = remainder.reshape(left * shape[mode], -1)
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)

        squared = s**2
        probabilities = squared / squared.sum().clamp(min=1e-30)
        nonzero = probabilities[probabilities > 1e-12]
        entropies.append(float(-(nonzero * nonzero.log()).sum()))

        rank = max(1, min(max_rank, s.numel()))
        cores.append(u[:, :rank].reshape(left, shape[mode], rank))
        remainder = (s[:rank, None] * vh[:rank]).reshape(rank, *shape[mode + 1 :])
        left = rank

    cores.append(remainder.reshape(left, shape[-1], 1))
    return cores, entropies


def tt_reconstruct(cores: list[torch.Tensor]) -> torch.Tensor:
    """Contract a tensor train back into the full tensor."""
    result = cores[0]
    for core in cores[1:]:
        left = result.reshape(-1, result.shape[-1])
        right = core.reshape(core.shape[0], -1)
        result = (left @ right).reshape(*result.shape[:-1], *core.shape[1:])
    return result.squeeze(0).squeeze(-1)


def tt_params(cores: list[torch.Tensor]) -> int:
    return int(sum(core.numel() for core in cores))


def entanglement_ranks(entropies: list[float], scale: float = 1.0) -> list[int]:
    """Effective bond dimensions ``exp(entropy)``, the parameter-free criterion."""
    return [max(1, int(round(scale * math.exp(entropy)))) for entropy in entropies]


def approximate(
    weight: torch.Tensor,
    layout: str,
    max_rank: int,
) -> tuple[torch.Tensor, int, list[float]]:
    """Return ``(reconstructed weight, tensor-train parameters, bond entropies)``."""
    tensor, shape = tensorize(weight, layout)
    cores, entropies = tt_decompose(tensor, max_rank)
    approx = tt_reconstruct(cores).reshape(shape)

    if layout == "interleaved":
        o1, i1, o2, i2, spatial = shape
        approx = approx.permute(0, 2, 1, 3, 4).contiguous()
    approx = approx.reshape(weight.shape)
    return approx.to(weight.dtype), tt_params(cores), entropies
