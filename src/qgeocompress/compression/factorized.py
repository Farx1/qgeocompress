"""Three factorization schemes for a conv, and a data-aware refit for all of them.

A rank-r factorization of ``W[c_out, c_in, kh, kw]`` comes from choosing which
matricization to truncate. Each choice has a different parameter cost and a
different spectrum, so the cheapest one is a per-layer question:

  scheme          matricization              parameters
  spatial-first   (c_out, c_in*kh*kw)        r*(c_in*kh*kw + c_out)
  channel-first   (c_out*kh*kw, c_in)        r*(c_in + c_out*kh*kw)
  separable       (c_in*kh, c_out*kw)        r*(c_in*kh + c_out*kw)

Measured on real activations at a 0.20 relative output error per layer, using
one scheme everywhere reaches 19.9% (spatial-first), 24.3% (channel-first) or
26.8% (separable) parameter reduction; picking the best scheme per layer
reaches 34.5%.

``data_aware.py`` whitens the input before truncating, which is exact for
spatial-first only — for the other two the input metric does not factor through
the matricization. :func:`refit_second_factor` recovers the same benefit for all
three: freeze the first factor and solve the second by weighted least squares
against the original layer's output on real activations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SCHEMES = ("spatial-first", "channel-first", "separable")


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
    raise ValueError(f"unknown scheme: {scheme}")


def max_rank(conv: nn.Conv2d, scheme: str) -> int:
    c_in, c_out = conv.in_channels, conv.out_channels
    kh, kw = conv.kernel_size
    if scheme == "spatial-first":
        return min(c_out, c_in * kh * kw)
    if scheme == "channel-first":
        return min(c_out * kh * kw, c_in)
    if scheme == "separable":
        return min(c_in * kh, c_out * kw)
    raise ValueError(f"unknown scheme: {scheme}")


def _matricize(weight: torch.Tensor, scheme: str) -> torch.Tensor:
    c_out, c_in, kh, kw = weight.shape
    if scheme == "spatial-first":
        return weight.reshape(c_out, c_in * kh * kw)
    if scheme == "channel-first":
        return weight.permute(0, 2, 3, 1).reshape(c_out * kh * kw, c_in)
    if scheme == "separable":
        return weight.permute(1, 2, 0, 3).reshape(c_in * kh, c_out * kw)
    raise ValueError(f"unknown scheme: {scheme}")


def scheme_spectrum(conv: nn.Conv2d, scheme: str) -> torch.Tensor:
    return torch.linalg.svdvals(_matricize(conv.weight.data.to(torch.float64), scheme))


def scheme_factors(
    conv: nn.Conv2d, scheme: str, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(down_weight, up_weight)`` for the requested scheme."""
    weight = conv.weight.data
    c_out, c_in, kh, kw = weight.shape
    flat = _matricize(weight.to(torch.float64), scheme)
    rank = max(1, min(rank, min(flat.shape)))
    u, s, vh = torch.linalg.svd(flat, full_matrices=False)
    u, s, vh = u[:, :rank], s[:rank], vh[:rank]

    if scheme == "spatial-first":
        down = vh.reshape(rank, c_in, kh, kw)
        up = (u * s).reshape(c_out, rank, 1, 1)
    elif scheme == "channel-first":
        down = vh.reshape(rank, c_in, 1, 1)
        up = (u * s).reshape(c_out, kh, kw, rank).permute(0, 3, 1, 2)
    else:  # separable
        down = u.reshape(c_in, kh, rank).permute(2, 0, 1).unsqueeze(-1)
        up = (s[:, None] * vh).reshape(rank, c_out, kw).permute(1, 0, 2).unsqueeze(2)

    return down.contiguous().to(weight.dtype), up.contiguous().to(weight.dtype)


class FactorizedConv2d(nn.Module):
    """A conv replaced by two convs, in any of the three schemes."""

    def __init__(self, conv: nn.Conv2d, scheme: str, rank: int):
        super().__init__()
        c_in, c_out = conv.in_channels, conv.out_channels
        kh, kw = conv.kernel_size
        sh, sw = conv.stride
        ph, pw = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
        rank = max(1, min(rank, max_rank(conv, scheme)))

        self.scheme = scheme
        self.rank = rank
        self.in_channels = c_in
        self.out_channels = c_out

        if scheme == "spatial-first":
            first = nn.Conv2d(c_in, rank, (kh, kw), (sh, sw), (ph, pw), conv.dilation, bias=False)
            second = nn.Conv2d(rank, c_out, 1, bias=conv.bias is not None)
        elif scheme == "channel-first":
            first = nn.Conv2d(c_in, rank, 1, bias=False)
            second = nn.Conv2d(
                rank, c_out, (kh, kw), (sh, sw), (ph, pw), conv.dilation, bias=conv.bias is not None
            )
        elif scheme == "separable":
            # Splitting stride and padding across the two axes preserves output size.
            first = nn.Conv2d(c_in, rank, (kh, 1), (sh, 1), (ph, 0), bias=False)
            second = nn.Conv2d(rank, c_out, (1, kw), (1, sw), (0, pw), bias=conv.bias is not None)
        else:
            raise ValueError(f"unknown scheme: {scheme}")

        self.down, self.up = first, second
        down_w, up_w = scheme_factors(conv, scheme, rank)
        with torch.no_grad():
            self.down.weight.copy_(down_w)
            self.up.weight.copy_(up_w)
            if conv.bias is not None and self.up.bias is not None:
                self.up.bias.copy_(conv.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


def refit_second_factor(
    block: FactorizedConv2d,
    conv: nn.Conv2d,
    inputs: list[torch.Tensor],
    ridge: float = 1e-3,
    max_patches: int = 60_000,
    seed: int = 0,
) -> None:
    """Solve the second factor by least squares against the original layer's output.

    Truncating a matricization minimizes ``||W - W_hat||_F``, which is not the
    objective that matters. Freezing the first factor makes the second one a
    linear problem in the metric that does: with ``H`` the im2col patches of the
    first factor's output and ``Y`` the original layer's output,

        B = Y H^T (H H^T + lambda I)^-1

    This is exact for whatever scheme the block uses, so all three become
    data-aware without needing a whitening that factors through their
    matricization.
    """
    up = block.up
    rows = up.out_channels
    cols = up.in_channels * up.kernel_size[0] * up.kernel_size[1]
    gram = torch.zeros(cols, cols, dtype=torch.float64)
    cross = torch.zeros(rows, cols, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        for x in inputs:
            target = F.conv2d(x, conv.weight, None, conv.stride, conv.padding, conv.dilation)
            hidden = block.down(x)
            patches = F.unfold(
                hidden,
                kernel_size=up.kernel_size,
                dilation=up.dilation,
                padding=up.padding,
                stride=up.stride,
            )
            patches = patches.permute(1, 0, 2).reshape(cols, -1)
            flat_target = target.permute(1, 0, 2, 3).reshape(rows, -1)
            if patches.shape[1] != flat_target.shape[1]:
                return  # shape mismatch (unexpected stride/padding split): leave as is
            if patches.shape[1] > max_patches:
                idx = torch.randperm(patches.shape[1], generator=generator)[:max_patches]
                patches, flat_target = patches[:, idx], flat_target[:, idx]
            p64, t64 = patches.to(torch.float64), flat_target.to(torch.float64)
            gram += p64 @ p64.T
            cross += t64 @ p64.T

    scale = float(torch.diagonal(gram).mean().clamp(min=1e-12))
    chol = torch.linalg.cholesky(gram + ridge * scale * torch.eye(cols, dtype=gram.dtype))
    solution = torch.cholesky_solve(cross.T, chol).T

    with torch.no_grad():
        # The target excluded the bias, so the fitted weight reproduces W*x only;
        # up.bias still carries the original conv's bias from __init__.
        up.weight.copy_(solution.reshape(up.weight.shape).to(up.weight.dtype))


def allocate_schemes_by_budget(
    layers: dict[str, nn.Conv2d],
    target_saved_params: int,
    importance: dict[str, float] | None = None,
    lambda_steps: int = 240,
    rank_probes: int = 24,
) -> dict[str, tuple[str, int]]:
    """Pick a (scheme, rank) per layer minimizing weighted error under a budget.

    Same separable Lagrangian as the single-scheme allocation, with the choice
    set widened from ranks to (scheme, rank) pairs. For a price ``lambda`` per
    saved parameter each layer independently takes the cheapest option, and
    sweeping ``lambda`` traces the frontier.
    """
    curves: dict[str, list[tuple[str, int, int, float]]] = {}
    for name, conv in layers.items():
        full = int(sum(p.numel() for p in conv.parameters()))
        weight = (importance or {}).get(name, 1.0)
        options: list[tuple[str, int, int, float]] = []
        for scheme in SCHEMES:
            if conv.kernel_size[0] == 1 and scheme != "spatial-first":
                continue  # the three matricizations coincide for 1x1
            spectrum = scheme_spectrum(conv, scheme)
            squared = spectrum**2
            total = float(squared.sum()) or 1.0
            tail = torch.flip(torch.cumsum(torch.flip(squared, [0]), dim=0), [0])
            top = max_rank(conv, scheme)
            for i in range(1, rank_probes + 1):
                rank = max(1, round(top * i / rank_probes))
                saved = full - scheme_params(conv, scheme, rank)
                if saved <= 0:
                    continue
                error = float(tail[rank].item() / total) if rank < spectrum.numel() else 0.0
                options.append((scheme, rank, saved, weight * error))
        if options:
            curves[name] = options

    best: dict[str, tuple[str, int]] | None = None
    best_saved = -1
    for step in range(lambda_steps + 1):
        lam = 10 ** (-8 + 10 * step / lambda_steps)
        allocation: dict[str, tuple[str, int]] = {}
        saved_total = 0
        for name, options in curves.items():
            scheme, rank, saved, _ = min(options, key=lambda row: row[3] - lam * row[2])
            allocation[name] = (scheme, rank)
            saved_total += saved
        if saved_total >= target_saved_params and (best is None or saved_total < best_saved):
            best, best_saved = allocation, saved_total

    if best is None:  # budget unreachable — take the most aggressive option per layer
        best = {
            name: max(options, key=lambda row: row[2])[:2] for name, options in curves.items()
        }
    return best


def measure_scheme_curves(
    layers: dict[str, nn.Conv2d],
    inputs_by_layer: dict[str, list[torch.Tensor]],
    rank_probes: int = 8,
) -> dict[str, list[tuple[str, int, int, float]]]:
    """Per-(layer, scheme, rank) curves of (saved params, MEASURED output error).

    The spectral tail energy of a matricization is not comparable across
    schemes: each truncates a different matrix, so a 1% tail means a different
    thing in each. Feeding those numbers to one Lagrangian silently prefers
    whichever scheme happens to have the flattest spectrum, and that cost 0.07
    mAP50 against the single-scheme allocation at an equal budget. Measuring the
    real relative output error on captured activations puts all schemes in the
    same units.
    """
    curves: dict[str, list[tuple[str, int, int, float]]] = {}
    for name, conv in layers.items():
        inputs = inputs_by_layer.get(name) or []
        if not inputs:
            continue
        full = int(sum(p.numel() for p in conv.parameters()))
        with torch.no_grad():
            reference = [
                F.conv2d(x, conv.weight, None, conv.stride, conv.padding, conv.dilation)
                for x in inputs
            ]
        denom = sum(float((r**2).sum()) for r in reference)
        options: list[tuple[str, int, int, float]] = []

        for scheme in SCHEMES:
            if conv.kernel_size[0] == 1 and scheme != "spatial-first":
                continue
            top = max_rank(conv, scheme)
            for i in range(1, rank_probes + 1):
                rank = max(1, round(top * i / rank_probes))
                saved = full - scheme_params(conv, scheme, rank)
                if saved <= 0:
                    continue
                block = FactorizedConv2d(conv, scheme, rank)
                num = 0.0
                with torch.no_grad():
                    for x, ref in zip(inputs, reference, strict=True):
                        num += float(((block(x) - ref) ** 2).sum())
                options.append((scheme, rank, saved, (num / denom) ** 0.5 if denom else 0.0))

        # Leaving a layer alone must be on the menu. Without it the allocation
        # is forced to factorize every layer, and layers that only ever save
        # parameters at very low rank — the detection head especially — get
        # destroyed no matter how the budget is priced.
        options.append(("none", 0, 0, 0.0))
        curves[name] = options
    return curves


def importance_from_curves(
    curves: dict[str, list[tuple[str, int, int, float]]],
    sensitivity: dict[str, float],
    reference_ratio: float = 0.5,
    floor: float = 1e-3,
) -> dict[str, float]:
    """Network error per unit of MEASURED layer error, in the curves' own units.

    ``data_aware.transfer_importance`` divides by a whitened spectral error,
    which is a different scale from the measured output error these curves use.
    Mixing the two re-introduces exactly the units mismatch the transfer
    coefficient exists to remove.
    """
    ratios: dict[str, float] = {}
    for name, options in curves.items():
        spatial = [o for o in options if o[0] == "spatial-first"]
        if not spatial:
            continue
        target = reference_ratio * max(o[1] for o in spatial)
        _, _, _, error = min(spatial, key=lambda o: abs(o[1] - target))
        ratios[name] = sensitivity.get(name, 0.0) / max(error, 1e-6)

    if not ratios:
        return {}
    scale = max(ratios.values()) or 1.0
    return {name: max(floor, value / scale) for name, value in ratios.items()}


def allocate_from_curves(
    curves: dict[str, list[tuple[str, int, int, float]]],
    target_saved_params: int,
    importance: dict[str, float] | None = None,
    lambda_steps: int = 240,
) -> dict[str, tuple[str, int]]:
    """Lagrangian allocation over measured (scheme, rank) curves."""
    best: dict[str, tuple[str, int]] | None = None
    best_saved = -1
    for step in range(lambda_steps + 1):
        lam = 10 ** (-8 + 10 * step / lambda_steps)
        allocation: dict[str, tuple[str, int]] = {}
        saved_total = 0
        for name, options in curves.items():
            weight = (importance or {}).get(name, 1.0)
            scheme, rank, saved, _ = min(
                options, key=lambda row: weight * row[3] - lam * row[2]
            )
            allocation[name] = (scheme, rank)
            saved_total += saved
        if saved_total >= target_saved_params and (best is None or saved_total < best_saved):
            best, best_saved = allocation, saved_total

    if best is None:
        best = {n: max(o, key=lambda row: row[2])[:2] for n, o in curves.items()}
    return best
