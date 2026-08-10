"""Checks for the activation-weighted factorization math."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qgeocompress.compression.data_aware import (
    collect_cross_grams,
    collect_patch_gram,
    data_aware_factors,
    drift_corrected_weight,
    rank_for_energy,
)


def _apply(conv: nn.Conv2d, down: torch.Tensor, up: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    h = F.conv2d(x, down, None, conv.stride, conv.padding, conv.dilation)
    return F.conv2d(h, up, conv.bias)


def test_data_aware_beats_plain_svd_on_biased_inputs():
    """Inputs that only excite part of the input space must steer the factorization."""
    torch.manual_seed(0)
    conv = nn.Conv2d(16, 16, 3, padding=1, bias=False)
    # Half the input channels carry signal; plain SVD cannot know that.
    x = torch.randn(2, 16, 12, 12)
    x[:, 8:] *= 0.01
    gram = collect_patch_gram(conv, [x])

    ref = conv(x)
    plain = _apply(conv, *data_aware_factors(conv, 6, None), x)
    aware = _apply(conv, *data_aware_factors(conv, 6, gram), x)

    err_plain = (ref - plain).norm() / ref.norm()
    err_aware = (ref - aware).norm() / ref.norm()
    assert err_aware < err_plain


def test_data_aware_matches_plain_svd_when_gram_is_identity():
    torch.manual_seed(0)
    conv = nn.Conv2d(8, 8, 1, bias=False)
    identity = torch.eye(8, dtype=torch.float64)
    d0, u0 = data_aware_factors(conv, 4, None)
    d1, u1 = data_aware_factors(conv, 4, identity)
    w0 = (u0.reshape(8, 4) @ d0.reshape(4, 8))
    w1 = (u1.reshape(8, 4) @ d1.reshape(4, 8))
    assert torch.allclose(w0, w1, atol=1e-5)


def test_drift_correction_targets_the_original_output():
    """With a drifted input, the refit must reproduce the ORIGINAL layer output.

    Regression: the solve returned G'^-1 C^T instead of C G'^-1. Both equal the
    identity at zero drift, so only a genuinely drifted input separates them.
    """
    torch.manual_seed(0)
    conv = nn.Conv2d(6, 6, 1, bias=False)
    clean = torch.randn(1, 6, 10, 10)
    mix = torch.linalg.qr(torch.randn(6, 6))[0] * 0.9  # invertible, clearly not identity
    drifted = F.conv2d(clean, mix.reshape(6, 6, 1, 1))

    cross, gram = collect_cross_grams(conv, [clean], [drifted])
    target = conv(clean)
    naive = F.conv2d(drifted, conv.weight.data)
    assert (target - naive).norm() > 0.1 * target.norm()

    # Well-determined case: with a negligible ridge the refit is essentially exact.
    exact = F.conv2d(drifted, drift_corrected_weight(conv, cross, gram, ridge=1e-10))
    assert (target - exact).norm() < 1e-3 * target.norm()

    # The production ridge trades exactness for stability but must still beat naive.
    ridged = F.conv2d(drifted, drift_corrected_weight(conv, cross, gram))
    assert (target - ridged).norm() < 0.5 * (target - naive).norm()


def test_rank_for_energy_is_monotone():
    torch.manual_seed(0)
    conv = nn.Conv2d(16, 16, 3, padding=1, bias=False)
    ranks = [rank_for_energy(conv, e, None) for e in (0.5, 0.9, 0.99)]
    assert ranks == sorted(ranks)
    assert ranks[-1] <= 16
