"""Checks for the three factorization schemes and the least-squares refit."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from qgeocompress.compression.factorized import (
    SCHEMES,
    FactorizedConv2d,
    allocate_schemes_by_budget,
    max_rank,
    refit_second_factor,
    scheme_params,
)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_full_rank_reproduces_the_conv(scheme):
    """Whatever the matricization, the untruncated factorization is exact."""
    torch.manual_seed(0)
    conv = nn.Conv2d(12, 16, 3, stride=2, padding=1, bias=True)
    x = torch.randn(1, 12, 20, 20)
    block = FactorizedConv2d(conv, scheme, max_rank(conv, scheme))
    out, ref = block(x), conv(x)
    assert out.shape == ref.shape
    assert (out - ref).norm() < 1e-4 * ref.norm()


@pytest.mark.parametrize("scheme", SCHEMES)
def test_scheme_params_matches_the_built_block(scheme):
    conv = nn.Conv2d(16, 24, 3, padding=1, bias=False)
    block = FactorizedConv2d(conv, scheme, 5)
    assert sum(p.numel() for p in block.parameters()) == scheme_params(conv, scheme, 5)


def test_separable_is_the_cheapest_scheme_per_rank():
    """r*k*(c_in + c_out) beats both r*(c_in*k^2 + c_out) and r*(c_in + c_out*k^2)."""
    conv = nn.Conv2d(64, 64, 3, padding=1, bias=False)
    costs = {s: scheme_params(conv, s, 8) for s in SCHEMES}
    assert costs["separable"] < costs["spatial-first"]
    assert costs["separable"] < costs["channel-first"]


def test_refit_lowers_output_error_on_biased_inputs():
    """Truncation minimizes ||W - W'||_F; the refit minimizes what matters."""
    torch.manual_seed(0)
    conv = nn.Conv2d(16, 16, 3, padding=1, bias=False)
    x = torch.randn(2, 16, 14, 14)
    x[:, 8:] *= 0.01  # half the input space is barely excited

    ref = conv(x)
    block = FactorizedConv2d(conv, "separable", 6)
    before = ((block(x) - ref).norm() / ref.norm()).item()
    refit_second_factor(block, conv, [x])
    after = ((block(x) - ref).norm() / ref.norm()).item()

    assert after < before


def test_budget_allocation_meets_the_target_and_mixes_schemes():
    convs = {
        "a": nn.Conv2d(64, 128, 3, padding=1, bias=False),
        "b": nn.Conv2d(128, 64, 3, padding=1, bias=False),
        "c": nn.Conv2d(64, 64, 1, bias=False),
    }
    full = sum(sum(p.numel() for p in c.parameters()) for c in convs.values())
    allocation = allocate_schemes_by_budget(convs, target_saved_params=int(full * 0.4))

    saved = sum(
        sum(p.numel() for p in convs[n].parameters()) - scheme_params(convs[n], s, r)
        for n, (s, r) in allocation.items()
    )
    assert saved >= full * 0.4
    # A 1x1 conv has only one distinct matricization.
    assert allocation["c"][0] == "spatial-first"


def test_every_block_type_triggers_the_no_fuse_guard():
    """A block missing from the guard crashes Ultralytics' Conv+BN fusion."""
    from qgeocompress.compression.structural_low_rank import (
        StructuralLowRankConv2d,
        has_factorized_layers,
    )

    conv = nn.Conv2d(8, 8, 3, padding=1, bias=False)
    assert not has_factorized_layers(nn.Sequential(conv))
    assert has_factorized_layers(nn.Sequential(StructuralLowRankConv2d(conv, 4)))
    assert has_factorized_layers(nn.Sequential(FactorizedConv2d(conv, "separable", 4)))


def test_curves_always_offer_leaving_the_layer_alone():
    """Without a no-op option the allocation must damage every layer.

    Regression: forcing a factorization on layers that only save parameters at
    very low rank (the detection head) collapsed mAP50 from 0.74 to 0.34.
    """
    from qgeocompress.compression.factorized import (
        allocate_from_curves,
        measure_scheme_curves,
    )

    convs = {"tiny": nn.Conv2d(16, 16, 1, bias=False)}
    curves = measure_scheme_curves(convs, {"tiny": [torch.randn(1, 16, 8, 8)]}, rank_probes=4)
    assert any(option[0] == "none" for option in curves["tiny"])

    # A budget of zero must leave everything untouched.
    assert allocate_from_curves(curves, target_saved_params=0)["tiny"][0] == "none"
