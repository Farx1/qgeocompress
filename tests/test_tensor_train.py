"""Checks for the tensor-train (MPS) factorization of conv kernels."""

from __future__ import annotations

import math

import pytest
import torch

from qgeocompress.compression.tensor_train import (
    approximate,
    balanced_factors,
    tensorize,
    tt_decompose,
    tt_reconstruct,
)

LAYOUTS = ("chw", "split", "interleaved")


def test_balanced_factors_are_balanced_and_exact():
    for n in (64, 128, 256, 96):
        a, b = balanced_factors(n)
        assert a * b == n
        assert a <= b


@pytest.mark.parametrize("layout", LAYOUTS)
def test_untruncated_train_reproduces_the_kernel(layout):
    torch.manual_seed(0)
    weight = torch.randn(32, 16, 3, 3)
    approx, params, entropies = approximate(weight, layout, max_rank=10**6)
    assert torch.allclose(approx, weight, atol=1e-5)
    assert params >= weight.numel()  # an exact train cannot be cheaper than dense
    assert len(entropies) == len(tensorize(weight, layout)[1]) - 1


@pytest.mark.parametrize("layout", LAYOUTS)
def test_truncation_costs_fewer_parameters_and_more_error(layout):
    torch.manual_seed(0)
    weight = torch.randn(32, 16, 3, 3)
    tight, tight_params, _ = approximate(weight, layout, max_rank=2)
    loose, loose_params, _ = approximate(weight, layout, max_rank=16)
    assert tight_params < loose_params
    assert (tight - weight).norm() >= (loose - weight).norm()


def test_entropy_is_bounded_by_the_bond_dimension():
    """The von Neumann entropy of a Schmidt spectrum cannot exceed log of its size.

    The measurement that killed this direction relies on the bound: trained
    YOLO11n kernels sit at S/S_max = 0.93 across the channel cuts, so a matrix
    product state has no small bond to exploit.
    """
    torch.manual_seed(0)
    tensor, shape = tensorize(torch.randn(16, 8, 3, 3), "interleaved")
    _, entropies = tt_decompose(tensor, 10**6)

    left = 1
    for bond, entropy in enumerate(entropies):
        left *= shape[bond]
        right = math.prod(shape[bond + 1 :])
        assert 0.0 <= entropy <= math.log(min(left, right)) + 1e-9


def test_reconstruct_inverts_decompose_on_a_plain_tensor():
    torch.manual_seed(0)
    tensor = torch.randn(4, 5, 6)
    cores, _ = tt_decompose(tensor, 10**6)
    assert torch.allclose(tt_reconstruct(cores), tensor.to(torch.float64), atol=1e-8)
