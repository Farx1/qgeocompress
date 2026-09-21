import runpy
from pathlib import Path

import pytest
import torch
from torch import nn

from qgeocompress.compression.data_aware import data_aware_factors
from qgeocompress.compression.margin_weighting import margin_priority, normalized_priorities


def test_near_threshold_has_more_priority_and_clipping_is_finite():
    assert margin_priority(torch.tensor([0.25])) == pytest.approx(400)
    assert margin_priority(torch.tensor([0.26])) > margin_priority(torch.tensor([0.9]))
    assert normalized_priorities([1.0, 3.0]).tolist() == [0.5, 1.5]


@pytest.mark.parametrize("scores", [[], [float("nan")], [float("inf")], [-0.1], [1.1]])
def test_reject_invalid_scores(scores):
    with pytest.raises(ValueError):
        margin_priority(torch.tensor(scores))


@pytest.mark.parametrize("values", [[], [0.0], [-1.0], [float("nan")]])
def test_reject_invalid_priorities(values):
    with pytest.raises(ValueError):
        normalized_priorities(values)


def test_affine_flip_bound_and_gram_identity():
    generator = torch.Generator().manual_seed(71)
    w, delta = [torch.randn(3, 4, generator=generator, dtype=torch.float64) for _ in range(2)]
    x = torch.randn(40, 4, generator=generator, dtype=torch.float64)
    a = torch.randn(40, 3, generator=generator, dtype=torch.float64)
    margins = (a * (x @ w.T)).sum(1) + 0.37
    shifts = (a * (x @ delta.T)).sum(1)
    flips = (margins > 0) != (margins + shifts > 0)
    exact = shifts.square() / margins.square()
    alpha = a.square().sum(1) / margins.square()
    upper = alpha * (x @ delta.T).square().sum(1)
    assert torch.all(flips.double() <= exact + 1e-12)
    assert torch.all(exact <= upper + 1e-12)
    gram = x.T @ (alpha[:, None] * x) / len(x)
    torch.testing.assert_close(torch.trace(delta @ gram @ delta.T), upper.mean())


def test_weighted_svd_attains_tail_energy():
    generator = torch.Generator().manual_seed(5)
    conv = nn.Conv2d(4, 3, 1, bias=False).double()
    with torch.no_grad():
        conv.weight.copy_(torch.randn(conv.weight.shape, generator=generator, dtype=torch.float64))
    x = torch.randn(4, 20, generator=generator, dtype=torch.float64)
    gram = x @ x.T
    down, up = data_aware_factors(conv, 1, gram)
    regularized = gram + 1e-6 * gram.diag().mean() * torch.eye(4, dtype=torch.float64)
    factor = torch.linalg.cholesky(regularized)
    w = conv.weight.flatten(1)
    delta = up.flatten(1) @ down.flatten(1) - w
    attained = (delta @ factor).square().sum()
    optimum = torch.linalg.svdvals(w @ factor)[1:].square().sum()
    torch.testing.assert_close(attained, optimum)


def test_counterexamples_to_overclaiming():
    # Smaller squared normalized error need not mean fewer threshold flips.
    margins = torch.ones(2)
    shift_a, shift_b = torch.tensor([-1.1, 0.0]), torch.tensor([0.9, 0.9])
    assert shift_a.square().sum() < shift_b.square().sum()
    assert ((margins + shift_a) < 0).sum() > ((margins + shift_b) < 0).sum()
    # A zero first derivative does not certify a finite nonlinear perturbation.
    w = torch.tensor(0.0, requires_grad=True)
    margin = 1 - w.square()
    assert torch.autograd.grad(margin, w)[0] == 0
    assert 1 - (w.detach() + 2).square() < 0


def test_scene_split_is_disjoint_and_deterministic_with_duplicate_tiles():
    study = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/margin_weighting_study.py"))
    paths = [Path(f"P{i:04}__{tile}.jpg") for i in range(60) for tile in [0, 1]]
    cal, evaluation = study["split_scenes"](paths, 0)
    assert len(cal) == 16 and len(evaluation) == 32
    assert {p.stem.split("__")[0] for p in cal}.isdisjoint({p.stem.split("__")[0] for p in evaluation})
    assert (cal, evaluation) == study["split_scenes"](list(reversed(paths)), 0)
    assert all(p.stem.endswith("__0") for p in cal + evaluation)
