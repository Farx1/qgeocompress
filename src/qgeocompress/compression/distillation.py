"""Label-free recovery: refit the factorized layers against the uncompressed model.

Every step before this one is closed form and therefore limited to what a
per-layer least-squares fit can express. The residual error is a joint property
of all the factorized layers at once, and only a joint optimization removes it.

The teacher is the original checkpoint, so no labels are involved: the student
is asked to reproduce the teacher's own predictions. That matters on a 24-image
training split, where fine-tuning against ground truth destroys the model
(measured: mAP50 0.927 -> 0.469) while matching a teacher cannot invent a
supervision signal that is not already in the weights.

Only the ``down``/``up`` factors are trainable. BatchNorm stays in eval mode:
its statistics are recalibrated separately and letting them drift here fights
that step.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from ultralytics import YOLO

from qgeocompress.compression.structural_low_rank import StructuralLowRankConv2d


def _head_inputs(model: YOLO, x: torch.Tensor) -> list[torch.Tensor]:
    """Feature maps entering the detection head.

    Distilling the head's raw output does not work: that tensor concatenates box
    coordinates in pixel units with class scores in [0, 1], so a plain MSE is
    dominated by the coordinates and ignores what actually drives AP (measured:
    loss fell 5x while mAP50 dropped 0.04). The head's input feature maps are
    post-BN activations of comparable scale across levels, which is what FitNet
    style feature distillation relies on.
    """
    captured: list[list[torch.Tensor]] = []
    head = model.model.model[-1]
    handle = head.register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
    try:
        model.model(x)
    finally:
        handle.remove()
    features = captured[0]
    return list(features) if isinstance(features, (list, tuple)) else [features]


def _feature_loss(student: list[torch.Tensor], teacher: list[torch.Tensor]) -> torch.Tensor:
    """Per-level MSE normalised by the teacher's scale, so no level dominates."""
    terms = [
        torch.nn.functional.mse_loss(s, t) / (t.detach().pow(2).mean() + 1e-8)
        for s, t in zip(student, teacher, strict=True)
    ]
    return sum(terms) / len(terms)


def factor_parameters(model: YOLO) -> list[nn.Parameter]:
    """Trainable parameters of the low-rank blocks only."""
    params: list[nn.Parameter] = []
    for module in model.model.modules():
        if isinstance(module, StructuralLowRankConv2d):
            params.extend(module.parameters())
    return params


def distill_recover(
    student: YOLO,
    teacher: YOLO,
    images: list[torch.Tensor],
    steps: int = 300,
    lr: float = 1e-4,
    batch_size: int = 2,
    seed: int = 0,
    log_every: int = 0,
) -> dict[str, Any]:
    """Optimize the factor weights so the student's outputs match the teacher's."""
    params = factor_parameters(student)
    if not params or not images:
        return {"steps": 0, "loss_start": None, "loss_end": None, "trainable_params": 0}

    teacher.model.eval()
    for p in teacher.model.parameters():
        p.requires_grad_(False)

    student.model.eval()  # keep BatchNorm running stats frozen
    for p in student.model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)

    with torch.no_grad():
        targets = [[t.detach() for t in _head_inputs(teacher, x)] for x in images]

    optimizer = torch.optim.Adam(params, lr=lr)
    generator = torch.Generator().manual_seed(seed)
    loss_start = loss_end = None

    for step in range(steps):
        idx = torch.randint(len(images), (batch_size,), generator=generator).tolist()
        optimizer.zero_grad(set_to_none=True)
        loss = sum(
            _feature_loss(_head_inputs(student, images[i]), targets[i]) for i in idx
        ) / batch_size
        loss.backward()
        optimizer.step()

        value = float(loss.detach())
        loss_start = value if loss_start is None else loss_start
        loss_end = value
        if log_every and step % log_every == 0:
            print(f"    distill step {step:4d}  loss {value:.6f}")

    for p in params:
        p.requires_grad_(False)

    return {
        "steps": steps,
        "lr": lr,
        "loss_start": loss_start,
        "loss_end": loss_end,
        "trainable_params": int(sum(p.numel() for p in params)),
    }
