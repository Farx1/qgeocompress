"""Experimental rectangle moments for distillation, not an IoU certificate."""

import torch


def rectangle_moments(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return covariance entries (xx, yy, sqrt(2)xy) and complex fourth moment.

    Boxes have shape (..., 5), ordered cx, cy, width, height, angle (radians).
    Moments refer to the uniform distribution inside the centered rectangle.
    """
    if boxes.shape[-1] != 5:
        raise ValueError("expected (..., 5) boxes")
    width, height, angle = boxes[..., 2:].unbind(-1)
    u, v = width.square() / 12, height.square() / 12
    cos, sin = angle.cos(), angle.sin()
    covariance = torch.stack(
        (u * cos.square() + v * sin.square(), u * sin.square() + v * cos.square(), 2**0.5 * (u - v) * sin * cos), -1
    )
    kappa = (3 * (width.pow(4) + height.pow(4)) - 10 * width.square() * height.square()) / 240
    fourth = torch.stack((kappa * (4 * angle).cos(), kappa * (4 * angle).sin()), -1)
    return covariance, fourth


def geometry_error(student: torch.Tensor, teacher: torch.Tensor, fourth: bool = False) -> torch.Tensor:
    """Dimensionless per-box error using frozen teacher scales.

    Caller supplies finite decoded boxes of positive size. The floor stabilizes
    subpixel boxes; this is a smooth surrogate, not polygon IoU or Wasserstein distance.
    """
    cov_s, four_s = rectangle_moments(student)
    cov_t, four_t = rectangle_moments(teacher.detach())
    scale = (cov_t[..., 0] + cov_t[..., 1]).clamp_min(1.0)
    center = (student[..., :2] - teacher.detach()[..., :2]).square().sum(-1) / scale
    loss = center + (cov_s - cov_t).square().sum(-1) / scale.square()
    if fourth:
        loss = loss + (four_s - four_t).square().sum(-1) / scale.pow(4)
    return loss
