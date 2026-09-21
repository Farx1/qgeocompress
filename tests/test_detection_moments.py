import math

import cv2
import numpy as np
import torch

from qgeocompress.compression.detection_moments import geometry_error, rectangle_moments


def test_equivalent_rectangle_parameterizations():
    boxes = torch.tensor([[0, 0, 12, 4, 0.3], [2, 5, 7, 7, -0.2]], dtype=torch.float64)
    swapped = boxes.clone()
    swapped[:, 2], swapped[:, 3] = boxes[:, 3], boxes[:, 2]
    swapped[:, 4] += math.pi / 2
    for a, b in zip(rectangle_moments(boxes), rectangle_moments(swapped), strict=True):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(
        geometry_error(boxes, swapped, fourth=True), torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0
    )


def test_fourth_moment_detects_rotation_hidden_from_covariance():
    teacher = torch.tensor([[0, 0, 12, 12, 0.0]], dtype=torch.float64)
    student = teacher.clone()
    student[:, 4] = math.pi / 4
    assert geometry_error(student, teacher).item() < 1e-25
    assert geometry_error(student, teacher, fourth=True).item() > 1


def test_square_rotation_gradient_and_zero_matching_loss():
    teacher = torch.tensor([[0, 0, 12, 12, 0.0]], dtype=torch.float64)
    student = torch.tensor([[0, 0, 12, 12, math.pi / 8]], dtype=torch.float64, requires_grad=True)
    loss = geometry_error(student, teacher, fourth=True).sum()
    grad = torch.autograd.grad(loss, student)[0]
    assert torch.isfinite(grad).all() and abs(grad[0, 4]) > 0.1
    assert geometry_error(teacher, teacher, fourth=True).item() == 0


def test_moments_agree_with_numerical_integration():
    # Midpoint quadrature is independent of the closed-form implementation.
    box = torch.tensor([[0, 0, 8, 3, 0.43]], dtype=torch.float64)
    grid = (torch.arange(300, dtype=torch.float64) + 0.5) / 300 - 0.5
    x, y = torch.meshgrid(grid * 8, grid * 3, indexing="ij")
    z = torch.complex(x, y) * torch.exp(torch.tensor(0.43j, dtype=torch.complex128))
    expected = (z**4).mean()
    _, actual = rectangle_moments(box)
    torch.testing.assert_close(actual[0], torch.stack((expected.real, expected.imag)), rtol=1e-4, atol=1e-5)


def test_square_polygon_counterexample_and_stationary_maximum():
    a = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)
    rotation = np.array([[1, -1], [1, 1]], dtype=np.float32) / math.sqrt(2)
    b = (a @ rotation.T).astype(np.float32)
    intersection = cv2.intersectConvexConvex(a, b)[0]
    assert abs(intersection / (8 - intersection) - 1 / math.sqrt(2)) < 1e-6
    teacher = torch.tensor([[0, 0, 12, 12, 0.0]], dtype=torch.float64)
    student = torch.tensor([[0, 0, 12, 12, math.pi / 4]], dtype=torch.float64, requires_grad=True)
    loss = geometry_error(student, teacher, fourth=True).sum()
    assert abs(float(loss.detach()) - 36 / 25) < 1e-12
    assert abs(torch.autograd.grad(loss, student)[0][0, 4]) < 1e-12


def test_gradcheck_and_spatial_rescaling():
    student = torch.tensor([[3, 4, 12, 5, 0.2]], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([[2, 5, 11, 6, -0.1]], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x: geometry_error(x, teacher, fourth=True), (student,))
    scaling = torch.tensor([2, 2, 2, 2, 1], dtype=torch.float64)
    torch.testing.assert_close(geometry_error(student, teacher, fourth=True),
                               geometry_error(student * scaling, teacher * scaling, fourth=True))
