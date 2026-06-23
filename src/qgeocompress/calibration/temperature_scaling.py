from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS


class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaling for calibration."""

    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temperature]))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.05)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 50,
) -> float:
    """Fit temperature on validation logits (binary/multi-class flattened)."""
    x = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    scaler = TemperatureScaler()
    criterion = nn.BCEWithLogitsLoss()

    optimizer = LBFGS(scaler.parameters(), lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(x), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(scaler.temperature.item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return logits / max(temperature, 1e-3)
