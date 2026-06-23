from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO


def magnitude_prune(model: YOLO, sparsity: float = 0.2) -> tuple[YOLO, dict[str, Any]]:
    """Apply global magnitude pruning to Conv2d and Linear layers."""
    pytorch_model = copy.deepcopy(model.model)
    parameters_to_prune = []
    for module in pytorch_model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            parameters_to_prune.append((module, "weight"))

    if not parameters_to_prune:
        return model, {"method": "magnitude-pruning", "sparsity": 0.0, "layers": 0}

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )
    for module, name in parameters_to_prune:
        prune.remove(module, name)

    model.model = pytorch_model
    total_zeros = sum((p == 0).sum().item() for p in pytorch_model.parameters())
    total = sum(p.numel() for p in pytorch_model.parameters())
    actual_sparsity = total_zeros / max(total, 1)
    return model, {
        "method": "magnitude-pruning",
        "target_sparsity": sparsity,
        "actual_sparsity": round(actual_sparsity, 4),
        "layers": len(parameters_to_prune),
    }
