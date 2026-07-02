from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np


CORRUPTIONS = ("cloud", "blur", "noise", "jpeg", "brightness", "contrast", "resolution")


def apply_corruption(
    image: np.ndarray,
    corruption: str,
    severity: int = 1,
) -> np.ndarray:
    """Apply a synthetic corruption to an image (labels unchanged)."""
    severity = max(1, min(5, severity))
    img = image.copy()

    if corruption == "blur":
        k = 2 * severity + 1
        return cv2.GaussianBlur(img, (k, k), 0)

    if corruption == "noise":
        sigma = severity * 5
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return out

    if corruption == "jpeg":
        quality = max(10, 95 - severity * 15)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc = cv2.imencode(".jpg", img, encode_param)
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    if corruption == "brightness":
        factor = 1.0 + (severity - 3) * 0.15
        out = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return out

    if corruption == "contrast":
        factor = 1.0 + (severity - 3) * 0.2
        mean = img.mean()
        out = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
        return out

    if corruption == "resolution":
        h, w = img.shape[:2]
        scale = 1.0 - severity * 0.08
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    if corruption == "cloud":
        return _cloud_overlay(img, severity)

    raise ValueError(f"Unknown corruption: {corruption}. Choose from {CORRUPTIONS}")


def _cloud_overlay(image: np.ndarray, severity: int) -> np.ndarray:
    h, w = image.shape[:2]
    overlay = image.copy().astype(np.float32)
    n_patches = severity * 3
    for _ in range(n_patches):
        cx, cy = random.randint(0, w), random.randint(0, h)
        rx, ry = random.randint(w // 10, w // 4), random.randint(h // 10, h // 4)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (rx, ry), random.randint(0, 180), 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        opacity = 0.3 + severity * 0.08
        overlay = overlay * (1 - mask[..., None] * opacity) + 255 * mask[..., None] * opacity
    return np.clip(overlay, 0, 255).astype(np.uint8)


def corruption_grid() -> list[dict[str, Any]]:
    return [{"name": c, "severities": [1, 2, 3, 4, 5]} for c in CORRUPTIONS]
