"""End-to-end checks against a real YOLO11n-OBB checkpoint and real DOTA128 images.

Everything else in the suite mocks Ultralytics. This file does not: it is the
only proof that the probe → select → compress → validate chain works on an
actual model. Slow (~1 min on CPU) and needs network on first run to fetch
``yolo11n-obb.pt``, so it is deselected by default — run with ``pytest -m slow``.
"""

from __future__ import annotations

import pytest
import torch.nn as nn
import yaml

from qgeocompress.compression.layer_sensitivity import (
    capture_layer_inputs,
    measure_local_output_error,
)
from qgeocompress.compression.structural_low_rank import (
    apply_structural_low_rank,
    list_structural_candidates,
)
from qgeocompress.evaluation.obb_validate import validate_no_fuse
from qgeocompress.utils.config import project_root

pytestmark = pytest.mark.slow

DOTA_NAMES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool",
]


@pytest.fixture(scope="module")
def model():
    from ultralytics import YOLO

    return YOLO("yolo11n-obb.pt", task="obb")


@pytest.fixture(scope="module")
def data_yaml(tmp_path_factory) -> str:
    root = project_root() / "datasets" / "dota128"
    assert (root / "images" / "train").is_dir(), "DOTA128 must be bundled in the repo"
    path = tmp_path_factory.mktemp("data") / "dota128.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/train",
                "names": dict(enumerate(DOTA_NAMES)),
            }
        )
    )
    return str(path)


def test_probe_measures_real_activation_error(model):
    """Regression: local_output_error was always null because raw images were fed
    into intermediate convs instead of their own activations."""
    images = sorted((project_root() / "datasets" / "dota128" / "images" / "train").glob("*.jpg"))[:2]
    names = [name for name, _ in list_structural_candidates(model.model, ["backbone"])][:3]
    assert names, "expected structural candidates in the backbone"

    captured = capture_layer_inputs(model.model, names, [str(p) for p in images])

    for name in names:
        conv = model.model.get_submodule(name)
        assert captured[name], f"no activation captured for {name}"
        assert captured[name][0].shape[1] == conv.in_channels
        err = measure_local_output_error(conv, 0.84, captured[name])
        assert err is not None and 0.0 < err < 1.0, f"{name}: implausible error {err}"


def test_structural_compression_validates_on_real_data(model, data_yaml):
    """Compressed model must lose real parameters and still run a real val pass."""
    import copy

    candidates = [name for name, _ in list_structural_candidates(model.model, ["backbone", "neck"])]
    compressed, meta = apply_structural_low_rank(
        copy.deepcopy(model),
        rank_ratio=0.84,
        selected_layer_names=candidates[:3],
    )

    assert meta["num_replaced_layers"] == 3
    assert meta["compressed_params"] < meta["original_params"]
    assert meta["real_param_reduction_pct"] > 0
    assert any(m.__class__.__name__ == "StructuralLowRankConv2d" for m in compressed.model.modules())

    results = validate_no_fuse(compressed, data_yaml, imgsz=640, device="cpu")
    assert 0.0 <= float(results.box.map50) <= 1.0


def test_no_fuse_restores_ultralytics_loader(model, data_yaml):
    """The no-fuse patch must not leak into later Ultralytics calls."""
    from ultralytics.nn.backends.pytorch import PyTorchBackend

    original = PyTorchBackend.load_model
    validate_no_fuse(model, data_yaml, imgsz=640, device="cpu")
    assert PyTorchBackend.load_model is original


def test_bn_recalibration_updates_running_stats(model, data_yaml):
    """BN recalibration must actually move BatchNorm statistics."""
    import copy

    from qgeocompress.compression.layer_sensitivity import recalibrate_batchnorm

    candidate = copy.deepcopy(model)
    bn = next(m for m in candidate.model.modules() if isinstance(m, nn.BatchNorm2d))
    before = bn.running_mean.clone()

    ran = recalibrate_batchnorm(candidate, data_yaml=data_yaml, num_batches=4, device="cpu")

    assert ran == 4
    assert not bn.running_mean.equal(before)
