from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

from qgeocompress.utils.config import project_root, save_json


def get_data_yaml(dataset_name: str) -> str:
    """Return Ultralytics data YAML path (built-in or local override)."""
    local = project_root() / "configs" / "dataset" / f"{dataset_name}.yaml"
    if local.exists():
        cfg = __import__("yaml").safe_load(local.read_text())
        return cfg.get("data_yaml", dataset_name + ".yaml")
    return f"{dataset_name}.yaml"


def resolve_data_yaml(dataset: str | None = None, data_yaml: str | Path | None = None) -> str:
    """Resolve Ultralytics data YAML from explicit path or dataset name."""
    if data_yaml is not None:
        path = Path(data_yaml)
        if not path.is_absolute():
            path = project_root() / path
        if not path.exists():
            raise FileNotFoundError(f"Data YAML not found: {path}")
        return str(path.resolve())
    if dataset is None:
        raise ValueError("Either dataset or data_yaml must be provided")
    yaml_path = get_data_yaml(dataset)
    path = Path(yaml_path)
    if not path.is_absolute() and not path.exists():
        candidate = project_root() / yaml_path
        if candidate.exists():
            return str(candidate.resolve())
    return yaml_path


def validate_dataset(dataset_name: str) -> dict[str, Any]:
    """Validate dataset via Ultralytics and return summary stats."""
    data_yaml = get_data_yaml(dataset_name)
    data_info = check_det_dataset(data_yaml)

    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "data_yaml": data_yaml,
        "nc": data_info.get("nc"),
        "names": data_info.get("names"),
        "train_images": _count_images(data_info.get("train")),
        "val_images": _count_images(data_info.get("val")),
        "test_images": _count_images(data_info.get("test")),
    }
    return summary


def _count_images(path: str | None) -> int:
    if not path:
        return 0
    p = Path(path)
    if p.is_file():
        with open(p) as f:
            return sum(1 for _ in f)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        return sum(1 for f in p.rglob("*") if f.suffix.lower() in exts)
    return 0


def prepare_dataset_report(dataset_name: str, output_dir: Path | None = None) -> dict[str, Any]:
    """Generate dataset validation report and optional class distribution figure."""
    output_dir = output_dir or project_root() / "results" / "summaries"
    figures_dir = project_root() / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = validate_dataset(dataset_name)

    # Quick sanity check: run a tiny predict to ensure OBB pipeline loads
    try:
        model = YOLO("yolo11n-obb.pt")
        data_yaml = get_data_yaml(dataset_name)
        data_info = check_det_dataset(data_yaml)
        val_path = data_info.get("val") or data_info.get("train")
        if val_path:
            sample = _first_image(val_path)
            if sample:
                model.predict(sample, imgsz=640, verbose=False)
                summary["sample_inference"] = "ok"
    except Exception as exc:
        summary["sample_inference"] = f"failed: {exc}"

    report_path = output_dir / f"dataset_report_{dataset_name}.json"
    save_json(summary, report_path)
    summary["report_path"] = str(report_path)

    _plot_class_distribution(summary, figures_dir / f"class_distribution_{dataset_name}.png")
    return summary


def _first_image(path: str) -> str | None:
    p = Path(path)
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    if p.is_file():
        with open(p) as f:
            line = f.readline().strip()
            return line if line else None
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in exts:
                return str(f)
    return None


def _plot_class_distribution(summary: dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    names = summary.get("names") or {}
    if not names:
        return
    labels = [names[k] if isinstance(names, dict) else n for k, n in enumerate(names)]
    counts = [1] * len(labels)  # placeholder until per-class stats are computed
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(labels)), counts, color="#2d6a9f")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Classes — {summary.get('dataset', 'dataset')}")
    ax.set_ylabel("present in dataset")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    summary["class_distribution_figure"] = str(out_path)
