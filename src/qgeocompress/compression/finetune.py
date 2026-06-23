from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from qgeocompress.data.prepare_dota import get_data_yaml
from qgeocompress.utils.config import project_root


@dataclass
class FinetuneConfig:
    lr0: float = 1e-5
    lrf: float = 0.1
    freeze: int | None = None
    use_last: bool = False
    optimizer: str = "AdamW"


# Absolute mAP50 drop or relative drop that aborts fine-tuning (reload instability).
RELOAD_DROP_ABS_THRESHOLD = 0.10
RELOAD_DROP_REL_THRESHOLD = 0.15

FINETUNE_BACKEND = "ultralytics_train_api"
UNSUPPORTED_TRAIN_API_REASON = (
    "Excluded: Ultralytics train() does not preserve compressed low-rank state; "
    "observed transfer 106/541 items. Use custom training loop or structural low-rank (Phase 3)."
)


def compute_reload_drop(map_before: float | None, map_after_reload: float | None) -> float | None:
    if map_before is None or map_after_reload is None:
        return None
    return round(max(0.0, map_before - map_after_reload), 6)


def reload_is_unstable(
    map_before: float | None,
    map_after_reload: float | None,
    abs_threshold: float = RELOAD_DROP_ABS_THRESHOLD,
    rel_threshold: float = RELOAD_DROP_REL_THRESHOLD,
) -> bool:
    """True when reloading the compressed checkpoint destroys validation mAP."""
    if map_before is None or map_after_reload is None:
        return False
    drop = map_before - map_after_reload
    if drop > abs_threshold:
        return True
    return map_before > 1e-6 and (drop / map_before) > rel_threshold


def compute_map50_recovery(
    map_baseline: float | None,
    map_before: float | None,
    map_after: float | None,
) -> float | None:
    """Fraction of mAP50 gap recovered after fine-tuning (0–100 %)."""
    if map_baseline is None or map_before is None or map_after is None:
        return None
    denom = map_baseline - map_before
    if denom <= 1e-9:
        return 100.0 if map_after >= map_baseline else 0.0
    recovery = (map_after - map_before) / denom
    return round(max(0.0, min(100.0, recovery * 100.0)), 2)


def load_baseline_map50(dataset: str = "dota128") -> float | None:
    """Load baseline mAP50 from the latest baseline summary JSON, if available."""
    summaries = project_root() / "results" / "summaries"
    if not summaries.exists():
        return None

    candidates: list[tuple[str, dict[str, Any]]] = []
    for path in summaries.glob("*.json"):
        if path.name in {"final_report.json"} or path.name.startswith("dataset_report"):
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("compression") in ("baseline", "none") and data.get("dataset") == dataset:
            if data.get("map50") is not None:
                candidates.append((path.name, data))

    if not candidates:
        return None
    _, latest = sorted(candidates, key=lambda x: x[0])[-1]
    return float(latest["map50"])


def finetune_compressed_model(
    model: YOLO,
    dataset: str,
    epochs: int,
    imgsz: int,
    device: str,
    output_dir: Path,
    seed: int = 42,
    config: FinetuneConfig | None = None,
) -> dict[str, Path]:
    """Fine-tune the in-memory compressed YOLO model (no checkpoint reload)."""
    config = config or FinetuneConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = get_data_yaml(dataset)

    train_kwargs: dict[str, Any] = {
        "data": data_yaml,
        "task": "obb",
        "epochs": epochs,
        "imgsz": imgsz,
        "device": device,
        "seed": seed,
        "project": str(output_dir.parent),
        "name": output_dir.name,
        "exist_ok": True,
        "verbose": False,
        "optimizer": config.optimizer,
        "lr0": config.lr0,
        "lrf": config.lrf,
        "warmup_epochs": 1,
        "cos_lr": True,
        "patience": 0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
    }
    if config.freeze is not None:
        train_kwargs["freeze"] = config.freeze

    train_results = model.train(**train_kwargs)
    save_dir = Path(train_results.save_dir)
    weights_dir = save_dir / "weights"
    return {
        "save_dir": save_dir,
        "best": weights_dir / "best.pt",
        "last": weights_dir / "last.pt",
    }


def select_finetune_checkpoint(
    checkpoints: dict[str, Path],
    best_metrics: dict[str, Any] | None,
    last_metrics: dict[str, Any] | None,
    use_last: bool = False,
) -> tuple[Path | None, str | None]:
    """Pick best or last checkpoint based on mAP50 (or forced last)."""
    best_path = checkpoints.get("best")
    last_path = checkpoints.get("last")
    best_exists = best_path is not None and best_path.exists()
    last_exists = last_path is not None and last_path.exists()

    if use_last and last_exists:
        return last_path, "last.pt"
    if not best_exists and not last_exists:
        return None, None
    if not last_exists:
        return best_path, "best.pt"
    if not best_exists:
        return last_path, "last.pt"

    best_map = (best_metrics or {}).get("map50")
    last_map = (last_metrics or {}).get("map50")
    if best_map is None and last_map is None:
        return best_path, "best.pt"
    if last_map is not None and (best_map is None or last_map >= best_map):
        return last_path, "last.pt"
    return best_path, "best.pt"
