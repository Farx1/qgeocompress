from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def config_path(*parts: str) -> Path:
    return project_root() / "configs" / Path(*parts)


def load_dataset_config(name: str) -> dict[str, Any]:
    return load_yaml(config_path("dataset", f"{name}.yaml"))


def load_model_config(name: str) -> dict[str, Any]:
    return load_yaml(config_path("model", f"{name}.yaml"))


def load_experiment_config(name: str) -> dict[str, Any]:
    return load_yaml(config_path("experiment", f"{name}.yaml"))


def make_run_id(prefix: str, config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    digest = hashlib.md5(payload.encode()).hexdigest()[:8]
    return f"{prefix}_{digest}"


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)
