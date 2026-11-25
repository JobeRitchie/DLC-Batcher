"""Persistent app settings and calibration profiles (~/.dlc_batcher/settings.json)."""
from __future__ import annotations

import json
import os
from pathlib import Path


def settings_dir() -> Path:
    root = os.environ.get("DLC_BATCHER_HOME")
    path = Path(root) if root else Path.home() / ".dlc_batcher"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _file() -> Path:
    return settings_dir() / "settings.json"


def load() -> dict:
    try:
        with open(_file(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save(data: dict) -> None:
    tmp = _file().with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, _file())
    except OSError:
        pass


def profile_key(gpu_name: str, net_type: str, width: int, height: int) -> str:
    """Calibration results depend on the GPU, the network and the frame size."""
    return f"{gpu_name or 'cpu'} | {net_type} | {width}x{height}"


def get_profile(data: dict, key: str) -> dict | None:
    return (data.get("profiles") or {}).get(key)


def put_profile(data: dict, key: str, profile: dict) -> None:
    data.setdefault("profiles", {})[key] = profile
