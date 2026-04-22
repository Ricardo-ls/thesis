from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def ensure_parent(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_npz(path, **arrays):
    path = ensure_parent(path)
    np.savez_compressed(path, **arrays)


def load_npz(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {key: _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_json(path, obj):
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=2, ensure_ascii=False, sort_keys=True)


def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
