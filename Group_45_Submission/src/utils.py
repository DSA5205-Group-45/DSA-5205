"""Shared helpers for hashing, JSON output and package versions."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    """Deterministic hash of a JSON-serialisable payload."""
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(dict(payload)), indent=2, default=str) + "\n")


def package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {
        "python": platform.python_version(),
    }
    for name in ("numpy", "pandas", "pyarrow", "scipy", "sklearn", "xgboost", "matplotlib", "yaml", "joblib"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # pragma: no cover - optional dependency
            versions[name] = "not_installed"
    return versions
