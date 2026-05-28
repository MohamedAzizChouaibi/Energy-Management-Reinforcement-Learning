"""Cache helpers for TomTom route segment JSON files."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_cache_dir() -> Path:
    return project_root() / "gps" / "cache"


def cache_key(origin: Any, destination: Any) -> str:
    """Return a stable cache key for an origin/destination pair."""
    raw = json.dumps(
        {"origin": origin, "destination": destination},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def route_cache_path(origin: Any, destination: Any, cache_dir: str | os.PathLike | None = None) -> Path:
    base = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return base / f"{cache_key(origin, destination)}.json"


def save_tomtom_cache(payload: dict, path: str | os.PathLike) -> Path:
    """Persist a route cache payload as formatted JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return out


def load_tomtom_cache(path: str | os.PathLike) -> dict:
    """Load a v3.1 route cache JSON payload."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{p} is not a route cache object")
    if "segments" not in payload:
        raise ValueError(f"{p} does not contain a 'segments' array")
    if not isinstance(payload["segments"], list) or not payload["segments"]:
        raise ValueError(f"{p} contains no route segments")
    return payload

