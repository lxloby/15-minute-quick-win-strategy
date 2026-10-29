"""Load config.yaml and resolve the handful of 'inherit from elsewhere' fields."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config(dict):
    """dict with attribute access and dotted lookup, so cfg.get('a.b.c') works."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def dotted(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, key: str) -> Path:
        """Resolve a configured directory against the project root."""
        p = Path(self.dotted(f"paths.{key}"))
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    # swing_pivot_length defaults to the breaker indicator's Length, as specified.
    if raw["strategy"].get("swing_pivot_length") is None:
        raw["strategy"]["swing_pivot_length"] = raw["indicators"]["breaker"]["length"]

    cfg = Config(raw)
    for key in ("data_dir", "cache_dir", "out_dir"):
        cfg.path(key).mkdir(parents=True, exist_ok=True)
    return cfg


def pip_size(cfg: Config, symbol: str) -> float:
    try:
        return float(cfg["data"]["pip_size"][symbol])
    except KeyError:
        raise KeyError(
            f"No pip_size configured for {symbol}; add it under data.pip_size"
        ) from None
