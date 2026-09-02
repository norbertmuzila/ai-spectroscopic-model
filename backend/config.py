"""Configuration loading and project paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("SPECTRO_CONFIG", ROOT / "config" / "config.yaml"))


class Config(dict):
    """Dict with dotted-path access: cfg.get_path('model.train.samples_per_class')."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def resolve(self, dotted: str, default: Any = None) -> Path:
        """Resolve a config value that is a path, relative to the project root."""
        value = self.get_path(dotted, default)
        if value is None:
            raise KeyError(f"missing path config: {dotted}")
        p = Path(value)
        return p if p.is_absolute() else (ROOT / p)


_cache: Config | None = None


def load_config(reload: bool = False) -> Config:
    global _cache
    if _cache is None or reload:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            _cache = Config(yaml.safe_load(fh))
    return _cache


def analysis_grid():
    """The common wavelength grid (nm) every spectrum is resampled onto."""
    import numpy as np

    cfg = load_config()
    lo = float(cfg.get_path("acquisition.grid_min_nm", 350.0))
    hi = float(cfg.get_path("acquisition.grid_max_nm", 1000.0))
    step = float(cfg.get_path("acquisition.grid_step_nm", 1.0))
    return np.arange(lo, hi + 0.5 * step, step, dtype=np.float64)


def ensure_dirs() -> None:
    cfg = load_config()
    for key in ("library.usgs_dir", "model.dir", "server.reports_dir"):
        cfg.resolve(key).mkdir(parents=True, exist_ok=True)
    cfg.resolve("server.database").parent.mkdir(parents=True, exist_ok=True)
    cfg.resolve("library.store_path").parent.mkdir(parents=True, exist_ok=True)
    cfg.resolve("spectrasuite.watch_dir").mkdir(parents=True, exist_ok=True)
