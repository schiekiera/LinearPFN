"""Repository root, the single config, and resolved paths.

`ROOT` is derived from this file (pipeline/ sits directly under the root), so
every stage works from any working directory. `load_config()` reads
configs/pipeline.yaml once; `resolve()` turns its `paths` block into absolute
`Path`s (lists stay lists). Nothing here checks that a path exists — an
input is absent until the stage that produces it has run.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "pipeline.yaml"


@functools.lru_cache(maxsize=8)
def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """The pipeline config as a dict (cached per path)."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    for key in ("paths", "cluster", "benchmark", "style"):
        if key not in cfg:
            raise KeyError(f"{path}: missing top-level block {key!r}")
    return cfg


def resolve(cfg: dict[str, Any] | None = None, root: Path = ROOT) -> dict[str, Any]:
    """`paths` block -> {name: absolute Path | [absolute Path, ...]}."""
    cfg = cfg or load_config()
    out: dict[str, Any] = {}
    for name, value in cfg["paths"].items():
        if isinstance(value, list):
            out[name] = [root / v for v in value]
        else:
            out[name] = root / value
    return out


def path(name: str, cfg: dict[str, Any] | None = None) -> Path:
    """One resolved path by name (raises for list-valued entries)."""
    value = resolve(cfg)[name]
    if isinstance(value, list):
        raise TypeError(f"paths.{name} is a list; use resolve()")
    return value


def rel(p: Path | str, root: Path = ROOT) -> str:
    """Repository-relative POSIX string (for provenance records and messages)."""
    p = Path(p)
    try:
        return p.resolve().relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def numbers_dir(cfg: dict[str, Any] | None = None) -> Path:
    """reports/paper — the numbers folder (created on demand)."""
    d = path("numbers", cfg)
    d.mkdir(parents=True, exist_ok=True)
    return d
