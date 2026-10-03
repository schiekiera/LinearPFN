"""Shared two-liner for the stage scripts: repo root on sys.path (any depth,
any cwd) and the config. Imported as `from _bootstrap import *` by scripts in
this folder; the folder itself is on sys.path when a script runs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import paths, store  # noqa: E402

CFG = paths.load_config()
P = paths.resolve(CFG)
NUM = P["numbers"]
PROV = NUM / "_prov"
FIG = P["figures"]
CONFIG = paths.CONFIG_PATH

__all__ = ["ROOT", "CFG", "P", "NUM", "PROV", "FIG", "CONFIG", "paths", "store", "Path"]
