"""Provenance-aware JSON outputs and content hashes.

Every number file under reports/paper/ is written through
`write_json(path, payload, inputs)`: the payload gets an `_inputs` block
({repo-relative path: sha256}) and a `_stamp` (which stage wrote it).
`is_current` reports an output fresh when every recorded hash still matches
— never by mtime, which `git checkout` and `rsync -a` scramble. Directory
inputs are represented by a cheap listing hash (`dir_stamp`), so no stage
hashes 100k files.

JSON is written with sorted keys and a fixed float representation so a
rerun on unchanged inputs is byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from pipeline.paths import ROOT, rel


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dir_stamp(directory: Path, pattern: str = "**/*") -> dict[str, Any]:
    """A cheap content stamp of a directory: sorted (relative path, size)
    pairs hashed together, plus the count. Sizes, not contents — it detects
    change, not corruption."""
    directory = Path(directory)
    entries = sorted((p.relative_to(directory).as_posix(), p.stat().st_size)
                     for p in directory.glob(pattern) if p.is_file())
    h = hashlib.sha256()
    for name, size in entries:
        h.update(f"{name}\0{size}\n".encode())
    return {"path": rel(directory, ROOT), "files": len(entries), "sha256": h.hexdigest()}


def _clean(x: Any) -> Any:
    """numpy -> python; NaN/inf -> None (JSON has no NaN); tuples -> lists."""
    if hasattr(x, "tolist"):
        return _clean(x.tolist())
    if hasattr(x, "item") and not isinstance(x, (list, dict, str, bytes)):
        return _clean(x.item())
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def hash_inputs(inputs: list[Path | str]) -> dict[str, str]:
    """{repo-relative path: sha256} for every existing file; a directory is
    represented by its dir_stamp hash."""
    out: dict[str, str] = {}
    for p in inputs:
        p = Path(p)
        if not p.is_absolute():
            p = ROOT / p
        if p.is_dir():
            out[rel(p, ROOT) + "/"] = dir_stamp(p)["sha256"]
        elif p.is_file():
            out[rel(p, ROOT)] = sha256_file(p)
        else:
            out[rel(p, ROOT)] = "MISSING"
    return dict(sorted(out.items()))


def write_json(path: Path, payload: dict[str, Any], inputs: list[Path | str],
               stage: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"_stage": stage, "_inputs": hash_inputs(inputs), **_clean(payload)}
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return path


def read_json(path: Path | str) -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def is_current(path: Path | str) -> bool:
    """True when the file exists and every recorded input hash still holds."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        doc = read_json(path)
    except (OSError, ValueError):
        return False
    recorded = doc.get("_inputs", {})
    for key, digest in recorded.items():
        p = ROOT / key.rstrip("/")
        if key.endswith("/"):
            now = dir_stamp(p)["sha256"] if p.is_dir() else "MISSING"
        else:
            now = sha256_file(p) if p.is_file() else "MISSING"
        if now != digest:
            return False
    return True
