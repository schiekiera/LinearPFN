"""Agreement between two posterior summaries of the same datasets (PIPs,
posterior-mean coefficients and median probability models), used by the
real-outcome comparison against the MCMC sampler.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _masks(p: int, d1: int) -> tuple[np.ndarray, np.ndarray]:
    mains = np.zeros(d1, dtype=bool)
    mains[:p] = True
    return mains, ~mains


def _stats(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.size == 0:
        return {"r": None, "rmse": None, "n": 0}
    r = None
    if a.size > 1 and a.std() > 0 and b.std() > 0:
        r = float(np.corrcoef(a, b)[0, 1])
    return {"r": r, "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
            "max_abs": float(np.abs(a - b).max()), "n": int(a.size)}


def agreement(rows_a: dict, rows_b: dict, keys: list[str], field: str) -> dict[str, Any] | None:
    """Pooled agreement of `field` (prob: d-1 effects; coef: d incl. intercept,
    intercept dropped) between two rows over shared keys; None without pairs."""
    parts: dict[str, list] = {"mains": [], "ints": [], "all": []}
    n_pairs = 0
    for k in keys:
        a, b = rows_a.get(k), rows_b.get(k)
        if a is None or b is None:
            continue
        va, vb = np.asarray(a[field], float), np.asarray(b[field], float)
        if va.size == 0 or vb.size == 0 or va.shape != vb.shape:
            continue
        if field == "coef":
            va, vb = va[1:], vb[1:]
        mains, ints = _masks(int(a["p"]), va.size)
        for name, m in (("mains", mains), ("ints", ints)):
            parts[name].append((va[m], vb[m]))
        parts["all"].append((va, vb))
        n_pairs += 1
    if n_pairs == 0:
        return None
    out = {"n_datasets": n_pairs}
    for name, lst in parts.items():
        a = np.concatenate([x for x, _ in lst])
        b = np.concatenate([y for _, y in lst])
        out[name] = _stats(a, b)
    return out
