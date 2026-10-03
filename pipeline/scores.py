"""Per-method scores of a set of benchmark rows: AUC and F1 per dataset, coefficient
RMSE, precision, recall and fit-time summaries. Metrics come from bench.report and
bench.shared, so every number stage scores rows the same way.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.report import _tier1
from bench.shared import _macro_rmse


def prf_dataset(sel: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    """Per-dataset precision, recall, F1 with the 0/0 -> 0 convention."""
    tp = float((sel & truth).sum())
    p = tp / sel.sum() if sel.sum() else 0.0
    r = tp / truth.sum() if truth.sum() else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def macro_prf(rows: list[dict]) -> dict[str, float]:
    """Macro precision/recall/F1 at each row's own selection, over datasets
    with >= 1 true active (matches bench.report's macro-F1 denominator)."""
    ps, rs, fs = [], [], []
    for r in rows:
        t = r["gamma_true"].astype(bool)
        if not t.any():
            continue
        p, rc, f = prf_dataset(r["gamma_hat"].astype(bool), t)
        ps.append(p)
        rs.append(rc)
        fs.append(f)
    nan = float("nan")
    return {"precision_macro": float(np.mean(ps)) if ps else nan,
            "recall_macro": float(np.mean(rs)) if rs else nan,
            "f1_macro_check": float(np.mean(fs)) if fs else nan,
            "n_scored": len(fs)}


def fit_seconds(rows: list[dict]) -> dict[str, float]:
    s = np.asarray([r["fit_seconds"] for r in rows], dtype=float)
    return {"median": float(np.median(s)), "mean": float(s.mean()),
            "p90": float(np.percentile(s, 90)), "max": float(s.max()),
            "total_core_h": float(s.sum() / 3600.0)}


def metrics(rows: list[dict]) -> dict[str, Any]:
    """Every per-cell number the paper may quote for one method on one slice."""
    if not rows:
        return {"n": 0}
    t = _tier1(rows)
    out = {"n": len(rows),
           "auc_macro": t["auc_macro"], "auc_pooled": t["auc"], "pr_auc_pooled": t["pr_auc"],
           "f1_macro": t["f1_macro"], "f1_pooled": t["f1"],
           "precision_pooled": t["precision"], "recall_pooled": t["recall"],
           "heredity_viol": t["heredity_viol"], "n_effects": int(t["n_effects"]),
           "rmse_macro": _macro_rmse(rows)}
    out.update(macro_prf(rows))
    out["fit_seconds"] = fit_seconds(rows)
    return out
