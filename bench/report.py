"""Tiered benchmark report from persisted per-dataset results.

Tiers stay separate (different questions, different eligible methods):
- Tier 1 — selection vs TRUE gamma, all methods: AUC + PR-AUC primary
  (threshold-free, from each method's score), F1/precision/recall at each
  method's own default operating point, heredity-violation rate (selected
  interaction with an unselected parent).
- Tier 2 — probability quality, only methods emitting probabilities:
  Murphy decomposition / ECE vs truth (reusing linearpfn.validate._murphy),
  plus r vs the EXACT posterior's PIPs on p <= 5 datasets (paired through
  the persisted 'exact' rows). Stability frequencies carry the standing
  caveat: selection frequencies are not posterior probabilities.
- Tier 3 — coefficients: RMSE vs TRUE beta for every method that returns
  coefficients; RMSE vs the EXACT posterior mean only for Bayesian methods.
  Two tables, two estimands — mixing them would be a category error.
- Timing — median + IQR of fit_seconds per method per p (single-core
  pinned protocol; see bench.harness).

Strata first (per p, pooled over n; collinearity terciles by each dataset's
max off-diagonal |r|), pooled numbers last.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bench.methods import TIER2_METHODS
from linearpfn.prior import interaction_pairs
from linearpfn.validate import _murphy

BAYES_COEF_METHODS = ("linearpfn", "susie", "exact")


def _auc_midrank(score: np.ndarray, truth: np.ndarray) -> float:
    """ROC AUC via Mann-Whitney with MIDRANKS: tied scores (common for
    path-entry ranks, where every never-active effect scores 0) share their
    average rank instead of an arbitrary sort-order-dependent one."""
    from scipy.stats import rankdata

    pos = truth.astype(bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _average_precision(score: np.ndarray, truth: np.ndarray) -> float:
    pos = truth.astype(bool)
    if pos.sum() == 0 or pos.all():
        return float("nan")
    order = np.argsort(-score, kind="stable")
    hits = pos[order]
    prec = np.cumsum(hits) / np.arange(1, hits.size + 1)
    return float(prec[hits].sum() / pos.sum())


def _is_main_mask(rows: list[dict]) -> np.ndarray:
    return np.concatenate(
        [np.arange(r["gamma_true"].size) < r["p"] for r in rows]
    )


def _heredity_rate(rows: list[dict]) -> float:
    viol = total = 0
    for r in rows:
        p = r["p"]
        sel = r["gamma_hat"]
        if sel.size <= p:
            continue
        pairs = np.asarray(interaction_pairs(p), dtype=int).reshape(-1, 2)
        ints = sel[p:]
        parents_in = sel[pairs[:, 0]] & sel[pairs[:, 1]]
        viol += int((ints & ~parents_in).sum())
        total += len(pairs)
    return viol / max(total, 1)


def _tier1(rows: list[dict]) -> dict:
    truth = np.concatenate([r["gamma_true"] for r in rows]).astype(float)
    sel = np.concatenate([r["gamma_hat"] for r in rows])
    have_score = all(r["score"] is not None for r in rows)
    if have_score:
        score = np.concatenate([r["score"] for r in rows])
        auc = _auc_midrank(score, truth)
        pr_auc = _average_precision(score, truth)
        per_ds = [_auc_midrank(r["score"], r["gamma_true"].astype(float))
                  for r in rows]
        per_ds = [a for a in per_ds if not np.isnan(a)]
        auc_macro = float(np.mean(per_ds)) if per_ds else float("nan")
    else:
        auc = pr_auc = auc_macro = float("nan")
    tp = float((sel & (truth > 0)).sum())
    prec = tp / max(sel.sum(), 1)
    rec = tp / max(truth.sum(), 1)
    # Macro F1: per-dataset F1 averaged over datasets with >= 1 true active
    # (pooled F1 lets a few large-p datasets dominate the count of effects).
    per_f1 = []
    for r in rows:
        t = r["gamma_true"]
        if not t.any():
            continue
        s = r["gamma_hat"]
        tp_d = float((s & t).sum())
        p_d = tp_d / max(s.sum(), 1)
        r_d = tp_d / max(t.sum(), 1)
        per_f1.append(2 * p_d * r_d / max(p_d + r_d, 1e-12))
    return {
        "auc": auc,
        "auc_macro": auc_macro,
        "pr_auc": pr_auc,
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-12),
        "f1_macro": float(np.mean(per_f1)) if per_f1 else float("nan"),
        "heredity_viol": _heredity_rate(rows),
        "n_effects": truth.size,
    }


def _tier2(rows: list[dict], exact_by_key: dict) -> dict | None:
    if any(r["prob"] is None for r in rows):
        return None
    prob = np.concatenate([r["prob"] for r in rows])
    truth = np.concatenate([r["gamma_true"] for r in rows]).astype(float)
    m = _murphy(prob, truth)
    out = {"ece": m["ece"], "brier": m["brier"], "resolution": m["resolution"],
           "reliability": m["reliability"]}
    paired = [(r, exact_by_key.get((r["n"], r["p"], r["i"]))) for r in rows
              if r["p"] <= 5]
    paired = [(r, e) for r, e in paired if e is not None and e["prob"] is not None]
    if paired:
        pr = np.concatenate([r["prob"] for r, _ in paired])
        ex = np.concatenate([e["prob"] for _, e in paired])
        out["r_vs_exact_p5"] = float(np.corrcoef(pr, ex)[0, 1])
        out["rmse_vs_exact_p5"] = float(np.sqrt(np.mean((pr - ex) ** 2)))
    return out


def _tier3(rows: list[dict], exact_by_key: dict) -> dict | None:
    with_coef = [r for r in rows if r["coef"] is not None]
    if not with_coef:
        return None
    err_true = np.concatenate(
        [r["coef"][1:] - r["beta_true"][1:] for r in with_coef]
    )
    out = {"rmse_vs_true": float(np.sqrt(np.mean(err_true**2))),
           "n_datasets": len(with_coef)}
    paired = [(r, exact_by_key.get((r["n"], r["p"], r["i"]))) for r in with_coef
              if r["p"] <= 5]
    paired = [(r, e) for r, e in paired if e is not None and e["coef"] is not None]
    if paired:
        err_ex = np.concatenate(
            [r["coef"][1:] - e["coef"][1:] for r, e in paired]
        )
        out["rmse_vs_exact_p5"] = float(np.sqrt(np.mean(err_ex**2)))
    return out


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "-" if np.isnan(v) else f"{v:.4f}"
    return str(v)


def _table(header: list[str], rows: list[list]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows),
        "",
    ]


def write_report(
    results: dict[str, list[dict]],
    out_path: Path,
    notes: list[str] | None = None,
) -> None:
    """results: method name -> list of persisted per-dataset rows."""
    exact_by_key = {
        (r["n"], r["p"], r["i"]): r for r in results.get("exact", [])
    }
    all_rows = [r for rows in results.values() for r in rows]
    p_values = sorted({r["p"] for r in all_rows})
    absr = np.asarray([r["max_absr"] for r in all_rows])
    edges = np.percentile(absr, [100 / 3, 200 / 3]) if absr.size else [0, 0]
    lines = ["# Benchmark report", ""]
    if notes:
        lines += ["**Run notes (nothing here is silent):**", ""]
        lines += [f"- {n}" for n in notes] + [""]
    lines += [
        "Ground truth: prior-sampled datasets (deterministic per-cell streams;",
        "all methods see identical data). 'exact' is the Bayes-optimal ceiling",
        "under the true prior, not a competitor. Timing: single-core pinned,",
        "fit+tuning inside the method's process. Collinearity terciles are by",
        f"each dataset's max off-diagonal |r| (edges {edges[0]:.2f}, {edges[1]:.2f}).",
        "",
        "AUC/PR-AUC use each method's native ranking (PIPs for Bayesian",
        "methods, path-entry order for penalized ones) with midrank ties.",
        "Pooled AUC ranks effects across ALL datasets, which assumes scores",
        "share a scale between datasets — true for calibrated probabilities,",
        "only roughly for path ranks — so 'AUC/ds' (per-dataset macro",
        "average, the variant fairest to path methods) is shown alongside.",
        "Methods emitting only hard selections have no ranking and show '-'",
        "in these columns; their F1/prec/recall use their own threshold.",
        "",
        "## Tier 1 — selection vs true gamma (all methods)",
        "",
    ]
    for p in p_values:
        rows_t = []
        for m, rows in sorted(results.items()):
            sub = [r for r in rows if r["p"] == p]
            if sub:
                t = _tier1(sub)
                rows_t.append([m, t["auc"], t["auc_macro"], t["pr_auc"],
                               t["f1"], t["precision"], t["recall"],
                               t["heredity_viol"]])
        lines += [f"### p = {p}", ""]
        lines += _table(["method", "AUC", "AUC/ds", "PR-AUC", "F1", "prec",
                         "recall", "heredity viol"], rows_t)
    lines += ["### pooled (all strata)", ""]
    lines += _table(
        ["method", "AUC", "AUC/ds", "PR-AUC", "F1", "prec", "recall",
         "heredity viol"],
        [[m, *(lambda t: [t["auc"], t["auc_macro"], t["pr_auc"], t["f1"],
                          t["precision"], t["recall"],
                          t["heredity_viol"]])(_tier1(rows))]
         for m, rows in sorted(results.items()) if rows],
    )
    lines += ["### pooled by collinearity tercile (max |r| of X)", ""]
    for t_lo, t_hi, label in ((-1, edges[0], "low"), (edges[0], edges[1], "mid"),
                              (edges[1], 2.0, "high")):
        rows_t = []
        for m, rows in sorted(results.items()):
            sub = [r for r in rows if t_lo < r["max_absr"] <= t_hi]
            if sub:
                t1 = _tier1(sub)
                rows_t.append([m, t1["auc"], t1["auc_macro"], t1["pr_auc"],
                               t1["f1"]])
        lines += [f"#### {label} collinearity", ""]
        lines += _table(["method", "AUC", "AUC/ds", "PR-AUC", "F1"], rows_t)
    lines += [
        "## Tier 2 — probability quality (probability-emitting methods only)",
        "",
        "Stability frequencies are NOT posterior probabilities (expected",
        "miscalibration is part of what this tier measures); SuSiE's PIPs",
        "target a heredity-free prior.",
        "",
    ]
    rows_t = []
    for m in [m for m in sorted(results) if m in TIER2_METHODS]:
        t2 = _tier2(results[m], exact_by_key)
        if t2:
            rows_t.append([m, t2["ece"], t2["brier"], t2["resolution"],
                           t2.get("r_vs_exact_p5"), t2.get("rmse_vs_exact_p5")])
    lines += _table(["method", "ECE", "Brier", "resolution", "r vs exact (p<=5)",
                     "RMSE vs exact (p<=5)"], rows_t)
    lines += [
        "## Tier 3 — coefficients",
        "",
        "Two estimands, two tables: frequentist methods target the true beta;",
        "Bayesian posterior means also compare against the exact posterior",
        "mean (p <= 5). Do not read across.",
        "",
        "### vs true beta (all methods returning coefficients)",
        "",
    ]
    rows_t = []
    for m, rows in sorted(results.items()):
        t3 = _tier3(rows, exact_by_key)
        if t3:
            rows_t.append([m, t3["rmse_vs_true"], t3["n_datasets"]])
    lines += _table(["method", "RMSE vs true beta", "datasets"], rows_t)
    lines += ["### vs exact posterior mean (Bayesian methods, p <= 5)", ""]
    rows_t = []
    for m in [m for m in sorted(results) if m in BAYES_COEF_METHODS]:
        t3 = _tier3(results[m], exact_by_key)
        if t3 and "rmse_vs_exact_p5" in t3:
            rows_t.append([m, t3["rmse_vs_exact_p5"]])
    lines += _table(["method", "RMSE vs exact coef mean"], rows_t)
    lines += ["## Timing (single-core pinned; fit + tuning; seconds/dataset)", ""]
    rows_t = []
    for m, rows in sorted(results.items()):
        per_p = []
        for p in p_values:
            secs = [r["fit_seconds"] for r in rows if r["p"] == p]
            per_p.append(f"{np.median(secs):.2f} [{np.percentile(secs, 25):.2f}, "
                         f"{np.percentile(secs, 75):.2f}]" if secs else "-")
        rows_t.append([m, *per_p])
    lines += _table(["method"] + [f"p={p}" for p in p_values], rows_t)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
