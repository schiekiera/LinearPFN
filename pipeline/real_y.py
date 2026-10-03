"""The REAL-OUTCOME experiment (stage 18 -> stage 76): LinearPFN against the
posterior it targets on published tables with their actual response.

Rows under results/real_outcomes/<row>/<dataset>.npz (stage 18): the model row
and the reference row `mc3`, the MCMC sampler (one sampler on every table).
Per dataset: PIP, coefficient (head and probe) and median-probability-model
agreement with the reference, the sampler's convergence, and held-out
predictive NLL / RMSE of the model, the reference, OLS on the mains and the
context mean. Pooled: agreement over every slot (pipeline.agreement), the
median per-dataset statistics, and the held-out scores as a paired comparison.
Datasets listed in the
config but absent from the cache are reported as pending, never substituted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pipeline.agreement import _masks, _stats, agreement

FIELDS_KEEP = ("prob", "coef", "coef_probe", "fit_seconds", "probe_seconds", "nll_query",
               "rmse_query", "rhat_max", "ess_min", "converged", "converged_ctx", "n_steps",
               "n_chains", "kind", "n", "p", "n_ctx", "n_query", "y_sd", "y_mean",
               "nll_marginal_query", "rmse_marginal_query", "ols_mains_nll_query",
               "ols_mains_rmse_query", "features", "fit_seconds_ctx", "wall_seconds")


def load_rows(results_dir: Path | str, rows: tuple[str, ...]) -> dict[str, dict[str, dict]]:
    """{row: {dataset: record}}; arrays as stored, scalars unwrapped."""
    out: dict[str, dict[str, dict]] = {}
    for row in rows:
        d = Path(results_dir) / row
        recs: dict[str, dict] = {}
        for path in sorted(d.glob("*.npz")) if d.is_dir() else []:
            z = np.load(path, allow_pickle=False)
            rec = {}
            for k in z.files:
                if k not in FIELDS_KEEP:
                    continue
                v = z[k]
                rec[k] = v.item() if v.ndim == 0 else v
            recs[path.stem] = rec
        out[row] = recs
    return out


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.sum(a | b))
    return 1.0 if union == 0 else float(np.sum(a & b) / union)


def _per_dataset(name: str, m: dict, r: dict, feats: list[str]) -> dict[str, Any]:
    p, d1 = int(m["p"]), int(np.asarray(m["prob"]).size)
    mains, ints = _masks(p, d1)
    pm, pr = np.asarray(m["prob"], float), np.asarray(r["prob"], float)
    cm, cr = np.asarray(m["coef"], float)[1:], np.asarray(r["coef"], float)[1:]
    sm, sr = pm > 0.5, pr > 0.5
    out: dict[str, Any] = {
        "n": int(m["n"]), "p": p, "d": d1, "n_ctx": int(m["n_ctx"]), "n_query": int(m["n_query"]),
        "reference": r["kind"], "converged": bool(r.get("converged", True)),
        "converged_ctx": bool(r.get("converged_ctx", True)),
        "y_sd_raw": float(m["y_sd"]),
        "seconds": {"model": float(m["fit_seconds"]), "reference": float(r["fit_seconds"]),
                    "probe": float(m.get("probe_seconds", float("nan")))},
        "mc3": None if r["kind"] != "mc3" else {
            "rhat_max": float(r["rhat_max"]), "ess_min": float(r["ess_min"]),
            "steps": int(r["n_steps"]), "chains": int(r["n_chains"])},
        "pip": {"mains": _stats(pm[mains], pr[mains]), "ints": _stats(pm[ints], pr[ints]),
                "all": _stats(pm, pr)},
        "coef": {"head": _stats(cm, cr), "head_mains": _stats(cm[mains], cr[mains]),
                 "head_ints": _stats(cm[ints], cr[ints]),
                 "probe": _stats(np.asarray(m["coef_probe"], float)[1:], cr)
                 if "coef_probe" in m else None},
        "mpm": {"jaccard": _jaccard(sm, sr), "n_same": int(np.sum(sm == sr)), "n_slots": d1,
                "size_model": int(sm.sum()), "size_reference": int(sr.sum()),
                "model": [feats[i] for i in np.flatnonzero(sm)],
                "reference": [feats[i] for i in np.flatnonzero(sr)]},
        "held_out": {
            "nll": {"model": float(m["nll_query"]), "reference": float(r["nll_query"]),
                    "ols_mains": float(m["ols_mains_nll_query"]),
                    "marginal": float(m["nll_marginal_query"])},
            "rmse": {"model": float(m["rmse_query"]), "reference": float(r["rmse_query"]),
                     "ols_mains": float(m["ols_mains_rmse_query"]),
                     "marginal": float(m["rmse_marginal_query"])}},
        "pip_table": {"mains": [{"effect": feats[i], "model": float(pm[i]),
                                 "reference": float(pr[i]), "coef_model": float(cm[i]),
                                 "coef_reference": float(cr[i])} for i in range(p)],
                      "top_ints": [{"effect": feats[i], "model": float(pm[i]),
                                    "reference": float(pr[i])}
                                   for i in np.argsort(-np.maximum(pm, pr))
                                   if i >= p and max(pm[i], pr[i]) >= 0.1]},
    }
    out["name"] = name
    return out


def _labels(feats: np.ndarray, p: int, d1: int) -> list[str]:
    from linearpfn.prior import _pairs_array

    names = [str(f) for f in feats]
    pairs = _pairs_array(p) if d1 > p else np.zeros((0, 2), int)
    return names + [f"{names[i]}:{names[j]}" for i, j in pairs]


def _median(vals: list[float]) -> float | None:
    v = [x for x in vals if x is not None and x == x]
    return float(np.median(v)) if v else None


def _mean(vals: list[float]) -> float | None:
    v = [x for x in vals if x is not None and x == x]
    return float(np.mean(v)) if v else None


def _paired(ds: list[dict], metric: str, a: str, b: str) -> dict[str, Any]:
    """Held-out `metric` of a minus b per dataset: mean, median, wins of a."""
    diffs = [d["held_out"][metric][a] - d["held_out"][metric][b] for d in ds]
    return {"mean": float(np.mean(diffs)), "median": float(np.median(diffs)),
            "wins": int(sum(x < 0 for x in diffs)), "n": len(diffs)}


def summarize(cfg: dict, results_dir: Path | str) -> dict[str, Any]:
    model_row = cfg["model_row"]
    wanted = [d["name"] if isinstance(d, dict) else str(d) for d in cfg["datasets"]]
    notes = {d["name"]: d for d in cfg["datasets"] if isinstance(d, dict)}
    rows = load_rows(results_dir, (model_row, "mc3"))
    model = rows.get(model_row, {})
    per: dict[str, dict] = {}
    pending = []
    for name in wanted:
        m = model.get(name)
        r = rows["mc3"].get(name)
        if m is None or r is None:
            pending.append(name)
            continue
        feats = _labels(m["features"], int(m["p"]), int(np.asarray(m["prob"]).size))
        rec = _per_dataset(name, m, r, feats)
        rec["outcome"] = notes.get(name, {}).get("outcome")
        rec["stress"] = bool(notes.get(name, {}).get("stress", False))
        per[name] = rec
    names = [n for n in wanted if n in per]
    conv = [n for n in names if per[n]["converged"] and per[n]["converged_ctx"]]
    ds = [per[n] for n in conv]
    ref_rows = {n: rows["mc3"][n] for n in conv}
    mod_rows = {n: model[n] for n in conv}
    for rec in ref_rows.values():
        rec["p"] = int(rec["p"])
    probe_rows = {n: {**model[n], "coef": model[n]["coef_probe"]} for n in conv
                  if "coef_probe" in model[n]}
    pooled = None
    if ds:
        pooled = {
            "n_datasets": len(ds),
            "pip": agreement(mod_rows, ref_rows, conv, "prob"),
            "coef_head": agreement(mod_rows, ref_rows, conv, "coef"),
            "coef_probe": agreement(probe_rows, ref_rows, conv, "coef") if probe_rows else None,
            # mean over TABLES, so the seven carry equal weight whatever their
            # size; the `pip` / `coef_head` blocks above pool over slots instead
            # and the two therefore differ. The main-text summary table reads
            # this block.
            "mean": {
                "pip_r_all": _mean([d["pip"]["all"]["r"] for d in ds]),
                "pip_r_mains": _mean([d["pip"]["mains"]["r"] for d in ds]),
                "pip_rmse_all": _mean([d["pip"]["all"]["rmse"] for d in ds]),
                "coef_rmse_head": _mean([d["coef"]["head"]["rmse"] for d in ds]),
                "mpm_frac": _mean([d["mpm"]["n_same"] / d["mpm"]["n_slots"] for d in ds]),
            },
            "median": {
                "pip_r_mains": _median([d["pip"]["mains"]["r"] for d in ds]),
                "pip_rmse_all": _median([d["pip"]["all"]["rmse"] for d in ds]),
                "pip_max_abs": _median([d["pip"]["all"]["max_abs"] for d in ds]),
                "coef_rmse_head": _median([d["coef"]["head"]["rmse"] for d in ds]),
                "mpm_jaccard": _median([d["mpm"]["jaccard"] for d in ds]),
                "seconds_model": _median([d["seconds"]["model"] for d in ds]),
                "seconds_reference": _median([d["seconds"]["reference"] for d in ds]),
            },
            "protocol": {
                "n_min": min(d["n"] for d in ds), "n_max": max(d["n"] for d in ds),
                "p_min": min(d["p"] for d in ds), "p_max": max(d["p"] for d in ds),
                "n_stress": int(sum(d["stress"] for d in ds)),
                "query_share_median": float(np.median([d["n_query"] / d["n"] for d in ds])),
                "chains": sorted({d["mc3"]["chains"] for d in ds if d["mc3"]}),
                "steps_min": min(d["mc3"]["steps"] for d in ds if d["mc3"]),
                "steps_max": max(d["mc3"]["steps"] for d in ds if d["mc3"]),
                "rhat_max": max(d["mc3"]["rhat_max"] for d in ds if d["mc3"]),
                "ess_min": min(d["mc3"]["ess_min"] for d in ds if d["mc3"]),
            },
            "mpm_identical": int(sum(d["mpm"]["jaccard"] == 1.0 for d in ds)),
            "mpm_slots_same": int(sum(d["mpm"]["n_same"] for d in ds)),
            "mpm_slots": int(sum(d["mpm"]["n_slots"] for d in ds)),
            "max_pip_gap": {"value": max(d["pip"]["all"]["max_abs"] for d in ds),
                            "dataset": max(ds, key=lambda d: d["pip"]["all"]["max_abs"])["name"]},
            "held_out": {
                "nll": {k: float(np.mean([d["held_out"]["nll"][k] for d in ds]))
                        for k in ("model", "reference", "ols_mains", "marginal")},
                "rmse": {k: float(np.mean([d["held_out"]["rmse"][k] for d in ds]))
                         for k in ("model", "reference", "ols_mains", "marginal")},
                "model_minus_reference": {m: _paired(ds, m, "model", "reference")
                                          for m in ("nll", "rmse")},
                "model_minus_ols": {m: _paired(ds, m, "model", "ols_mains")
                                    for m in ("nll", "rmse")},
            },
        }
    return {"model_row": model_row, "datasets": per, "order": names, "pending": pending,
            "not_converged": [n for n in names if n not in conv], "pooled": pooled,
            "preprocessing": "features z-scored (ddof 0); y centred and scaled to unit sd; "
                             "held-out split uses the context rows' statistics",
            "reference_rule": "MC3 (part-A ladder, 4 chains) on every table"}
