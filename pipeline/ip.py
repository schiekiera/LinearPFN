"""Main-benchmark rows -> metric tables.

Reads <results>/<method>/ip<panel>_*.npz (the fits of stage 17) and never
refits. Metric definitions are the benchmark's own
(`pipeline.scores.metrics`: macro F1 at each method's own selection over
datasets with >= 1 true active, pooled coefficient RMSE over every
candidate slot with the intercept excluded, midrank AUC), so every number
stage scores rows the same way. This module only loads the rows and slices
them: pooled, by p bin, by n bin, by realized-density cell (k_main/p) and
by realized-R2 cell. The rows carry no n_bin / p_bin fields in their meta,
so the bins are assigned here from the config's bin edges.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.scores import metrics

PANELS = ("1g", "2g", "mx", "k0", "kc", "ki", "kn", "xc", "xi", "xn")   # 1g/2g: outcomes from
# the prior on the 64 sampled designs and their prior-X twins; mx: the main benchmark on every
# eligible design; k0/kc/ki/kn: a plain-draw misspecification panel (control and the coef / int /
# noise knobs); xc/xi/xn: the same knobs on the main benchmark's own datasets, mx their control


def parse_bins(labels: list[str]) -> list[tuple[int, int, str]]:
    """'3-5' -> (3, 5, '3-5'); a value belongs to the first bin whose [lo, hi] holds it."""
    out = []
    for lab in labels:
        lo, hi = lab.split("-")
        out.append((int(lo), int(hi), lab))
    return out


def bin_of(value: int, bins: list[tuple[int, int, str]]) -> str | None:
    for lo, hi, lab in bins:
        if lo <= value <= hi:
            return lab
    return None


def methods_present(results_dir: Path) -> list[str]:
    d = Path(results_dir)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name != "truth")


def _load_rows(out_dir: Path, method: str, prefix: str) -> list[dict]:
    """bench.shared._load_all with optional fields: the MCMC sampler's rows
    (stage 20) carry no max_absr / prob."""
    import json

    import numpy as np

    def opt(z, key):
        return z[key] if key in z.files and z[key].size else None

    rows = []
    for path in sorted((out_dir / method).glob(f"{prefix}*.npz")):
        z = np.load(path, allow_pickle=False)
        rows.append({"n": int(z["n"]), "p": int(z["p"]), "i": path.stem,
                     "gamma_hat": z["gamma_hat"].astype(bool), "score": opt(z, "score"),
                     "prob": opt(z, "prob"), "coef": opt(z, "coef"),
                     "fit_seconds": float(z["fit_seconds"]),
                     "gamma_true": z["gamma_true"].astype(bool), "beta_true": z["beta_true"],
                     "max_absr": float(z["max_absr"]) if "max_absr" in z.files else float("nan"),
                     "meta": json.loads(str(z["meta"]))})
    return rows


def load_panel(results_dir: Path, method: str, panel: str) -> list[dict]:
    """Rows of one method on one panel; strict key prefix + meta check."""
    if panel not in PANELS:
        raise ValueError(f"unknown panel {panel!r}")
    if not (Path(results_dir) / method).is_dir():
        return []
    rows = _load_rows(Path(results_dir), method, prefix=f"ip{panel}_")
    pat = re.compile(rf"^ip{panel}_")
    return [r for r in rows if r["meta"].get("panel") == f"ip{panel}" and pat.match(r["i"])]


def load_all(results_dir: Path, methods: list[str] | None = None,
             panels: tuple[str, ...] = PANELS) -> dict[str, dict[str, list[dict]]]:
    """{panel: {method: rows}} for every method dir present (empty lists dropped)."""
    methods = methods or methods_present(Path(results_dir))
    out: dict[str, dict[str, list[dict]]] = {}
    for panel in panels:
        by_m = {m: load_panel(results_dir, m, panel) for m in methods}
        by_m = {m: r for m, r in by_m.items() if r}
        if by_m:
            out[panel] = by_m
    return out


def _by(rows: list[dict], fn) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        k = fn(r)
        if k is not None:
            out[k].append(r)
    return dict(out)


def panel_table(by_method: dict[str, list[dict]], p_bins: list[str],
                n_bins: list[str]) -> dict[str, Any]:
    """pooled / p-bin / n-bin / density-cell / R2-cell metrics per method.
    Cell labels are the realized 0.1-wide bins' lower edges (frac_target and
    r2_target in the row meta, which stage 15 assigns from the realized
    k_main/p and R2)."""
    pb, nb = parse_bins(p_bins), parse_bins(n_bins)
    table: dict[str, Any] = {"pooled": {}, "p_bin": {}, "n_bin": {}, "density": {}, "signal": {}}
    slicers = {"p_bin": lambda r: bin_of(r["p"], pb), "n_bin": lambda r: bin_of(r["n"], nb),
               "density": lambda r: f"{r['meta']['frac_target']:g}",
               "signal": lambda r: f"{r['meta']['r2_target']:g}"}
    for m, rows in sorted(by_method.items()):
        table["pooled"][m] = metrics(rows)
        for key, fn in slicers.items():
            for b, sub in _by(rows, fn).items():
                table[key].setdefault(b, {})[m] = metrics(sub)
    return table


SLICES = ("p_bin", "n_bin", "density", "signal")


def slice_order(key: str, labels, p_bins: list[str], n_bins: list[str]) -> list[str]:
    """p and n bins in the configured order, density and R2 cells by lower edge."""
    if key in ("p_bin", "n_bin"):
        return [b for b in (p_bins if key == "p_bin" else n_bins) if b in labels]
    return sorted(labels, key=float)


def slice_margins(table: dict[str, Any], model: str, baselines: list[str], metric: str,
                  p_bins: list[str], n_bins: list[str]) -> dict[str, list[tuple[str, float]]]:
    """Per slice key, [(bin, model minus the best baseline of that bin)] in bin order:
    the margin the Results text describes for Figure fig:ip-sweeps (higher is
    better, so `metric` is an AUC or F1 key). Bins without the model are skipped."""
    out: dict[str, list[tuple[str, float]]] = {}
    for key in SLICES:
        rows = []
        for b in slice_order(key, table.get(key, {}), p_bins, n_bins):
            cell = table[key][b]
            others = [cell[m][metric] for m in baselines if m in cell]
            if model in cell and others:
                rows.append((b, float(cell[model][metric]) - float(max(others))))
        out[key] = rows
    return out


def counts(all_rows: dict[str, dict[str, list[dict]]]) -> dict[str, dict[str, int]]:
    return {panel: {m: len(r) for m, r in by_m.items()} for panel, by_m in all_rows.items()}


def hard_zeroed(rows: list[dict]) -> list[dict]:
    """The same fits with their dense posterior-mean coefficients set to exactly
    zero outside each row's own selection (gamma_hat, the MPM rule), the
    intercept kept. Selections, scores, PIPs and fit times are the source
    rows', so only the coefficient metrics differ. The paper reports the dense
    mean; this read-out is the like-for-like counterpart
    of the baselines that zero every non-selected slot. Rows without
    coefficients are dropped."""
    out = []
    for r in rows:
        if r.get("coef") is None:
            continue
        coef = np.asarray(r["coef"], float).copy()
        coef[1:] = np.where(np.asarray(r["gamma_hat"], bool), coef[1:], 0.0)
        out.append({**r, "coef": coef})
    return out


# ------------------------------------------------- per-dataset distributions

def dataset_scores(rows: list[dict]) -> dict[str, dict[str, float]]:
    """{dataset key: {auc, f1, rmse, fit_seconds}} with the benchmark's own
    per-dataset definitions (bench.report._tier1): midrank AUC only for
    methods with a native ranking (else absent), F1 only for datasets with
    >= 1 true active, RMSE over every candidate slot with the intercept
    excluded (absent for methods without coefficients)."""
    import numpy as np

    from bench.report import _auc_midrank

    out: dict[str, dict[str, float]] = {}
    for r in rows:
        d: dict[str, float] = {"fit_seconds": float(r["fit_seconds"])}
        t = r["gamma_true"].astype(bool)
        if r.get("score") is not None:
            a = _auc_midrank(np.asarray(r["score"], float), t.astype(float))
            if not np.isnan(a):
                d["auc"] = float(a)
        if t.any():
            s = r["gamma_hat"].astype(bool)
            tp = float((s & t).sum())
            p_d = tp / max(s.sum(), 1)
            r_d = tp / max(t.sum(), 1)
            d["f1"] = 2 * p_d * r_d / max(p_d + r_d, 1e-12)
        if r.get("coef") is not None:
            err = np.asarray(r["coef"][1:], float) - np.asarray(r["beta_true"][1:], float)
            d["rmse"] = float(np.sqrt(np.mean(err ** 2)))
        out[r["i"]] = d
    return out


def summary(values) -> dict[str, float] | None:
    """mean / quartiles / p90 of a per-dataset vector (None when empty)."""
    import numpy as np

    v = np.asarray([x for x in values if x == x], float)
    if v.size == 0:
        return None
    q = np.percentile(v, [25, 50, 75, 90])
    return {"n": int(v.size), "mean": float(v.mean()), "q25": float(q[0]), "q50": float(q[1]),
            "q75": float(q[2]), "p90": float(q[3])}


def paired(model: dict[str, dict], other: dict[str, dict], metric: str,
           seed: int = 0, boots: int = 2000) -> dict[str, float] | None:
    """Model minus other on the datasets both score: mean difference with a
    seeded bootstrap 95 % interval, the share of datasets the model wins
    (ties excluded from the numerator, counted in n), and the Wilcoxon
    signed-rank p (ties dropped). None when fewer than 10 datasets pair."""
    import numpy as np
    from scipy.stats import wilcoxon

    keys = [k for k in model if k in other and metric in model[k] and metric in other[k]]
    if len(keys) < 10:
        return None
    a = np.asarray([model[k][metric] for k in keys], float)
    b = np.asarray([other[k][metric] for k in keys], float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(boots, d.size))
    means = d[idx].mean(axis=1)
    higher_better = metric != "rmse"
    wins = (d > 0) if higher_better else (d < 0)
    nz = d[d != 0]
    p = float(wilcoxon(nz).pvalue) if nz.size >= 10 else float("nan")
    return {"n": int(d.size), "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
            "ci_lo": float(np.percentile(means, 2.5)), "ci_hi": float(np.percentile(means, 97.5)),
            "win_share": float(wins.mean()), "tie_share": float((d == 0).mean()),
            "wilcoxon_p": p}


def distributions(by_method: dict[str, list[dict]], model_row: str,
                  metrics_: tuple[str, ...] = ("auc", "f1", "rmse")) -> dict:
    """Per method: summaries of the per-dataset metrics and of fit time;
    per method other than the model row: paired comparisons with it."""
    scores = {m: dataset_scores(rows) for m, rows in by_method.items()}
    out: dict = {"summary": {}, "fit_time": {}, "paired": {}}
    for m, sc in scores.items():
        out["summary"][m] = {k: summary(d[k] for d in sc.values() if k in d) for k in metrics_}
        out["fit_time"][m] = summary(d["fit_seconds"] for d in sc.values())
    if model_row in scores:
        for m, sc in scores.items():
            if m == model_row:
                continue
            out["paired"][m] = {k: paired(scores[model_row], sc, k) for k in metrics_}
            # fit time on the common datasets: ratio of medians, model / other
            keys = [k for k in scores[model_row] if k in sc]
            if keys:
                import numpy as np

                a = np.median([scores[model_row][k]["fit_seconds"] for k in keys])
                b = np.median([sc[k]["fit_seconds"] for k in keys])
                out["paired"][m]["fit_time"] = {
                    "n": len(keys), "model_median": float(a), "other_median": float(b),
                    "ratio_other_over_model": float(b / a) if a else None}
    return out


def sampler_summary(rows: list[dict], results_dir: Path, method: str = "mc3") -> dict | None:
    """Convergence facts of the sampler rows (rhat_max, ess_min, converged are
    stored in the npz beside the fit fields): share converged, rhat / ESS
    quantiles, chain steps."""
    import numpy as np

    d = Path(results_dir) / method
    if not rows or not d.is_dir():
        return None
    conv, rhat, ess, steps = [], [], [], []
    for r in rows:
        z = np.load(d / f"{r['i']}.npz", allow_pickle=False)
        if "converged" in z.files:
            conv.append(bool(z["converged"]))
        if "rhat_max" in z.files:
            rhat.append(float(z["rhat_max"]))
        if "ess_min" in z.files:
            ess.append(float(z["ess_min"]))
        if "n_steps" in z.files:
            steps.append(int(z["n_steps"]))
    return {"n": len(rows), "converged_share": float(np.mean(conv)) if conv else None,
            "rhat_max_median": float(np.median(rhat)) if rhat else None,
            "rhat_max_p90": float(np.percentile(rhat, 90)) if rhat else None,
            "ess_min_median": float(np.median(ess)) if ess else None,
            "steps_median": float(np.median(steps)) if steps else None}


def design_row_units(designs_csv: Path) -> dict[str, str]:
    """{design id: row_unit} from the pool's design table (empty if absent)."""
    import csv

    if not Path(designs_csv).is_file():
        return {}
    return {Path(r["clean_file"]).stem: r["row_unit"] for r in csv.DictReader(open(designs_csv))}


def row_unit_slice(by_method: dict[str, list[dict]], model_row: str,
                   unit: dict[str, str]) -> dict[str, dict]:
    """Per row unit: datasets, designs, each method's mean AUC / F1 / RMSE and the
    paired mean differences (model minus method) on that subset."""
    scores = {m: dataset_scores(rows) for m, rows in by_method.items()}
    keys_by_unit: dict[str, list[str]] = {}
    designs_by_unit: dict[str, set[str]] = {}
    for r in by_method.get(model_row, []):
        u = unit.get(r["meta"].get("design", ""), "unknown")
        keys_by_unit.setdefault(u, []).append(r["i"])
        designs_by_unit.setdefault(u, set()).add(r["meta"]["design"])
    out: dict[str, dict] = {}
    for u, keys in keys_by_unit.items():
        block = {"n": len(keys), "designs": len(designs_by_unit[u]), "mean": {}, "diff": {}}
        for m, sc in scores.items():
            block["mean"][m] = {k: summary(sc[i][k] for i in keys if i in sc and k in sc[i])
                                for k in ("auc", "f1", "rmse")}
            if m != model_row:
                ref = {i: scores[model_row][i] for i in keys if i in scores[model_row]}
                block["diff"][m] = {k: paired(ref, sc, k) for k in ("auc", "f1", "rmse")}
        out[u] = block
    return out


def interaction_counts(rows: list[dict]) -> dict[str, Any] | None:
    """How many interactions are active in the truth of a panel's scored datasets
    (one method's rows; the truth is the same for every method): median, mean,
    quartiles, maximum and the share of datasets with none."""
    if not rows:
        return None
    k = np.asarray([int(np.asarray(r["gamma_true"], bool)[r["p"]:].sum()) for r in rows])
    return {"n": int(k.size), "median": float(np.median(k)), "mean": float(k.mean()),
            "q1": float(np.quantile(k, 0.25)), "q3": float(np.quantile(k, 0.75)),
            "max": int(k.max()), "share_none": float((k == 0).mean())}
