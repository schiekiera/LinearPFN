"""The misspecification panel (bench.knobs; stage 16: the main benchmark's own
datasets with one stage of the outcome model replaced, the main rows as
control): rows of the control panel and the three one-knob panels -> metric
tables, each knob paired by dataset with its control.

Reads the knob panels' and the control's fits (stage 17) with
`pipeline.ip`'s loaders and metric definitions, so a number here is the same
quantity the main benchmark reports, plus the PIP calibration (ECE over every
non-intercept slot, `validate._murphy`'s binning) for the methods that return
probabilities. A knob row and its control share the design, the active main
effects, c and sigma2 (the knob applied to the control's own draw), so the
comparison is PAIRED: per-dataset knob minus control
with a seeded bootstrap interval and a Wilcoxon test, as `ip.paired` does
for method-vs-method on one panel.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from pipeline import ip
from pipeline.scores import metrics

_PANEL = re.compile(r"^ip[0-9a-z]{2}_")


def ece(rows: list[dict], n_bins: int = 15) -> dict[str, Any] | None:
    """Expected calibration error of the returned probabilities over every
    non-intercept slot of the rows (pooled slots, `validate._murphy` bins);
    None for methods without probabilities. Also the mains / interactions
    split, since the two are never pooled elsewhere."""
    if not rows or any(r.get("prob") is None for r in rows):
        return None
    probs = np.concatenate([np.asarray(r["prob"], float) for r in rows])
    truth = np.concatenate([r["gamma_true"].astype(float) for r in rows])
    is_main = np.concatenate([np.r_[np.ones(r["p"], bool),
                                    np.zeros(r["gamma_true"].size - r["p"], bool)] for r in rows])
    if probs.size != truth.size:
        return None

    def one(pr, tr):
        if pr.size == 0:
            return float("nan")
        idx = np.clip(np.digitize(pr, np.linspace(0.0, 1.0, n_bins + 1)) - 1, 0, n_bins - 1)
        e = 0.0
        for b in range(n_bins):
            mask = idx == b
            if mask.any():
                e += float(mask.mean()) * abs(float(pr[mask].mean()) - float(tr[mask].mean()))
        return e

    return {"ece": one(probs, truth), "ece_mains": one(probs[is_main], truth[is_main]),
            "ece_ints": one(probs[~is_main], truth[~is_main]), "slots": int(probs.size)}


def with_ece(rows: list[dict]) -> dict[str, Any]:
    met = metrics(rows)
    e = ece(rows)
    met["ece"] = e["ece"] if e else None
    return met


def index_of(key: str) -> str | None:
    """The dataset identity shared across panels: the key without its
    `ip<panel>_` prefix (design + index, e.g. `ipxc_<design>_f0g0_17` and
    `ipmx_<design>_f0g0_17` -> `<design>_f0g0_17`)."""
    m = _PANEL.match(key)
    return key[m.end():] if m else None


def by_index(rows: list[dict]) -> dict[str, dict]:
    """{index: per-dataset scores} (ip.dataset_scores keyed by the shared index)."""
    scores = ip.dataset_scores(rows)
    return {index_of(k): v for k, v in scores.items() if index_of(k) is not None}


def paired_control(knob_rows: list[dict], control_rows: list[dict],
                   metrics_: tuple[str, ...] = ("auc", "f1", "rmse"),
                   with_ece: bool = True) -> dict[str, Any]:
    """Knob minus control on the datasets both score, paired by index: mean
    difference, bootstrap 95 % interval, win share (knob better), Wilcoxon p;
    plus the pooled ECE on each side and its difference (no interval: ECE is
    a pooled-slot quantity)."""
    a, b = by_index(knob_rows), by_index(control_rows)
    out: dict[str, Any] = {k: ip.paired(a, b, k) for k in metrics_}
    ea, eb = (ece(knob_rows), ece(control_rows)) if with_ece else (None, None)
    if ea and eb:
        out["ece"] = {"knob": ea["ece"], "control": eb["ece"], "diff": ea["ece"] - eb["ece"],
                      "knob_mains": ea["ece_mains"], "control_mains": eb["ece_mains"],
                      "knob_ints": ea["ece_ints"], "control_ints": eb["ece_ints"]}
    else:
        out["ece"] = None
    return out


def restrict(by_method: dict[str, list[dict]], keys: set[str]) -> dict[str, list[dict]]:
    return {m: [r for r in rows if r["i"] in keys] for m, rows in by_method.items()}


def exact_subset(by_method: dict[str, list[dict]], exact_row: str = "exact") -> dict[str, Any]:
    """Every method on the datasets the exact posterior scored (p <= 5 by
    construction): pooled metrics + ECE."""
    if exact_row not in by_method or not by_method[exact_row]:
        return {}
    keys = {r["i"] for r in by_method[exact_row]}
    return {m: with_ece(rows) for m, rows in restrict(by_method, keys).items() if rows}


def null_datasets(rows: list[dict]) -> dict[str, Any] | None:
    """Behaviour under the global null (no true active effect): the share of
    those datasets on which the method selects nothing and the mean number of
    false selections."""
    nulls = [r for r in rows if not r["gamma_true"].any()]
    if not nulls:
        return None
    sel = np.asarray([int(r["gamma_hat"].sum()) for r in nulls])
    return {"n": len(nulls), "empty_share": float((sel == 0).mean()),
            "mean_false_selections": float(sel.mean())}
