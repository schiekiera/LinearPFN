"""Validation harness for LinearPFN: metrics, gates, markdown report.

Evaluates a trained checkpoint of the amortized spike-and-slab regression
model against the exact enumeration reference (p <= 5) and against OLS /
oracle baselines (larger p), across strata of context size n, dimension p,
and true-R2 terciles.

The predictive gate is RELATIVE: fraction of headroom closed,
(marginal_nll - model_nll) / (marginal_nll - exact_nll), evaluated PER
STRATUM — absolute nats do not transfer across strata because headroom
shrinks where the problem is easy. Strata whose headroom is below
`HEADROOM_FLAG_NATS` are flagged and excluded from the hard-gate decision
(a percentage of a near-zero denominator is noise), but their raw numbers
are still reported.

Hard gates (the criteria; for a full model — smoke runs are informational,
controlled by the config's validate.hard_gates):
- headroom closed >= 0.95 in every unflagged exact-referenced stratum;
- pooled coefficient recovery r >= 0.99 (exact panel). Scored on the
  coefficient head's posterior mean E[beta | data] when the checkpoint has
  a coefficient head (the head is what the benchmark reports, so the gate
  validates the estimate users get), else on the probe. The other read-out
  is reported beside it, never gated; `pooled.coef_source` names the gated
  one;
- empirical coverage of central 50/80/90% predictive intervals within
  +/- 3 percentage points of nominal, pooled;
- selection AUC within 0.01 of the exact posterior's AUC (paired: both
  scored against the simulated truth on identical datasets), worst subset
  of {pooled, mains, interactions};
- selection ECE within 0.01 of the exact posterior's measured ECE (paired,
  worst subset);
- selection resolution ratio (model Murphy resolution / exact posterior's)
  >= 0.90 pooled and >= 0.85 per subset.

Why the selection gates take this form:
- PIP correlation with the exact posterior is not a gate: pooled
  correlation is dominated by the mass of near-zero PIPs the head gets
  trivially right. Restricting to effects with exact PIP > 0.1 DROPS r
  while RMSE rises, so the pooled figure overstates agreement on
  decision-relevant effects. r is still REPORTED, full and restricted,
  with no threshold.
- Absolute ECE thresholds are not attainable by the reference itself:
  binned ECE carries a finite-sample bias, so even the EXACT posterior's
  measured ECE is well above 0 at small panels, and an absolute bound
  penalizes panel size, not miscalibration. Hence the paired form: model
  ECE may exceed the exact posterior's measured ECE on the identical
  datasets by at most 0.01.
- Margins are noise-floor multiples at a panel size of 100
  datasets/cell: paired AUC-gap SE ~0.001-0.003 and resolution-ratio SE
  ~0.02-0.04 (interactions, the noisiest), so 0.01 AUC ~ 3-10 SE and the
  0.10/0.15 ratio margins ~ 3-4 SE — only real deficits fail the gates.
- Resolution alone is gameable (extreme PIPs raise resolution while
  reliability degrades), which is why the ECE gate accompanies it; AUC
  covers ranking. Together: ranking + calibration + sharpness, each
  measured against what the exact posterior itself achieves.

The surrogate-R2 of the probe ridge fit is tracked as a first-class
misspecification diagnostic: its distribution (and 5th percentile) is
reported per stratum — this is where the fitted surface leaving the
linear-plus-interaction class shows up.

Run: python -m linearpfn.validate --config configs/smoke.yaml --ckpt <path>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr, ndtri

from linearpfn._refcache import ReferenceCache, cache_components
from linearpfn._valreport import write_report
from linearpfn.mc3 import fit_mc3
from linearpfn.model import BarDistribution, LinearPFNModel
from linearpfn.prior import (
    PriorConfig,
    XMeta,
    build_design,
    interaction_pairs,
    n_effects,
    sample_dataset_with_meta,
)
from linearpfn.probe import extract_coefficients
from linearpfn.reference import fit_exact, max_enum_p
from linearpfn.train import (
    _binary_auc,
    build_model,
    load_config,
    make_prior_config,
    resolve_device,
)

__all__ = [
    "GATE_THRESHOLDS",
    "check_panels",
    "check_validate_config",
    "load_surrogate_baseline",
    "run_validation",
    "surrogate_quantile",
]

HEADROOM_FLAG_NATS = 0.02
# The six hard-gate thresholds (docstring above), single-sourced here so the
# paper pipeline (stages 30 / 80 / 99) reads the same numbers the gates use.
GATE_THRESHOLDS = {
    "headroom_min": 0.95,  # headroom closed, every unflagged exact stratum
    "coef_r_min": 0.99,  # pooled coefficient recovery r, exact panel
    "coverage_tol": 0.03,  # |coverage - nominal| at every COVERAGE_LEVELS level
    "auc_gap": 0.01,  # model AUC may fall this far below the exact posterior's
    "ece_gap": 0.01,  # model ECE may exceed the exact posterior's by this much
    "res_pooled": 0.90,  # resolution ratio vs exact, pooled
    "res_subset": 0.85,  # resolution ratio vs exact, every subset
}
COVERAGE_LEVELS = ((0.25, 0.75, 50), (0.10, 0.90, 80), (0.05, 0.95, 90))
MAX_PROBED_LARGE_P = 10  # probing cost grows with d; subsample datasets for p > 5
# MC3 convergence criteria (cell-level, across all its datasets). Unconverged
# cells are reported UNCONVERGED with reference metrics suppressed: an
# unconverged reference is biased in the model's favour (see _valreport).
MC3_RHAT_MAX = 1.1
MC3_ESS_MIN = 200.0


def exact_reach(val_cfg: dict, prior_cfg: PriorConfig) -> int:
    """Largest p scored against the exact reference in this run: the prior's
    enumeration reach, optionally capped by `validate.exact_p_max`. The cap
    exists for the mains-only mixture panel: mains-only
    enumerates to p = 20, but under the mixture's dense half the p = 20
    posterior needs more than the 5M-component pruning cap to hold
    1 - 5e-4 of its mass, so the fit raises there; p = 10 is exact and
    cheap, p >= 20 runs as model-only large-p sweeps, as p >= 8 does on the
    interaction panels. Without the knob the reach is the prior's own."""
    reach = max_enum_p(prior_cfg)
    cap = val_cfg.get("exact_p_max")
    return reach if cap is None else min(reach, int(cap))


def check_validate_config(val_cfg: dict, prior_cfg: PriorConfig | None = None) -> None:
    """Fail loudly at load time on config mistakes that would otherwise
    produce a report silently missing sections (runs are expensive)."""
    p_values = list(val_cfg["p_values"])
    n_values = list(val_cfg["n_values"])
    mc3_ps = list(val_cfg.get("mc3_p_values", []) or [])
    if prior_cfg is not None and not prior_cfg.include_interactions and mc3_ps:
        raise ValueError(
            f"mc3_p_values {mc3_ps} with the mains-only prior: MC3 targets the "
            "interactions prior; mains-only enumerates exactly (drop mc3_p_values)"
        )
    if (
        prior_cfg is not None
        and mc3_ps
        and (
            tuple(prior_cfg.heredity_weights) != (1.0, 0.0, 0.0)
            or not prior_cfg.reject_empty
            or prior_cfg.random_c
            or prior_cfg.random_rho_int
        )
    ):
        raise ValueError(
            f"mc3_p_values {mc3_ps} with a random-scale or mixture prior: MC3's kernel "
            "enforces strong "
            "heredity and non-empty sets at a fixed (c, rho), so it cannot target "
            "the mixture / random-scale prior (drop mc3_p_values)"
        )
    missing = [p for p in mc3_ps if p not in p_values]
    if missing:
        raise ValueError(
            f"mc3_p_values {missing} not in p_values {p_values}: "
            "those MC3 panels would silently never run"
        )
    stray = [p for p in (val_cfg.get("mc3_steps_by_p", {}) or {}) if p not in mc3_ps]
    if stray:
        raise ValueError(
            f"mc3_steps_by_p keys {stray} not in mc3_p_values: "
            "those chain-length overrides would silently never apply"
        )
    by_p = val_cfg.get("datasets_per_cell_by_p", {}) or {}
    stray_d = [p for p in by_p if p not in p_values]
    if stray_d:
        raise ValueError(
            f"datasets_per_cell_by_p keys {stray_d} not in p_values {p_values}: "
            "those cell-size overrides would silently never apply"
        )
    if any(int(v) < 1 for v in by_p.values()):
        raise ValueError("datasets_per_cell_by_p values must be >= 1")
    cap = val_cfg.get("exact_p_max")
    if cap is not None:
        if int(cap) < 2:
            raise ValueError(f"exact_p_max {cap} must be >= 2")
        if prior_cfg is not None and int(cap) > max_enum_p(prior_cfg):
            raise ValueError(
                f"exact_p_max {cap} exceeds the prior's enumeration reach "
                f"{max_enum_p(prior_cfg)}: the knob would be silently inert"
            )
    # Family envelope [2, 30], EXTENDED by a wider prior (a mains-only prior
    # may train to p = 100, beyond a fixed bound of 30). Cells above a
    # config's own p_max stay allowed: a p <= 3 smoke model may be validated
    # at p = 6 on purpose.
    p_lo, p_hi = 2, max(30, prior_cfg.p_max if prior_cfg is not None else 30)
    bad_p = [p for p in p_values if not p_lo <= p <= p_hi]
    if bad_p:
        raise ValueError(f"p_values {bad_p} outside the prior envelope [{p_lo}, {p_hi}]")
    bad_n = [n for n in n_values if n < 10]
    if bad_n:
        raise ValueError(f"n_values {bad_n} too small for a usable context (need >= 10)")
    if int(val_cfg["n_query"]) < 1:
        raise ValueError("n_query must be >= 1")
    if int(val_cfg["datasets_per_cell"]) < 1:
        raise ValueError("datasets_per_cell must be >= 1")


def check_panels(results: dict, val_cfg: dict, exact_p_max: int = 5) -> None:
    """Raise if any requested panel produced no rows (the report is written
    first so the partial output is available for debugging). `exact_p_max`
    is the enumeration reach of the prior variant (reference.max_enum_p)."""
    cells = results["cells"]
    empty: list[str] = []
    if any(p <= exact_p_max for p in val_cfg["p_values"]):
        if not any("nll_exact" in c for c in cells if c["p"] <= exact_p_max):
            empty.append(f"exact headroom (p<={exact_p_max})")
        if not any(("coef_head" in r or "coef_probe" in r) and "coef_exact" in r
                   for r in results["records"]):
            empty.append(f"coefficient recovery (p<={exact_p_max})")
    for p in val_cfg.get("mc3_p_values", []) or []:
        ok = [c for c in cells
              if c["p"] == p and ("nll_exact" in c or c.get("unconverged"))]
        if not ok:
            empty.append(f"MC3 reference (p={p})")
    per_p = results["pooled"]["sel"].get("per_p", {})
    for p in val_cfg["p_values"]:
        if p not in per_p:
            empty.append(f"selection per-p (p={p})")
    if empty:
        raise RuntimeError(
            "empty panels — report written but these sections have no rows: "
            + "; ".join(empty)
        )


_DISCRETE_TYPES = {"likert", "binary", "count"}


def _column_composition(column_types: np.ndarray) -> str:
    """Prior stratum label: all_gaussian / all_discrete / mixed."""
    kinds = {str(t) for t in column_types}
    if kinds == {"gaussian"}:
        return "all_gaussian"
    if kinds <= _DISCRETE_TYPES:
        return "all_discrete"
    return "mixed"


def _corr_regime(meta: XMeta, X: np.ndarray) -> str:
    """Prior stratum label from the X-structure ground truth and realized X:
    generic (no factor block) / factor / has_duplicates (max |r| > 0.9)."""
    if meta.p_fac == 0:
        return "generic"
    p = X.shape[1]
    if p < 2:
        return "factor"
    R = np.corrcoef(X, rowvar=False)
    max_r = float(np.abs(R[np.triu_indices(p, k=1)]).max())
    return "has_duplicates" if max_r > 0.9 else "factor"


def c_tercile_edges(cfg: PriorConfig) -> tuple[float, float] | None:
    """Analytic terciles of the truncated-lognormal c prior (None when c is
    fixed). Fixed analytic edges — not per-run empirical ones — so strata
    keep the same meaning across runs and panel sizes."""
    if not cfg.random_c:
        return None
    a = ndtr((math.log(cfg.c_min) - cfg.c_logmean) / cfg.c_logsd)
    b = ndtr((math.log(cfg.c_max) - cfg.c_logmean) / cfg.c_logsd)
    e1 = math.exp(cfg.c_logmean + cfg.c_logsd * ndtri(a + (b - a) / 3.0))
    e2 = math.exp(cfg.c_logmean + cfg.c_logsd * ndtri(a + 2.0 * (b - a) / 3.0))
    return float(e1), float(e2)


def _prior_strata(records: list[dict], prior_cfg: PriorConfig) -> list[dict]:
    """Headroom-closed per prior stratum (REPORTED, not gated: the hard
    gates are per (n, p); a pooled number can hide a weak cell)."""
    edges = c_tercile_edges(prior_cfg)

    def c_label(r: dict) -> str:
        if edges is None:
            return "fixed"
        if r["c_value"] < edges[0]:
            return "low"
        return "mid" if r["c_value"] < edges[1] else "high"

    labelers = {
        "heredity": lambda r: r["heredity_mode"],
        "columns": lambda r: r["col_comp"],
        "correlation": lambda r: r["corr_regime"],
        "c_tercile": c_label,
    }
    rows: list[dict] = []
    for kind, fn in labelers.items():
        for label in sorted({fn(r) for r in records}):
            sub = [r for r in records if fn(r) == label]
            row: dict = {
                "kind": kind,
                "label": label,
                "count": len(sub),
                "nll_model": float(np.mean([r["nll_model"] for r in sub])),
                "nll_marginal": float(np.mean([r["nll_marginal"] for r in sub])),
            }
            exact = [r for r in sub if r.get("reference") == "exact"]
            if exact:
                marg = float(np.mean([r["nll_marginal"] for r in exact]))
                modl = float(np.mean([r["nll_model"] for r in exact]))
                ref = float(np.mean([r["nll_exact"] for r in exact]))
                row["count_exact"] = len(exact)
                row["headroom"] = marg - ref
                if marg - ref >= HEADROOM_FLAG_NATS:
                    row["gap_closed"] = (marg - modl) / (marg - ref)
            rows.append(row)
    return rows


def _model_mean_fn(model: LinearPFNModel, bar: BarDistribution, X_ctx, y_ctx, device):
    Xc = torch.from_numpy(X_ctx).float().to(device)
    yc = torch.from_numpy(y_ctx).float().unsqueeze(0).to(device)

    def mean_fn(probes: np.ndarray) -> np.ndarray:
        X_all = torch.cat([Xc, torch.from_numpy(probes).float().to(device)]).unsqueeze(0)
        with torch.no_grad():
            logits = model(X_all, yc, Xc.shape[0])
        return bar.mean(logits.float())[0].cpu().numpy()

    return mean_fn


def _gaussian_nll(y: np.ndarray, mean: np.ndarray | float, var: float) -> float:
    var = max(var, 1e-9)
    return float(np.mean(0.5 * math.log(2 * math.pi * var) + (y - mean) ** 2 / (2 * var)))


def _ols_nll(Z_ctx: np.ndarray, y_ctx: np.ndarray, Z_q: np.ndarray, y_q: np.ndarray) -> float:
    coef, *_ = np.linalg.lstsq(Z_ctx, y_ctx, rcond=None)
    sigma2 = float(np.mean((y_ctx - Z_ctx @ coef) ** 2))
    return _gaussian_nll(y_q, Z_q @ coef, sigma2)


def _tick(timers: dict[str, float] | None, key: str, t0: float) -> None:
    if timers is not None:
        timers[key] = timers.get(key, 0.0) + (time.monotonic() - t0)


def _evaluate_cell(
    model: LinearPFNModel,
    bar: BarDistribution,
    prior_cfg: PriorConfig,
    n_ctx: int,
    p: int,
    val_cfg: dict,
    device: torch.device,
    ref_cache: ReferenceCache | None = None,
    timers: dict[str, float] | None = None,
) -> list[dict]:
    """Per-dataset records for one (n, p) stratum."""
    n_query = val_cfg["n_query"]
    records: list[dict] = []
    rng = np.random.default_rng([val_cfg["seed"], n_ctx, p])
    # per-p cell-size override (e.g. mains-only p=20: ~1 min/dataset for the
    # 2^20-model exact fit); the dataset stream is a deterministic prefix, so
    # reducing the count truncates rather than reshuffles
    by_p = val_cfg.get("datasets_per_cell_by_p", {}) or {}
    n_datasets = int(by_p.get(p, val_cfg["datasets_per_cell"]))
    for i in range(n_datasets):
        ds, meta = sample_dataset_with_meta(rng, prior_cfg, n=n_ctx + n_query, p=p)
        X_ctx, X_q = ds.X[:n_ctx], ds.X[n_ctx:]
        y_ctx, y_q = ds.y[:n_ctx], ds.y[n_ctx:]
        Z_all = build_design(ds.X, prior_cfg)
        signal = Z_all @ ds.beta
        rec: dict = {
            "n": n_ctx,
            "p": p,
            "true_r2": float(signal.var() / ds.y.var()),
            "nll_marginal": _gaussian_nll(y_q, float(y_ctx.mean()), float(y_ctx.var())),
            # prior stratum labels (all degenerate under the fixed-scale configuration)
            "heredity_mode": ds.heredity_mode,
            "c_value": float(ds.c),
            "col_comp": _column_composition(meta.column_types),
            "corr_regime": _corr_regime(meta, ds.X),
        }
        X_t = torch.from_numpy(ds.X).float().unsqueeze(0).to(device)
        y_ctx_t = torch.from_numpy(y_ctx).float().unsqueeze(0).to(device)
        t_fwd = time.monotonic()
        with torch.no_grad():
            logits, sel, coef_out = model.forward_with_heads(X_t, y_ctx_t, n_ctx)
            logits = logits.float()
            rec["sel_probs"] = torch.sigmoid(sel.float())[0].cpu().numpy()
            if coef_out is not None:
                # E[beta | data] in gamma[1:] order (no intercept), the vector
                # the benchmark reports; scored against coef_exact[1:]
                rec["coef_head"] = model.coefficient_posterior_mean(
                    sel[0], coef_out[0]).cpu().numpy()
            rec["gamma_true"] = ds.gamma[1:].copy()
            rec["nll_model"] = float(bar.nll(logits, torch.from_numpy(y_q).unsqueeze(0)
                                             .to(device)).mean())
            levels = [lv for lo, hi, _ in COVERAGE_LEVELS for lv in (lo, hi)]
            qs = bar.quantile(logits, torch.tensor(levels, device=device))[0].cpu().numpy()
        _tick(timers, "forward", t_fwd)
        for j, (_lo, _hi, name) in enumerate(COVERAGE_LEVELS):
            lo_q, hi_q = qs[:, 2 * j], qs[:, 2 * j + 1]
            rec[f"cover{name}"] = float(np.mean((y_q >= lo_q) & (y_q <= hi_q)))
        d = n_effects(p, prior_cfg)
        Z_ctx, Z_q = Z_all[:n_ctx], Z_all[n_ctx:]
        if n_ctx > d:
            rec["nll_ols"] = _ols_nll(Z_ctx, y_ctx, Z_q, y_q)
        active = ds.gamma
        # near-singular oracle fits (active count ~ n_ctx) produce absurd NLLs
        # from a collapsed MLE variance; require 2x parameters of headroom
        if n_ctx > 2 * int(active.sum()):
            rec["nll_oracle"] = _ols_nll(Z_ctx[:, active], y_ctx, Z_q[:, active], y_q)
        use_exact = p <= exact_reach(val_cfg, prior_cfg)
        mc3_cap = int(val_cfg.get("mc3_datasets_per_cell", val_cfg["datasets_per_cell"]))
        use_mc3 = (
            not use_exact and p in val_cfg.get("mc3_p_values", []) and i < mc3_cap
        )
        res = None
        if use_exact or use_mc3 or i < MAX_PROBED_LARGE_P:
            t_probe = time.monotonic()
            res = extract_coefficients(
                np.random.default_rng([val_cfg["seed"], n_ctx, p, 7000 + i]),
                X_ctx,
                _model_mean_fn(model, bar, X_ctx, y_ctx, device),
                ridge_lambda=val_cfg["ridge_lambda"],
                jitter_sd=val_cfg["jitter_sd"],
                cfg=prior_cfg,
            )
            _tick(timers, "probe", t_probe)
            rec["surrogate_r2"] = res.r2
        if use_exact or use_mc3:
            assert res is not None
            cached = ref_cache.get(n_ctx, p, i) if use_exact and ref_cache is not None else None
            if cached is not None:
                rec["reference"] = "exact"
                rec["coef_probe"] = res.coef
                rec.update(cached)
                records.append(rec)
                continue
            t_ref = time.monotonic()
            if use_exact:
                post = fit_exact(
                    X_ctx.astype(float), y_ctx.astype(float), prior_cfg,
                    standardization_tol=1.0,
                )
            else:  # sampled reference — approximate, labelled as such downstream
                # chain length is p-dependent (mc3_steps_by_p overrides), per the
                # measured scaling: 16k converges p <= 10; p = 15 needs ~128k
                # with multi-flip proposals
                by_p = val_cfg.get("mc3_steps_by_p", {}) or {}
                mc3_steps = int(by_p.get(p, val_cfg.get("mc3_steps", 4000)))
                mc3_res = fit_mc3(
                    X_ctx.astype(float), y_ctx.astype(float), prior_cfg,
                    n_steps=mc3_steps,
                    burn_in=mc3_steps // 4,
                    seed=val_cfg["seed"] * 1000003 + n_ctx * 1009 + p * 101 + i,
                    standardization_tol=1.0,
                    multi_flip=float(val_cfg.get("mc3_multi_flip", 0.0)),
                )
                post = mc3_res.posterior
                rec["mc3_rhat"] = mc3_res.diagnostics.rhat_max
                rec["mc3_ess"] = mc3_res.diagnostics.ess_min
            rec["reference"] = "exact" if use_exact else "mc3"
            rec["comp_mass"] = float(getattr(post, "comp_mass", 1.0))
            rec["nll_exact"] = float(-post.predictive_logpdf(X_q, y_q).mean())
            rec["coef_probe"] = res.coef
            rec["coef_exact"] = post.coef_mean()
            rec["exact_pips"] = post.pip()[1:]
            _tick(timers, "reference", t_ref)
            if use_exact and ref_cache is not None:
                ref_cache.put(n_ctx, p, i, rec["nll_exact"], rec["comp_mass"],
                              rec["coef_exact"], rec["exact_pips"])
        records.append(rec)
    return records


def _murphy(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """Brier score with the Murphy decomposition, plus ECE, AUC, reliability.

    brier = reliability - resolution + uncertainty (up to binning error).
    ECE is necessary-not-sufficient: a head that emits the base rate for
    every effect has ECE ~ 0 and zero resolution — sharpness lives in the
    resolution term.
    """
    if probs.size == 0:  # e.g. the interactions subset of a mains-only run
        nan = float("nan")
        return {"brier": nan, "reliability": nan, "resolution": nan,
                "uncertainty": nan, "ece": nan, "auc": nan, "rows": []}
    base = float(labels.mean())
    idx = np.clip(np.digitize(probs, np.linspace(0.0, 1.0, n_bins + 1)) - 1, 0, n_bins - 1)
    rel = res = ece = 0.0
    rows: list[tuple[float, float, int]] = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            w = float(mask.mean())
            conf, acc = float(probs[mask].mean()), float(labels[mask].mean())
            rel += w * (conf - acc) ** 2
            res += w * (acc - base) ** 2
            ece += w * abs(conf - acc)
            rows.append((conf, acc, int(mask.sum())))
    return {
        "brier": float(np.mean((probs - labels) ** 2)),
        "reliability": rel,
        "resolution": res,
        "uncertainty": base * (1.0 - base),
        "ece": ece,
        "auc": _binary_auc(probs, labels),
        "rows": rows,
    }


def _is_main_mask(records: list[dict]) -> np.ndarray:
    return np.concatenate([np.arange(r["sel_probs"].size) < r["p"] for r in records])


def _heredity_stats(records: list[dict]) -> dict:
    """Rate of [interaction PIP > 0.5 while BOTH parent PIPs < 0.2], for the
    head everywhere and PAIRED with the exact posterior on identical exact-
    referenced datasets. Under a strong-heredity prior the exact
    posterior satisfies int-PIP <= min(parent PIPs) structurally, so its
    rate is 0 by construction; under the heredity mixture a nonzero
    exact rate is legitimate posterior mass on orphan interactions — the
    comparison is model vs exact, never model vs zero. Scatter points
    (max parent PIP, interaction PIP) are kept for the exact-panel figure."""

    def collect(recs: list[dict], key: str, keep_points: bool) -> tuple[int, int, list, list]:
        viol = total = 0
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for r in recs:
            p = r["p"]
            probs = r[key]
            if probs.size <= p:  # mains-only records carry no interaction slots
                continue
            pairs = np.asarray(interaction_pairs(p), dtype=int).reshape(-1, 2)
            if not len(pairs):
                continue
            parent_max = np.maximum(probs[pairs[:, 0]], probs[pairs[:, 1]])
            int_pip = probs[p:]
            viol += int(((int_pip > 0.5) & (parent_max < 0.2)).sum())
            total += len(pairs)
            if keep_points:
                xs.append(parent_max)
                ys.append(int_pip)
        return viol, total, xs, ys

    small = [r for r in records if r.get("reference") == "exact"]
    v_all, t_all, _, _ = collect(records, "sel_probs", False)
    v_p5, t_p5, hx, hy = collect(small, "sel_probs", True)
    v_ex, t_ex, ex_x, ex_y = collect(small, "exact_pips", True)
    return {
        "head_rate_all": v_all / max(t_all, 1),
        "n_int_all": t_all,
        "head_rate_p5": v_p5 / max(t_p5, 1),
        "exact_rate_p5": v_ex / max(t_ex, 1),
        "n_int_p5": t_p5,
        "points_head": (np.concatenate(hx), np.concatenate(hy)) if hx else None,
        "points_exact": (np.concatenate(ex_x), np.concatenate(ex_y)) if ex_x else None,
    }


def _selection_metrics(records: list[dict]) -> dict:
    """All selection metrics, pooled and split by mains vs interactions.

    The interaction logits are built from mean-pooled per-column streams
    (concat(h_j+h_k, h_j*h_k)); pair-specific information must arrive
    through backbone feature attention. Interactions lagging mains here is
    an architectural signal worth fixing before a long training run.
    """
    probs = np.concatenate([r["sel_probs"] for r in records])
    truth = np.concatenate([r["gamma_true"] for r in records]).astype(float)
    is_main = _is_main_mask(records)
    subset_masks = {
        "pooled": np.ones_like(is_main, dtype=bool),
        "mains": is_main,
        "interactions": ~is_main,
    }
    subsets = {name: _murphy(probs[m], truth[m]) for name, m in subset_masks.items()}
    # the gate panel is strictly against the EXACT reference; mc3-referenced
    # records appear only in the per-p breakdown, labelled approximate
    small = [r for r in records if r.get("reference") == "exact"]
    p5: dict[str, dict] = {}
    sp = se = np.empty(0)
    if small:
        sp = np.concatenate([r["sel_probs"] for r in small])
        se = np.concatenate([r["exact_pips"] for r in small])
        st = np.concatenate([r["gamma_true"] for r in small]).astype(float)
        sm = _is_main_mask(small)
        for name, m in (("pooled", np.ones_like(sm, dtype=bool)), ("mains", sm),
                        ("interactions", ~sm)):
            if not m.any():  # mains-only runs have no interaction effects
                nan = float("nan")
                p5[name] = dict.fromkeys(
                    ("r", "rmse", "res_model", "res_exact", "ece_model",
                     "ece_exact", "auc_model", "auc_exact", "r_high",
                     "rmse_high", "n_high"), nan)
                continue
            mm, me = _murphy(sp[m], st[m]), _murphy(se[m], st[m])
            # descriptive restricted view: effects the exact posterior itself
            # considers non-negligible — pooled r is inflated by the mass of
            # near-zero PIPs the head gets trivially right (measured), so the
            # high-evidence agreement is reported alongside, without a gate
            hi = se[m] > 0.1
            p5[name] = {
                "r": float(np.corrcoef(sp[m], se[m])[0, 1]),
                "rmse": float(np.sqrt(np.mean((sp[m] - se[m]) ** 2))),
                "res_model": mm["resolution"],
                "res_exact": me["resolution"],
                "ece_model": mm["ece"],
                "ece_exact": me["ece"],
                "auc_model": mm["auc"],
                "auc_exact": me["auc"],
                "r_high": (float(np.corrcoef(sp[m][hi], se[m][hi])[0, 1])
                           if hi.sum() >= 10 else float("nan")),
                "rmse_high": (float(np.sqrt(np.mean((sp[m][hi] - se[m][hi]) ** 2)))
                              if hi.any() else float("nan")),
                "n_high": int(hi.sum()),
            }
    per_p: dict[int, dict] = {}
    for p_val in sorted({r["p"] for r in records}):
        recs = [r for r in records if r["p"] == p_val]
        pm = _is_main_mask(recs)
        probs_p = np.concatenate([r["sel_probs"] for r in recs])
        truth_p = np.concatenate([r["gamma_true"] for r in recs]).astype(float)
        row = {
            "auc_mains": _binary_auc(probs_p[pm], truth_p[pm]),
            "auc_int": _binary_auc(probs_p[~pm], truth_p[~pm]),
            "res_int": _murphy(probs_p[~pm], truth_p[~pm])["resolution"],
        }
        srecs = [r for r in recs if "exact_pips" in r]
        if srecs and (~_is_main_mask(srecs)).any():
            sm_p = _is_main_mask(srecs)
            sp_p = np.concatenate([r["sel_probs"] for r in srecs])
            se_p = np.concatenate([r["exact_pips"] for r in srecs])
            st_p = np.concatenate([r["gamma_true"] for r in srecs]).astype(float)
            row["r_int_exact"] = float(np.corrcoef(sp_p[~sm_p], se_p[~sm_p])[0, 1])
            row["ratio_int"] = _murphy(sp_p[~sm_p], st_p[~sm_p])["resolution"] / max(
                _murphy(se_p[~sm_p], st_p[~sm_p])["resolution"], 1e-12
            )
        per_p[p_val] = row
    return {
        "subsets": subsets,
        "p5": p5,
        "per_p": per_p,
        "heredity": _heredity_stats(records),
        "hist_pred_p5": sp,
        "hist_exact_p5": se,
    }


def _cell_summary(cell: list[dict]) -> dict:
    """Aggregate one stratum; headroom-closed uses cell-mean NLLs."""
    out: dict = {"n": cell[0]["n"], "p": cell[0]["p"], "count": len(cell)}
    for key in ("nll_model", "nll_marginal", "nll_exact", "nll_ols", "nll_oracle",
                "cover50", "cover80", "cover90"):
        vals = [r[key] for r in cell if key in r]
        if vals:
            out[key] = float(np.mean(vals))
    r2s = [r["surrogate_r2"] for r in cell if "surrogate_r2" in r]
    if r2s:
        out["surrogate_r2_p5"] = float(np.percentile(r2s, 5))
        out["surrogate_r2_min"] = float(min(r2s))
    masses = [r["comp_mass"] for r in cell if "comp_mass" in r]
    if masses:
        # kept component mass of the quadrature posterior (1.0 = untruncated);
        # diffuse small-n cells can sit at ~1 - 1e-4 (see reference.fit_exact)
        out["comp_mass_min"] = float(min(masses))
    refs = {r["reference"] for r in cell if "reference" in r}
    if refs:
        out["reference"] = refs.pop()
    rhats = [r["mc3_rhat"] for r in cell if "mc3_rhat" in r]
    if rhats:
        out["mc3_rhat_max"] = float(max(rhats))
        out["mc3_ess_min"] = float(min(r["mc3_ess"] for r in cell if "mc3_ess" in r))
        out["unconverged"] = (
            out["mc3_rhat_max"] >= MC3_RHAT_MAX or out["mc3_ess_min"] <= MC3_ESS_MIN
        )
        if out["unconverged"]:
            # suppress reference-derived metrics: they would flatter the model
            for key in ("nll_exact", "headroom", "gap_closed", "flagged"):
                out.pop(key, None)
    if "nll_exact" in out:
        out["headroom"] = out["nll_marginal"] - out["nll_exact"]
        out["flagged"] = out["headroom"] < HEADROOM_FLAG_NATS
        if not out["flagged"]:
            out["gap_closed"] = (out["nll_marginal"] - out["nll_model"]) / out["headroom"]
        elif abs(out["headroom"]) > 1e-12:
            # informational only: with near-zero headroom the ratio is noise,
            # and values > 1 ("beating" the exact reference) are expected —
            # never superiority, which is why flagged cells sit outside the
            # hard gate
            out["gap_closed_info"] = (
                out["nll_marginal"] - out["nll_model"]
            ) / out["headroom"]
    return out


def _pooled_metrics(records: list[dict]) -> dict:
    """Pooled exact-referenced predictive/coef/coverage metrics and
    R2-tercile strata ("p5" keys denote the panel defined by
    reference == exact, which reaches far beyond p = 5 mains-only)."""
    small = [r for r in records if r.get("reference") == "exact"]
    pooled: dict = {}
    if small:  # runs without exact cells (e.g. mc3-only sweeps) skip the panel
        marg = float(np.mean([r["nll_marginal"] for r in small]))
        modl = float(np.mean([r["nll_model"] for r in small]))
        exact = float(np.mean([r["nll_exact"] for r in small]))
        pooled["p5"] = {
            "nll_model": modl, "nll_marginal": marg, "nll_exact": exact,
            "headroom": marg - exact, "gap_closed": (marg - modl) / (marg - exact),
        }
        probed = np.concatenate([r["coef_probe"] for r in small])
        exact_c = np.concatenate([r["coef_exact"] for r in small])
        pooled["coef_r_probe"] = float(np.corrcoef(probed, exact_c)[0, 1])
        pooled["coef_rmse_probe"] = float(np.sqrt(np.mean((probed - exact_c) ** 2)))
        if all("coef_head" in r for r in small):
            # the head omits the intercept, so pair it with coef_exact[1:]
            head = np.concatenate([r["coef_head"] for r in small])
            exact_h = np.concatenate([r["coef_exact"][1:] for r in small])
            pooled["coef_r_head"] = float(np.corrcoef(head, exact_h)[0, 1])
            pooled["coef_rmse_head"] = float(np.sqrt(np.mean((head - exact_h) ** 2)))
            pooled["coef_source"] = "head"
        else:
            pooled["coef_source"] = "probe"
        src = pooled["coef_source"]
        pooled["coef_r"] = pooled[f"coef_r_{src}"]
        pooled["coef_rmse"] = pooled[f"coef_rmse_{src}"]
    else:
        pooled["p5"] = None
        pooled["coef_source"] = "none"
        pooled["coef_r"] = float("nan")
        pooled["coef_rmse"] = float("nan")
    sel = _selection_metrics(records)
    pooled["sel"] = sel
    pooled["pip_r"] = sel["p5"].get("pooled", {}).get("r", float("nan"))
    pooled["pip_rmse"] = sel["p5"].get("pooled", {}).get("rmse", float("nan"))
    pooled["selection_auc"] = sel["subsets"]["pooled"]["auc"]
    pooled["ece"] = sel["subsets"]["pooled"]["ece"]
    pooled["reliability"] = sel["subsets"]["pooled"]["rows"]
    pooled["ece_effects"] = sum(r["gamma_true"].size for r in records)
    pooled["ece_datasets"] = len(records)
    for _, _, name in COVERAGE_LEVELS:
        pooled[f"cover{name}"] = float(np.mean([r[f"cover{name}"] for r in records]))
    edges = np.percentile([r["true_r2"] for r in records], [100 / 3, 200 / 3])
    pooled["r2_edges"] = [float(e) for e in edges]
    terciles = []
    for t, (lo, hi) in enumerate(((-np.inf, edges[0]), (edges[0], edges[1]),
                                  (edges[1], np.inf))):
        sub = [r for r in records if lo <= r["true_r2"] < hi]
        subs = [r for r in sub if r.get("reference") == "exact"]
        row = {
            "tercile": t + 1, "count": len(sub),
            "nll_model": float(np.mean([r["nll_model"] for r in sub])),
            "nll_marginal": float(np.mean([r["nll_marginal"] for r in sub])),
        }
        if subs:
            m, e, mo = (float(np.mean([r[k] for r in subs]))
                        for k in ("nll_marginal", "nll_exact", "nll_model"))
            row["gap_closed_p5"] = (m - mo) / (m - e) if (m - e) > HEADROOM_FLAG_NATS else None
        r2s = [r["surrogate_r2"] for r in sub if "surrogate_r2" in r]
        if r2s:
            row["surrogate_r2_p5"] = float(np.percentile(r2s, 5))
        terciles.append(row)
    pooled["terciles"] = terciles
    return pooled


def _gates(cells: list[dict], pooled: dict) -> list[dict]:
    t = GATE_THRESHOLDS
    small = [c for c in cells if c.get("reference") == "exact"]
    unflagged = [c for c in small if not c.get("flagged", False)]
    gates = [
        {
            "name": f"headroom closed >= {t['headroom_min']:.2f} in every unflagged exact stratum",
            "value": min((c["gap_closed"] for c in unflagged), default=float("nan")),
            "passed": bool(unflagged)
            and all(c["gap_closed"] >= t["headroom_min"] for c in unflagged),
        },
        {
            "name": f"pooled coefficient recovery r >= {t['coef_r_min']:.2f} "
                    f"(exact panel, {pooled['coef_source']})",
            "value": pooled["coef_r"],
            "passed": pooled["coef_r"] >= t["coef_r_min"],
        },
    ]
    cov_ok = all(abs(pooled[f"cover{name}"] - name / 100) <= t["coverage_tol"]
                 for _, _, name in COVERAGE_LEVELS)
    levels = "/".join(str(name) for _, _, name in COVERAGE_LEVELS)
    gates.append({
        "name": f"coverage within +/-{t['coverage_tol'] * 100:g}pp of nominal at {levels}%",
        "value": max(abs(pooled[f"cover{name}"] - name / 100) for _, _, name in COVERAGE_LEVELS),
        "passed": cov_ok,
    })
    # Selection gates are ANCHORED to the exact posterior on identical
    # datasets (paired), per subset. Subsets that are undefined for a run
    # (nan — e.g. interactions under the mains-only prior, or a micro panel
    # with single-class truth) are skipped; an empty exact panel fails.
    p5 = pooled["sel"]["p5"]
    auc_gaps = [d["auc_model"] - d["auc_exact"] for d in p5.values()
                if d and math.isfinite(d["auc_model"] - d["auc_exact"])]
    gates.append({
        "name": f"selection AUC within {t['auc_gap']:.2f} of the exact posterior's "
                "(worst subset, paired)",
        "value": min(auc_gaps, default=float("nan")),
        "passed": bool(auc_gaps) and all(g >= -t["auc_gap"] for g in auc_gaps),
    })
    ece_gaps = [d["ece_model"] - d["ece_exact"] for d in p5.values()
                if d and math.isfinite(d["ece_model"] - d["ece_exact"])]
    gates.append({
        "name": f"selection ECE within {t['ece_gap']:.2f} of the exact posterior's "
                "(worst subset, paired)",
        "value": max(ece_gaps, default=float("nan")),
        "passed": bool(ece_gaps) and all(g <= t["ece_gap"] for g in ece_gaps),
    })
    ratios = {name: d["res_model"] / max(d["res_exact"], 1e-12)
              for name, d in p5.items() if d and math.isfinite(d["res_model"])}
    res_ok = (
        "pooled" in ratios and ratios["pooled"] >= t["res_pooled"]
        and all(v >= t["res_subset"] for k, v in ratios.items() if k != "pooled")
    )
    gates.append({
        "name": f"selection resolution ratio vs exact: pooled >= {t['res_pooled']:.2f}, "
                f"subsets >= {t['res_subset']:.2f}",
        "value": min(ratios.values(), default=float("nan")),
        "passed": res_ok,
    })
    return gates


def load_surrogate_baseline(path: str | Path) -> dict:
    """Load a persisted in-prior surrogate-R2 calibration table."""
    return json.loads(Path(path).read_text())


def surrogate_quantile(value: float, n: int, p: int, baseline: dict) -> float:
    """Quantile of a surrogate-R2 value within the in-prior distribution of
    the matched (n, p) cell (nearest cell by (log2 n, p/5) distance).

    Report real-data diagnostics as this quantile, never as a raw threshold:
    the in-prior level is strongly stratum-dependent (measured 5th
    percentiles range 0.93-0.999 across cells) and capacity-dependent (the
    table must be regenerated whenever the model is retrained).
    """
    cells = [c for c in baseline["cells"] if c["values"]]
    best = min(
        cells,
        key=lambda c: (math.log2(c["n"]) - math.log2(n)) ** 2 + ((c["p"] - p) / 5.0) ** 2,
    )
    vals = np.sort(np.asarray(best["values"], dtype=float))
    return float(np.searchsorted(vals, value, side="right") / vals.size)


def _json_default(x):
    """numpy scalars/arrays -> JSON; anything else -> its repr."""
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "item"):
        return x.item()
    return repr(x)


def run_validation(
    config: dict,
    ckpt_path: str,
    out_dir: str,
    ref_cache_dir: str | None = None,
    rebuild_ref_cache: bool = False,
) -> dict:
    """Evaluate the checkpoint across all strata; write report; return results.

    `ref_cache_dir` enables the keyed exact-reference cache (see
    linearpfn._refcache): the first run builds it, later runs of the SAME
    panel/prior/code skip every fit_exact; any mismatch raises. Default off —
    behavior is then byte-identical to the uncached harness.
    """
    t_run = time.monotonic()
    device = resolve_device(config.get("device", "auto"))
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    prior_cfg = make_prior_config(config.get("data", {}))
    model, bar = build_model(state["config"]["model"], prior_cfg.include_interactions)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    bar.to(device)
    val_cfg = config["validate"]
    check_validate_config(val_cfg, prior_cfg)
    ref_cache = None
    if ref_cache_dir is not None:
        ref_cache = ReferenceCache(
            ref_cache_dir, cache_components(prior_cfg, val_cfg), rebuild=rebuild_ref_cache
        )
    timers: dict[str, float] = {}
    records: list[dict] = []
    cells: list[dict] = []
    for p in val_cfg["p_values"]:
        for n in val_cfg["n_values"]:
            cfg_cell = replace(prior_cfg, n_min=20, n_max=1024, p_min=2, p_max=30)
            cell = _evaluate_cell(model, bar, cfg_cell, n, p, val_cfg, device,
                                  ref_cache=ref_cache, timers=timers)
            summary = _cell_summary(cell)
            if summary.get("unconverged"):
                # strip reference fields so no downstream panel consumes an
                # unconverged (model-flattering) reference
                for r in cell:
                    for key in ("nll_exact", "coef_exact", "exact_pips"):
                        r.pop(key, None)
            records.extend(cell)
            cells.append(summary)
            status = (
                f" | closed {summary['gap_closed']:.3f}" if "gap_closed" in summary
                else " | UNCONVERGED" if summary.get("unconverged") else ""
            )
            print(f"cell n={n:5d} p={p:2d}: model nll {summary['nll_model']:.4f}{status}")
    pooled = _pooled_metrics(records)
    pooled["prior_strata"] = _prior_strata(records, prior_cfg)
    pooled["c_tercile_edges"] = c_tercile_edges(prior_cfg)
    gates = _gates(cells, pooled)
    results = {
        "cells": cells, "pooled": pooled, "gates": gates,
        "hard_gates": bool(val_cfg.get("hard_gates", False)),
        "ckpt": ckpt_path, "step": state.get("step"),
        "records": records,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    baseline = {
        "meta": {
            "ckpt": ckpt_path,
            "step": state.get("step"),
            "generated": datetime.now(UTC).isoformat(timespec="minutes"),
            "note": "in-prior surrogate-R2 calibration table; capacity-dependent — "
                    "regenerate whenever the model is retrained",
        },
        "cells": [
            {
                "n": n, "p": p,
                "values": sorted(
                    round(r["surrogate_r2"], 6) for r in records
                    if r["n"] == n and r["p"] == p and "surrogate_r2" in r
                ),
            }
            for p in val_cfg["p_values"] for n in val_cfg["n_values"]
        ],
    }
    (out / "surrogate_baseline.json").write_text(
        json.dumps(baseline, indent=1), encoding="utf-8")
    write_report(results, config, out)
    # Data-first record of the same numbers: everything but the
    # per-dataset records, so the manuscript pipeline reads values, not prose.
    (out / "results.json").write_text(json.dumps(
        {k: v for k, v in results.items() if k != "records"},
        indent=1, sort_keys=True, default=_json_default), encoding="utf-8")
    if ref_cache is not None:
        ref_cache.save()
    elapsed = time.monotonic() - t_run
    accounted = sum(timers.values())
    phase = " | ".join(f"{k} {v:.1f}s" for k, v in sorted(timers.items()))
    cache_note = (
        f" | ref-cache hits {ref_cache.hits} computed {ref_cache.computed}"
        if ref_cache is not None else ""
    )
    print(f"phase timing: total {elapsed:.1f}s | {phase} | "
          f"other {elapsed - accounted:.1f}s{cache_note}")
    mode = "HARD" if results["hard_gates"] else "informational (smoke)"
    print(f"\nGates ({mode}):")
    for g in gates:
        print(f"  [{'PASS' if g['passed'] else 'FAIL'}] {g['name']} (value {g['value']:.4f})")
    print(f"report written to {out_dir}/report.md")
    check_panels(results, val_cfg, exact_p_max=exact_reach(val_cfg, prior_cfg))
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate a LinearPFN checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--ref-cache", default=None, metavar="DIR",
        help="enable the keyed exact-reference cache in DIR (first run builds "
             "it; a stale cache RAISES instead of silently recomputing)",
    )
    parser.add_argument(
        "--rebuild-ref-cache", action="store_true",
        help="recompute and overwrite the reference cache (the only way past "
             "a staleness error)",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    out = args.out or f"reports/{Path(args.config).stem}"
    run_validation(config, args.ckpt, out, ref_cache_dir=args.ref_cache,
                   rebuild_ref_cache=args.rebuild_ref_cache)


if __name__ == "__main__":
    main()
