"""Pre-launch diagnostics of the prior: a battery of panels on prior draws.

Samples datasets from the PriorConfig defaults (or a config's data block)
and reports everything worth looking at before committing days of GPU time
to a training run: the R2 regime (overall and split by activity,
heredity mode, and drawn c), the interaction signal share by rho, null
fractions by p, E[k] against the closed form, the realized correlation
distribution (smoothness, tails, near-duplicate shares, copula attenuation),
the factor-share distribution, cross-block correlations, interaction-column
magnitudes, and NaN/inf/constant counts.

Output: <out>/preflight.json (all numbers + a `checks` list evaluating the
launch criteria) and <out>/preflight.png (plot grid, lazy matplotlib).

Run: python -m linearpfn.preflight [--config CFG] [--n-datasets 10000]
     [--seed 0] [--out reports/preflight]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from linearpfn.prior import (
    PriorConfig,
    build_design,
    expected_active_count,
    sample_dataset_with_meta,
    sample_X,
)
from linearpfn.train import load_config, make_prior_config

__all__ = ["run_preflight"]

PCTS = (5, 25, 50, 75, 95)
R_HIST_BINS = 40
ATTENUATION_CAP = 3000  # datasets contributing per-pair latent-vs-realized samples


def _pct(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {}
    return {f"p{q}": float(np.percentile(arr, q)) for q in PCTS}


def _pair_label(t1: str, t2: str) -> str:
    return "-".join(sorted((t1, t2)))


P5_TAIL_DRAWS = 4000


def _p5_maxr_tail(cfg: PriorConfig, seed: int, n_draws: int = P5_TAIL_DRAWS) -> dict:
    """P(max |r| >= t) at p = 5 — the acceptance diagnostic for the X-law's
    high-collinearity tail: coefficient error relative to the exact posterior
    grows sharply past max |r| ~ 0.95, so the share
    of draws that REACH that band is what coverage means here.

    Measured on a dedicated fixed-p sample rather than the main sweep's
    p ~ U{2..30} draws, which leave only ~n/29 datasets at p = 5 — too few
    to resolve a percent-level rate. X only (no y/params), so it is cheap.
    n is drawn from the envelope, so the realized correlations carry the
    same small-n sampling noise the model actually trains on.
    """
    rng = np.random.default_rng([seed, 5])
    iu, ju = np.triu_indices(5, k=1)
    maxr = np.empty(n_draws)
    high = 0
    for i in range(n_draws):
        n = int(rng.integers(cfg.n_min, cfg.n_max + 1))
        X, meta = sample_X(rng, n, 5, cfg)
        maxr[i] = np.abs(np.corrcoef(X, rowvar=False)[iu, ju]).max()
        high += int(meta.h2_high)
    return {
        "n_draws": n_draws,
        "share_ge_077": float((maxr >= 0.77).mean()),
        "share_ge_090": float((maxr >= 0.90).mean()),
        "share_ge_095": float((maxr >= 0.95).mean()),
        "share_ge_099": float((maxr >= 0.99).mean()),
        "share_high_regime": high / n_draws,
        "pcts": _pct(maxr),
    }


def run_preflight(cfg: PriorConfig, n_datasets: int, seed: int, out_dir: Path) -> dict:
    rng = np.random.default_rng(seed)
    r2s, ks_main, ks_int, ks_tot, modes, cs, rhos = [], [], [], [], [], [], []
    ps, int_shares, f_shares, maxr_per_ds = [], [], [], []
    r_pool: list[np.ndarray] = []
    r_by_pair: dict[str, list[np.ndarray]] = {}
    atten: dict[str, list[np.ndarray]] = {}  # per pair label: |r_lat| - |r_real|
    cross_r: list[np.ndarray] = []
    int_max_by_pair: dict[str, float] = {}
    int_max_all: list[float] = []
    n_bad = {"nan_inf": 0, "constant_cols": 0}

    y_escape: list[float] = []  # per-dataset fraction of rows outside [-8, 8]
    # per-active-effect standardized size |beta_j| sd(Z_j) / sd(y) — the
    # quantity meta-analytic |r| norms speak about (count-prior effect-size check)
    eff_main: list[float] = []
    eff_int: list[float] = []
    for i in range(n_datasets):
        ds, meta = sample_dataset_with_meta(rng, cfg)
        n, p = ds.X.shape
        Z = build_design(ds.X, cfg)
        signal = Z @ ds.beta
        r2s.append(float(signal.var() / ds.y.var()))
        y_escape.append(float((np.abs(ds.y) > 8.0).mean()))
        y_sd = float(ds.y.std())
        if y_sd > 0:
            for j in np.flatnonzero(ds.gamma[1:]) + 1:
                r_j = abs(ds.beta[j]) * float(Z[:, j].std()) / y_sd
                (eff_main if j <= p else eff_int).append(r_j)
        k_main = int(ds.gamma[1 : 1 + p].sum())
        k_int = int(ds.gamma[1 + p :].sum())
        ks_main.append(k_main)
        ks_int.append(k_int)
        ks_tot.append(k_main + k_int)
        modes.append(ds.heredity_mode)
        cs.append(ds.c)
        rhos.append(ds.rho)
        ps.append(p)
        f_shares.append(meta.p_fac / p)
        if k_int and signal.var() > 0:
            sig_int = Z[:, 1 + p :] @ ds.beta[1 + p :]
            int_shares.append(float(sig_int.var() / signal.var()))
        else:
            int_shares.append(float("nan"))
        if not np.isfinite(ds.X).all() or not np.isfinite(ds.y).all():
            n_bad["nan_inf"] += 1
        if (ds.X.std(axis=0) <= 0).any():
            n_bad["constant_cols"] += 1
        iu, ju = np.triu_indices(p, k=1)
        if iu.size:
            R = np.corrcoef(ds.X, rowvar=False)
            r_vals = R[iu, ju].astype(np.float32)
            r_pool.append(r_vals)
            maxr_per_ds.append(float(np.abs(r_vals).max()))
            types = meta.column_types
            labels = [
                _pair_label(str(types[a]), str(types[b]))
                for a, b in zip(iu, ju, strict=True)
            ]
            for lab in set(labels):
                mask = np.array([la == lab for la in labels])
                r_by_pair.setdefault(lab, []).append(r_vals[mask])
            if i < ATTENUATION_CAP and not cfg.is_gaussian_only():
                R_lat = np.corrcoef(meta.latent, rowvar=False)
                gap = (np.abs(R_lat[iu, ju]) - np.abs(R[iu, ju])).astype(np.float32)
                for lab in set(labels):
                    mask = np.array([la == lab for la in labels])
                    atten.setdefault(lab, []).append(gap[mask])
            if 0 < meta.p_fac < p and i < ATTENUATION_CAP:
                lat = meta.latent
                gen, fac = lat[:, ~meta.is_factor], lat[:, meta.is_factor]
                cc = np.corrcoef(gen.T, fac.T)[: gen.shape[1], gen.shape[1] :]
                cross_r.append(cc.ravel().astype(np.float32))
            if cfg.include_interactions:
                col_max = np.abs(Z[:, 1 + p :]).max(axis=0)
                int_max_all.append(float(col_max.max()))
                for t, (a, b) in enumerate(zip(iu, ju, strict=True)):
                    lab = _pair_label(str(types[a]), str(types[b]))
                    int_max_by_pair[lab] = max(int_max_by_pair.get(lab, 0.0), float(col_max[t]))

    r_all = np.concatenate(r_pool) if r_pool else np.zeros(0, dtype=np.float32)
    ps_arr = np.asarray(ps)
    k_arr = np.asarray(ks_tot)
    c_arr = np.asarray(cs)
    rho_arr = np.asarray(rhos)
    r2_arr = np.asarray(r2s)
    share_arr = np.asarray(int_shares)
    mode_arr = np.asarray(modes)

    # R2 splits
    r2_by_mode = {m: _pct(r2_arr[mode_arr == m]) for m in sorted(set(modes))}
    k_bins = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, 10**9)]
    r2_by_k = {
        f"{lo}-{hi if hi < 10**9 else 'inf'}": _pct(r2_arr[(k_arr >= lo) & (k_arr <= hi)])
        for lo, hi in k_bins
        if ((k_arr >= lo) & (k_arr <= hi)).any()
    }
    def _by_decile(values: np.ndarray, target: np.ndarray, base: np.ndarray, key: str):
        edges = np.percentile(values, np.arange(0, 101, 10))
        rows = []
        for i in range(10):
            m = base & (values >= edges[i]) & (values <= edges[i + 1])
            rows.append({
                f"{key}_range": [float(edges[i]), float(edges[i + 1])],
                "median": float(np.median(target[m])) if m.any() else float("nan"),
            })
        return rows

    all_ds = np.ones(r2_arr.size, dtype=bool)
    r2_by_c_decile = [
        {"c_range": row["c_range"], "r2_median": row["median"]}
        for row in _by_decile(c_arr, r2_arr, all_ds, "c")
    ]
    # interaction share by rho decile / k_int
    has_int = np.isfinite(share_arr)
    share_by_rho = [
        {"rho_range": row["rho_range"], "share_median": row["median"]}
        for row in _by_decile(rho_arr, share_arr, has_int, "rho")
    ]
    ki_arr = np.asarray(ks_int)
    share_by_kint = {
        lab: _pct(share_arr[has_int & m])
        for lab, m in (("1", ki_arr == 1), ("2", ki_arr == 2), ("3+", ki_arr >= 3))
        if (has_int & m).any()
    }
    # null fraction and E[k] by p
    null_by_p = {}
    ek_by_p = {}
    for p in sorted(set(ps)):
        m = ps_arr == p
        null_by_p[str(p)] = float((k_arr[m] == 0).mean())
        se = float(k_arr[m].std(ddof=1) / np.sqrt(m.sum())) if m.sum() > 1 else float("nan")
        analytic = float(expected_active_count(p, cfg))
        ek_by_p[str(p)] = {
            "mc": float(k_arr[m].mean()),
            "analytic": analytic,
            "mc_se": se,
            "z": float((k_arr[m].mean() - analytic) / se) if se > 0 else float("nan"),
        }
    # correlation histogram smoothness + tails
    hist, _edges = np.histogram(r_all, bins=R_HIST_BINS, range=(-1.0, 1.0))
    nz = np.flatnonzero(hist)
    empty_interior = int((hist[nz.min() : nz.max() + 1] == 0).sum()) if nz.size else 0
    maxr_arr = np.asarray(maxr_per_ds)
    results = {
        "n_datasets": n_datasets,
        "seed": seed,
        # nonnull: conditional on k > 0. With reject_empty=False the prior
        # deliberately places ~10% mass on null datasets whose R2 is exactly
        # 0, so pooled low percentiles are 0 BY CONSTRUCTION; the weak-signal
        # tail is only meaningful conditional on a signal existing.
        "r2": {"overall": _pct(r2_arr), "nonnull": _pct(r2_arr[k_arr > 0]),
               "by_mode": r2_by_mode, "by_k": r2_by_k,
               "by_c_decile": r2_by_c_decile},
        # y outside the bar distribution's [-8, 8] support: the DIRECT
        # measurement of the y-scale risk that extreme interaction-column
        # values proxy (they only reach y through an ACTIVE beta). Under the
        # fixed-scale configuration ~2e-5 of rows escape, pooled.
        "y_escape": {
            "pooled_row_frac": float(np.mean(y_escape)),
            "share_ds_any": float(np.mean(np.asarray(y_escape) > 0)),
        },
        "int_share": {"by_rho_decile": share_by_rho, "by_k_int": share_by_kint,
                      "n_datasets_with_int": int(has_int.sum())},
        "null": {"pooled": float((k_arr == 0).mean()), "by_p": null_by_p},
        "ek_by_p": ek_by_p,
        "corr": {
            "pooled_abs": _pct(np.abs(r_all)),
            "hist_counts": hist.tolist(),
            "empty_interior_bins": empty_interior,
            "tail_pos_frac": float((r_all > 0.9).mean()),
            "tail_neg_frac": float((r_all < -0.9).mean()),
            "by_pair": {
                lab: {"count": int(sum(len(a) for a in arrs)),
                      **_pct(np.abs(np.concatenate(arrs)))}
                for lab, arrs in sorted(r_by_pair.items())
            },
            "attenuation_by_pair": {
                lab: float(np.mean(np.concatenate(arrs)))
                for lab, arrs in sorted(atten.items())
            },
            "max_per_dataset": _pct(maxr_arr),
            "share_ds_max_gt_090": float((maxr_arr > 0.90).mean()),
            "share_ds_max_gt_095": float((maxr_arr > 0.95).mean()),
            "p5_tail": _p5_maxr_tail(cfg, seed),
        },
        "factor_share": {
            "pcts": _pct(np.asarray(f_shares)),
            "share_zero": float(np.mean(np.asarray(f_shares) == 0.0)),
            "share_one": float(np.mean(np.asarray(f_shares) == 1.0)),
        },
        "cross_block_r": (
            {
                "mean": float(np.mean(cc_all := np.concatenate(cross_r))),
                "sd": float(np.std(cc_all)),
                "share_exactly_zero": float(np.mean(cc_all == 0.0)),
            }
            if cross_r
            else None
        ),
        "interaction_max": (
            {
                "overall_max": float(max(int_max_all)),
                "share_ds_gt_15": float(np.mean(np.asarray(int_max_all) > 15.0)),
                "by_pair": {k: float(v) for k, v in sorted(int_max_by_pair.items())},
            }
            if int_max_all
            else None
        ),
        "bad_counts": n_bad,
        "c_pcts": _pct(c_arr),
        "rho_pcts": _pct(rho_arr),
        # standardized per-effect sizes; the count-prior launch check compares
        # the main-effect quartiles to meta-analytic norms for published
        # psychology correlations (Gignac & Szodorai 2016: .11/.19/.29)
        "effect_size": {"mains": _pct(eff_main), "interactions": _pct(eff_int),
                        "n_mains": len(eff_main), "n_interactions": len(eff_int)},
    }
    results["checks"] = _launch_checks(results, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight.json").write_text(json.dumps(results, indent=1))
    _figure(results, r_all, maxr_arr, np.asarray(f_shares), cross_r, int_max_all,
            r2_arr, out_dir)
    return results


def _launch_checks(res: dict, cfg: PriorConfig) -> list[dict]:
    """The launch criteria, evaluated in place so a review reads pass/fail
    instead of re-deriving thresholds. Law-aware: the rate prior (per-k
    slab) is checked on the R2 envelope; the count prior (fixed per-effect
    slab) is checked on per-effect standardized sizes against meta-analytic
    norms instead — R2 flatness is a constraint of the per-k slab only.
    R2 stays REPORTED under both laws."""
    r2 = res["r2"]["overall"]
    null_p2 = res["null"]["by_p"].get("2", 0.0)
    ek_z = [abs(v["z"]) for v in res["ek_by_p"].values() if np.isfinite(v["z"])]
    cross = res["cross_block_r"]
    r2_nonnull = res["r2"]["nonnull"]
    # The mixture prior keeps the fixed per-effect slab, so the count-prior
    # effect-size checks apply to both of its components; the rate-prior R2
    # bands are calibrated for the per-k slab and would be the wrong check.
    if cfg.sparsity_law in ("count", "mixture"):
        eff_m = res["effect_size"]["mains"]
        eff_i = res["effect_size"]["interactions"]
        # Bands bracket the Gignac & Szodorai quartiles (.11/.19/.29) with
        # room for the prior's deliberate tails on both sides.
        law_checks = [
            {"name": "main-effect |r| p25 in [0.06, 0.16]",
             "value": eff_m.get("p25"),
             "ok": 0.06 <= eff_m.get("p25", -1) <= 0.16},
            {"name": "main-effect |r| median in [0.14, 0.26]",
             "value": eff_m.get("p50"),
             "ok": 0.14 <= eff_m.get("p50", -1) <= 0.26},
            {"name": "main-effect |r| p75 in [0.24, 0.44]",
             "value": eff_m.get("p75"),
             "ok": 0.24 <= eff_m.get("p75", -1) <= 0.44},
            {"name": "interaction |r| median below main-effect median",
             "value": [eff_i.get("p50"), eff_m.get("p50")],
             "ok": bool(eff_i) and eff_i.get("p50", 1.0) < eff_m.get("p50", 0.0)},
        ]
    else:
        law_checks = [
            # The p5 band is evaluated on NON-NULL datasets: the (required,
            # intended) >= 5% null mass forces a pooled p5 to exactly 0.
            {"name": "R2 p5 among non-null datasets in [0.02, 0.10]",
             "value": r2_nonnull.get("p5"),
             "ok": 0.02 <= r2_nonnull.get("p5", -1) <= 0.10},
            {"name": "R2 p95 in [0.40, 0.70]", "value": r2.get("p95"),
             "ok": 0.40 <= r2.get("p95", -1) <= 0.70},
            {"name": "R2 median in [0.18, 0.40]", "value": r2.get("p50"),
             "ok": 0.18 <= r2.get("p50", -1) <= 0.40},
        ]
    checks = [
        *law_checks,
        {"name": "pooled null fraction in [0.05, 0.18]", "value": res["null"]["pooled"],
         "ok": 0.05 <= res["null"]["pooled"] <= 0.18},
        {"name": "null fraction at p=2 <= 0.60", "value": null_p2, "ok": null_p2 <= 0.60},
        # The criterion is the MAX of ~29 per-p z-scores; a single-test 3.0
        # threshold would false-alarm ~7.5% of the time by construction.
        # 3.5 keeps the max-statistic false-alarm rate ~1.3% while any
        # real closed-form/sampler drift still trips it loudly.
        {"name": "E[k] within 3.5 MC-SE of closed form at every p (max stat)",
         "value": max(ek_z) if ek_z else None, "ok": bool(ek_z) and max(ek_z) <= 3.5},
        {"name": "|r| histogram: no empty interior bin (40 bins)",
         "value": res["corr"]["empty_interior_bins"],
         "ok": res["corr"]["empty_interior_bins"] == 0},
        {"name": "correlation tails past +0.9 and -0.9",
         "value": [res["corr"]["tail_pos_frac"], res["corr"]["tail_neg_frac"]],
         "ok": res["corr"]["tail_pos_frac"] > 0 and res["corr"]["tail_neg_frac"] > 0},
        # X-law high-collinearity coverage. The band matters because
        # coefficient error relative to the exact posterior grows sharply past
        # max |r| ~ 0.95; without the high-communality regime this share is
        # ~0.8% (0.0% at 0.99), leaving the hardest regime nearly unreachable.
        # The upper bound guards the other way: the tail must not swallow the
        # prior's bulk.
        {"name": "p=5 tail P(max |r| >= .95) in [0.02, 0.10]",
         "value": res["corr"]["p5_tail"]["share_ge_095"],
         "ok": 0.02 <= res["corr"]["p5_tail"]["share_ge_095"] <= 0.10},
        {"name": "p=5 tail reaches max |r| >= .99 (share > 0)",
         "value": res["corr"]["p5_tail"]["share_ge_099"],
         "ok": res["corr"]["p5_tail"]["share_ge_099"] > 0.0},
        {"name": "pure-generic dataset share >= 0.10",
         "value": res["factor_share"]["share_zero"],
         "ok": res["factor_share"]["share_zero"] >= 0.10},
        {"name": "cross-block r non-degenerate (sd > 0.02) and centred (|mean| < 0.05)",
         "value": None if cross is None else [cross["mean"], cross["sd"]],
         "ok": cross is not None and cross["sd"] > 0.02 and abs(cross["mean"]) < 0.05},
        # The check is the DIRECT measurement of the y-scale risk rather than
        # a bound on raw interaction values (y escaping the bar distribution's
        # [-8, 8] support). Pure gaussian-gaussian products already reach ~25,
        # so a <= 25 bound on interaction values would nearly fail a plain
        # Gaussian design, while heavy-tailed marginal products (~100) sit in
        # mostly INACTIVE columns and barely reach y (~2e-5 of rows escape
        # under the fixed-scale configuration; threshold 5e-4 keeps tail-bin
        # rows negligible for the NLL).
        # The per-pair interaction maxima stay REPORTED for eyeballing.
        {"name": "rows with |y| > 8 (bar support) <= 5e-4 pooled",
         "value": res["y_escape"]["pooled_row_frac"],
         "ok": res["y_escape"]["pooled_row_frac"] <= 5e-4},
        {"name": "NaN/inf/constant counts == 0", "value": res["bad_counts"],
         "ok": res["bad_counts"]["nan_inf"] == 0 and res["bad_counts"]["constant_cols"] == 0},
    ]
    return checks


def _figure(res, r_all, maxr_arr, f_arr, cross_r, int_max_all, r2_arr, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    ax = axes.ravel()
    ax[0].hist(r2_arr, bins=50, color="tab:blue")
    ax[0].set_title("true R2")
    med = [row["r2_median"] for row in res["r2"]["by_c_decile"]]
    ax[1].plot(range(1, 11), med, "o-")
    ax[1].set_title("R2 median by c decile")
    ax[2].hist(r_all, bins=R_HIST_BINS, range=(-1, 1), color="tab:green")
    ax[2].set_yscale("log")
    ax[2].set_title("realized pairwise r (log count)")
    ax[3].hist(maxr_arr, bins=40, color="tab:orange")
    ax[3].set_title("max |r| per dataset")
    ax[4].hist(f_arr, bins=30, color="tab:purple")
    ax[4].set_title("factor share p_fac / p")
    if cross_r:
        ax[5].hist(np.concatenate(cross_r), bins=40, color="tab:red")
    ax[5].set_title("cross-block latent r")
    p_keys = sorted(int(k) for k in res["ek_by_p"])
    ax[6].plot(p_keys, [res["ek_by_p"][str(p)]["mc"] for p in p_keys], "o", label="MC")
    ax[6].plot(p_keys, [res["ek_by_p"][str(p)]["analytic"] for p in p_keys], "-",
               label="closed form")
    ax[6].set_title("E[k] by p")
    ax[6].legend()
    ax[7].plot(p_keys, [res["null"]["by_p"][str(p)] for p in p_keys], "o-")
    ax[7].set_title("null fraction by p")
    if int_max_all:
        ax[8].hist(np.log10(np.asarray(int_max_all)), bins=40, color="tab:gray")
        ax[8].axvline(np.log10(15.0), color="red", ls="--")
    ax[8].set_title("log10 max |interaction value| per dataset")
    fig.suptitle(f"prior preflight — {res['n_datasets']} datasets, seed {res['seed']}")
    fig.tight_layout()
    fig.savefig(out_dir / "preflight.png", dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prior preflight diagnostics")
    parser.add_argument("--config", default=None,
                        help="YAML config; its data block builds the prior "
                             "(default: current PriorConfig defaults)")
    parser.add_argument("--n-datasets", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="reports/preflight")
    args = parser.parse_args(argv)
    cfg = (
        make_prior_config(load_config(args.config)["data"])
        if args.config
        else PriorConfig()
    )
    res = run_preflight(cfg, args.n_datasets, args.seed, Path(args.out))
    print(f"preflight: {args.n_datasets} datasets -> {args.out}/preflight.json")
    for chk in res["checks"]:
        print(f"  [{'PASS' if chk['ok'] else 'FAIL'}] {chk['name']} (value {chk['value']})")


if __name__ == "__main__":
    main()
