"""Stage 18 (compute) — REAL OUTCOMES: LinearPFN against the posterior it
targets on published tables with their actual response column.

    python scripts/18_real_y_fit.py --index 3 \
        --ckpt checkpoints/paper_model/ckpt_final.pt \
        --config configs/paper_model.yaml --data-dir data/real_outcomes \
        --out-dir results/real_outcomes

Every benchmark simulates outcomes on real design matrices; this stage is the
one with real y. Without a true active set the only scores on real data are
(a) agreement with the posterior under the prior, sampled by the MCMC sampler
on every table (linearpfn.mc3_general with the step ladder and convergence
rule below; one reference throughout, so the tables compare like with like),
and (b) held-out predictive quality, which needs no truth. Seven tables, each
with a real, continuous, documented outcome whose signal level is what a
typical applied dataset shows, n in the training range and p <= 13 so the
sampler converges in minutes; four inside the prior's X law and three
collinearity stress tests (bodyfat, ozone, diabetes).

Preprocessing, identical for every method: features z-scored (ddof=0, the
generator contract), y centred and scaled to unit sd because the prior is
calibrated to unit-scale y (raw sd ranges 1.2 to 77 across the seven). The
full-table fit gives the agreement numbers; a seeded split (20 % query rows,
statistics from the context rows) gives held-out NLL and RMSE for the model,
the reference, OLS on the mains and the context mean.

One npz per (row, dataset) under <out-dir>/<row>/<dataset>.npz, rows
`linearpfn_strong` (head coefficients; the probe vector rides along as
coef_probe) and `mc3`; a rerun skips finished datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASETS = ("attitude", "swiss", "prostate", "tal_or", "bodyfat", "ozone", "diabetes")
# MCMC steps per chain by p: the ladder every MCMC run of the benchmark uses
STEP_LADDER = ((5, 100_000), (10, 200_000), (20, 500_000), (30, 1_000_000))
RHAT_MAX, ESS_MIN = 1.1, 200.0
QUERY_FRAC, MIN_QUERY = 0.2, 5


def steps_for(p: int) -> int:
    for bound, steps in STEP_LADDER:
        if p <= bound:
            return steps
    return STEP_LADDER[-1][1]


def load_table(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import pandas as pd

    df = pd.read_csv(path)
    assert df.columns[-1] == "y", path
    names = list(df.columns[:-1])
    return df[names].to_numpy(float), df["y"].to_numpy(float), names


def zscore(a: np.ndarray, stats: tuple | None = None) -> tuple[np.ndarray, tuple]:
    """(a - mean) / sd with ddof=0; `stats` reuses another block's moments."""
    mu, sd = stats if stats is not None else (a.mean(0), a.std(0))
    return (a - mu) / sd, (mu, sd)


def split_rows(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    nq = max(MIN_QUERY, int(round(QUERY_FRAC * n)))
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[nq:]), np.sort(perm[:nq])


def gaussian_nll(y: np.ndarray, mean, var: float) -> float:
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (y - mean) ** 2 / var))


def ols_mains(X_ctx, y_ctx, X_q, y_q) -> dict:
    A = np.c_[np.ones(len(y_ctx)), X_ctx]
    b = np.linalg.lstsq(A, y_ctx, rcond=None)[0]
    resid = y_ctx - A @ b
    dof = max(len(y_ctx) - A.shape[1], 1)
    pred = np.c_[np.ones(len(y_q)), X_q] @ b
    return {"nll": gaussian_nll(y_q, pred, float(resid @ resid / dof)),
            "rmse": float(np.sqrt(np.mean((y_q - pred) ** 2)))}


def fit_model(model, bar, cfg, X_ctx, y_ctx, X_q=None, y_q=None, probe_rng=None) -> dict:
    import torch

    from linearpfn.probe import extract_coefficients

    n = len(y_ctx)
    X_all = X_ctx if X_q is None else np.vstack([X_ctx, X_q])
    X_t = torch.from_numpy(X_all).float().unsqueeze(0)
    y_t = torch.from_numpy(y_ctx).float().unsqueeze(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, sel, coef_out = model.forward_with_heads(X_t, y_t, n)
        prob = torch.sigmoid(sel.float())[0].numpy()
        eff = model.coefficient_posterior_mean(sel[0], coef_out[0]).numpy()
    out = {"prob": prob, "coef": np.concatenate([[float(y_ctx.mean())], eff]),
           "fit_seconds": time.perf_counter() - t0}
    if X_q is not None:
        with torch.no_grad():
            lg = logits.float()
            out["nll_query"] = float(bar.nll(lg, torch.from_numpy(y_q).float().unsqueeze(0)).mean())
            out["pred_query"] = bar.mean(lg)[0].numpy()
    if probe_rng is not None:
        X_ctx_t = torch.from_numpy(X_ctx).float()

        def mean_fn(probes: np.ndarray) -> np.ndarray:
            Xp = torch.cat([X_ctx_t, torch.from_numpy(probes).float()]).unsqueeze(0)
            with torch.no_grad():
                return bar.mean(model(Xp, y_t, n).float())[0].numpy()

        t1 = time.perf_counter()
        res = extract_coefficients(probe_rng, X_ctx, mean_fn, cfg=cfg)
        out["coef_probe"] = res.coef
        out["probe_seconds"] = time.perf_counter() - t1
        out["probe_r2"] = float(res.r2)
    return out


def fit_reference(cfg, X_ctx, y_ctx, X_q=None, y_q=None, seed: int = 0, chains: int = 4) -> dict:
    """MCMC sampler under the prior (the step ladder), with the predictive on the query rows."""
    p = X_ctx.shape[1]
    from linearpfn.mc3_general import fit_mc3_general

    steps = steps_for(p)
    t0 = time.perf_counter()
    r = fit_mc3_general(X_ctx, y_ctx, cfg, X_query=X_q, y_query=y_q,
                        n_chains=chains, n_steps=steps, seed=seed)
    wall = time.perf_counter() - t0
    dg = r.diagnostics
    out = {"kind": "mc3", "prob": r.pip[1:], "coef": r.coef_mean, "fit_seconds": r.seconds,
           "wall_seconds": wall, "seconds_diag": r.seconds_diag, "rhat_max": float(dg.rhat_max),
           "ess_min": float(dg.ess_min), "n_steps": int(dg.n_steps), "n_chains": chains,
           "accept_rate": float(getattr(dg, "accept_rate", np.nan)),
           "converged": bool(dg.rhat_max < RHAT_MAX and dg.ess_min > ESS_MIN)}
    if X_q is not None:
        out["nll_query"] = np.nan if r.nll_query is None else float(r.nll_query)
        from linearpfn.prior import build_design
        out["pred_query"] = build_design(X_q, cfg) @ r.coef_mean
    return out


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def run_one(name: str, args, model, bar, cfg) -> None:
    Xr, yr, names = load_table(Path(args.data_dir) / f"{name}.csv")
    n, p = Xr.shape
    seed = int(np.frombuffer(name.encode(), dtype=np.uint8).sum()) + args.seed
    # full table: the agreement fits
    X, _ = zscore(Xr)
    y, (y_mu, y_sd) = zscore(yr)
    # held-out split: context statistics applied to the query rows
    ctx, q = split_rows(n, seed)
    X_ctx, xs = zscore(Xr[ctx])
    X_q, _ = zscore(Xr[q], xs)
    y_ctx, ys = zscore(yr[ctx])
    y_q, _ = zscore(yr[q], ys)
    common = {"n": n, "p": p, "n_ctx": len(ctx), "n_query": len(q), "dataset": name,
              "features": np.array(names), "y_mean": float(y_mu), "y_sd": float(y_sd),
              "split_seed": seed, "query_rows": q,
              "nll_marginal_query": gaussian_nll(y_q, float(y_ctx.mean()), float(y_ctx.var())),
              "rmse_marginal_query": float(np.sqrt(np.mean((y_q - y_ctx.mean()) ** 2))),
              **{f"ols_mains_{k}_query": v for k, v in ols_mains(X_ctx, y_ctx, X_q, y_q).items()}}
    out = Path(args.out_dir)
    rows = {}
    if "model" in args.rows:
        full = fit_model(model, bar, cfg, X, y, probe_rng=np.random.default_rng(seed))
        held = fit_model(model, bar, cfg, X_ctx, y_ctx, X_q, y_q)
        rows[args.model_row] = {**common, **full, "nll_query": held["nll_query"],
                                "pred_query": held["pred_query"],
                                "rmse_query": _rmse(y_q, held["pred_query"]),
                                "prob_ctx": held["prob"], "coef_ctx": held["coef"],
                                "ckpt": args.ckpt}
    if "reference" in args.rows:
        full = fit_reference(cfg, X, y, seed=seed, chains=args.chains)
        held = fit_reference(cfg, X_ctx, y_ctx, X_q, y_q, seed=seed + 1, chains=args.chains)
        rows[full["kind"]] = {**common, **full, "nll_query": held["nll_query"],
                              "pred_query": held["pred_query"],
                              "rmse_query": _rmse(y_q, held["pred_query"]),
                              "prob_ctx": held["prob"], "coef_ctx": held["coef"],
                              "converged_ctx": held["converged"],
                              "fit_seconds_ctx": held["fit_seconds"]}
    for row, rec in rows.items():
        (out / row).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / row / f"{name}.npz", **rec)
        msg = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()
               if k in ("fit_seconds", "nll_query", "rmse_query", "rhat_max", "ess_min",
                        "converged", "kind", "n_steps")}
        print(f"{name} [{row}] n={n} p={p}: {json.dumps(msg)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--index", type=int, default=None, help="one dataset of DATASETS (array task)")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--rows", nargs="*", default=["model", "reference"])
    ap.add_argument("--ckpt", default="checkpoints/paper_model/ckpt_final.pt")
    ap.add_argument("--config", default="configs/paper_model.yaml")
    ap.add_argument("--model-row", default="linearpfn_strong")
    ap.add_argument("--data-dir", default="data/real_outcomes")
    ap.add_argument("--out-dir", default="results/real_outcomes")
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    names = [args.datasets[args.index]] if args.index is not None else list(args.datasets)

    import torch

    from bench.methods.linearpfn_slot import _load

    torch.set_num_threads(1)
    model, bar, cfg = _load(args.ckpt, args.config)
    for name in names:
        out = Path(args.out_dir)
        have_ref = (out / "mc3" / f"{name}.npz").exists()
        if not args.force and (out / args.model_row / f"{name}.npz").exists() and have_ref:
            print(f"{name}: cached, skip", flush=True)
            continue
        run_one(name, args, model, bar, cfg)
    print("REAL-Y-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
