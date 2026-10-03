"""Stage 19: the `stability_refit` row — stability selection's support, with
coefficients from an OLS refit on it.

Stability selection returns a selection frequency per effect and no
coefficient vector (`bench/methods/stability.py`), so without this stage it
would be the one method with an empty coefficient column. This stage fills it
with least squares on the selected support of the CACHED fit, with every
non-selected slot at zero. Nothing is refitted — the inputs are the finished caches, the
selections are taken as they are, and the truth rows supply X and y.

The row is written in the benchmark's own npz layout by the benchmark's own
writer (`bench.shared._save`), so a `stability_refit` file is byte-for-
byte the shape of every other method row and the number stages read it with no
special case. `gamma_hat`, `score` and `prob` are copied from the stability
row, so F1, AUC and ECE are identical by construction and only the coefficient
column is new.

`fit_seconds` is the cached stability seconds PLUS the refit, so the row's
cost includes the selection it refits. The cached part was measured on one
CPU thread by stage 17 and the refit here on whatever machine runs this stage, so
the refit seconds and the platform are recorded in the row's meta; the refit
is a single `lstsq` and moves the median by well under a millisecond.

Run (pin the threads, so the refit is single-threaded like the cached fits):

    OMP_NUM_THREADS=1 python scripts/19_stability_refit.py
    OMP_NUM_THREADS=1 python scripts/19_stability_refit.py --force
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ROOT, P  # noqa: E402

SOURCE_ROW = "stability"
TARGET_ROW = "stability_refit"


def refit(X: np.ndarray, y: np.ndarray, gamma_hat: np.ndarray, cfg) -> tuple[np.ndarray, float]:
    """OLS on the selected support (intercept always in), zeros elsewhere.

    Guard: when the support plus the intercept exceeds the number of rows the
    system is underdetermined, and there is no penalized estimate to fall back
    on here, so the row keeps an all-zero coefficient vector.
    """
    from linearpfn.prior import build_design

    Z = build_design(X, cfg)
    cols = np.concatenate([[True], np.asarray(gamma_hat, bool)])
    t0 = time.perf_counter()
    coef = np.zeros(Z.shape[1])
    if cols.sum() <= Z.shape[0]:
        sol, *_ = np.linalg.lstsq(Z[:, cols], y, rcond=None)
        coef[cols] = sol
    return coef, time.perf_counter() - t0


def one_cache(results_dir: Path, cfg, force: bool) -> dict[str, int]:
    """Write every missing `stability_refit` row of one results directory."""
    from bench.harness import MethodResult
    from bench.shared import _save
    from linearpfn.prior import Dataset

    src, truth = results_dir / SOURCE_ROW, results_dir / "truth"
    if not src.is_dir():
        return {"written": 0, "skipped": 0, "missing_truth": 0, "underdetermined": 0}
    (results_dir / TARGET_ROW).mkdir(parents=True, exist_ok=True)
    counts = {"written": 0, "skipped": 0, "missing_truth": 0, "underdetermined": 0}
    host = f"{platform.system()} {platform.machine()}"
    for path in sorted(src.glob("*.npz")):
        out = results_dir / TARGET_ROW / path.name
        if out.is_file() and not force:
            counts["skipped"] += 1
            continue
        tz = truth / path.name
        if not tz.is_file():
            counts["missing_truth"] += 1
            continue
        s = np.load(path, allow_pickle=False)
        t = np.load(tz, allow_pickle=False)
        gamma_hat = s["gamma_hat"].astype(bool)
        coef, secs = refit(t["X"], t["y"], gamma_hat, cfg)
        if not coef.any() and gamma_hat.sum() + 1 > t["X"].shape[0]:
            counts["underdetermined"] += 1
        ds = Dataset(X=t["X"], y=t["y"],
                     gamma=np.concatenate([[True], t["gamma_true"].astype(bool)]),
                     beta=t["beta_true"], sigma2=1.0)
        res = MethodResult(gamma_hat=gamma_hat, score=s["score"], prob=s["prob"], coef=coef,
                           fit_seconds=float(s["fit_seconds"]) + secs,
                           meta={"refit": "ols_on_selected_support", "from": SOURCE_ROW,
                                 "refit_seconds": secs, "refit_host": host})
        res.validate(coef.size)
        _save(out, ds, res, json.loads(str(s["meta"])))
        counts["written"] += 1
    return counts


def main() -> int:
    from linearpfn.train import load_config, make_prior_config

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", nargs="+",
                    default=["results_main_strong", "results_knob_main"],
                    help="config path keys of the caches to fill")
    ap.add_argument("--config", default=None, help="prior config (default: paths.final_config)")
    ap.add_argument("--force", action="store_true", help="rewrite rows that already exist")
    args = ap.parse_args()

    pin = Path(args.config) if args.config else P["final_config"]
    cfg = make_prior_config(load_config(str(pin))["data"])
    total = {}
    for key in args.results:
        d = P[key]
        counts = one_cache(d, cfg, args.force)
        total[key] = counts
        rel = d.relative_to(ROOT) if d.is_relative_to(ROOT) else d
        print(f"{rel}/{TARGET_ROW}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    if any(c["missing_truth"] for c in total.values()):
        print("WARNING: some stability rows have no truth file and were skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
