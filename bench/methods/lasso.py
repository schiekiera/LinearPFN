"""Lasso on the expanded design.

Input convention: the expanded design `build_design(X, cfg)[:, 1:]` (mains +
interaction products of standardized columns), re-standardized internally
because the lasso penalty is scale-sensitive and interaction columns have
variance 1 + rho^2 under the contract; coefficients are mapped back to the
contract scale. Selection = nonzero support at the 10-fold CV-min alpha.
Tier-1 score = the largest alpha at which the effect first enters the path
(earlier entry = stronger evidence), computed on the same alpha grid CV
used.
"""

from __future__ import annotations

import time

import numpy as np

from bench.harness import MethodResult
from linearpfn.prior import Dataset, PriorConfig, build_design


def _fit(ds: Dataset, cfg: PriorConfig) -> tuple[dict, float]:
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LassoCV, lasso_path

    # correlated real designs leave the path's smallest alphas short of
    # convergence; the CV-selected alpha converges — silence the spam
    warnings.simplefilter("ignore", ConvergenceWarning)

    Z = build_design(ds.X, cfg)[:, 1:]
    scale = Z.std(axis=0)
    scale[scale < 1e-12] = 1.0
    Zs = Z / scale
    t0 = time.perf_counter()
    cv = LassoCV(cv=10, n_alphas=100, max_iter=50_000, random_state=0)
    cv.fit(Zs, ds.y)
    alphas, coefs, _ = lasso_path(Zs, ds.y, alphas=cv.alphas_)
    fit_seconds = time.perf_counter() - t0
    entry = np.zeros(Z.shape[1])
    nz = np.abs(coefs) > 1e-12  # (d-1, n_alphas), alphas descending
    for j in range(Z.shape[1]):
        hits = np.flatnonzero(nz[j])
        if hits.size:
            entry[j] = float(alphas[hits[0]])  # first (largest) entering alpha
    coef_back = cv.coef_ / scale
    out = {
        "support": np.abs(cv.coef_) > 1e-12,
        "score": entry,
        "coef": np.concatenate([[float(cv.intercept_)], coef_back]),
        "alpha": float(cv.alpha_),
    }
    return out, fit_seconds


def lasso_raw(ds: Dataset, cfg: PriorConfig) -> MethodResult:
    out, secs = _fit(ds, cfg)
    return MethodResult(
        gamma_hat=out["support"], score=out["score"], prob=None,
        coef=out["coef"], fit_seconds=secs, meta={"alpha": out["alpha"]},
    )

