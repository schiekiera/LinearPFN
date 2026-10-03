"""Stability selection: complementary-pairs subsampling around the lasso.

Settings: Shah & Samworth CPSS with B = 50 complementary pairs
(100 lasso-path fits on floor(n/2) subsamples), per-effect selection
frequency maximized over the alpha path restricted to
[0.1 * alpha_max, alpha_max] (without the restriction every effect enters
as alpha -> 0 and frequencies saturate), operating point PI >= 0.75.
Frequencies double as Tier-2 pseudo-probabilities with the standing caveat:
selection frequencies are NOT posterior probabilities and are expected to
be miscalibrated — that expectation is part of what Tier 2 measures.
No heredity constraint; the heredity-violation rate is an informative
Tier-1 metric for this method. The benchmark runs the lasso-wrapped
canonical form.
"""

from __future__ import annotations

import time

import numpy as np

from bench.harness import MethodResult
from linearpfn.prior import Dataset, PriorConfig, build_design

B_PAIRS = 50
ALPHA_RATIO = 0.1
N_ALPHAS = 50
THRESHOLD = 0.75


def stability_cpss(ds: Dataset, cfg: PriorConfig) -> MethodResult:
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import lasso_path

    warnings.simplefilter("ignore", ConvergenceWarning)

    Z = build_design(ds.X, cfg)[:, 1:]
    scale = Z.std(axis=0)
    scale[scale < 1e-12] = 1.0
    Zs = Z / scale
    n, m = Zs.shape
    y = ds.y
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    yc = y - y.mean()
    alpha_max = float(np.abs(Zs.T @ yc).max() / n)
    alphas = np.geomspace(alpha_max, ALPHA_RATIO * alpha_max, N_ALPHAS)
    half = n // 2
    counts = np.zeros(m)
    total = 0
    for _ in range(B_PAIRS):
        perm = rng.permutation(n)
        for rows in (perm[:half], perm[half : 2 * half]):
            _, coefs, _ = lasso_path(Zs[rows], y[rows], alphas=alphas)
            counts += (np.abs(coefs) > 1e-12).any(axis=1)
            total += 1
    freq = counts / total
    fit_seconds = time.perf_counter() - t0
    return MethodResult(
        gamma_hat=freq >= THRESHOLD, score=freq, prob=freq,
        coef=None, fit_seconds=fit_seconds,
        meta={"B_pairs": B_PAIRS, "threshold": THRESHOLD,
              "alpha_ratio": ALPHA_RATIO},
    )
