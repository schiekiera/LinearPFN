"""Exact enumeration reference as the benchmark ceiling row (p <= 5).

Not a competitor — the Bayes-optimal answer under the true prior, i.e. the
score every method is implicitly chasing. Selection = median probability
model (PIP > 0.5); coefficients = exact posterior means. Runner caps it at
p <= MAX_ENUM_P via DEFAULT_P_CAPS.
"""

from __future__ import annotations

import time

import numpy as np

from bench.harness import MethodResult
from linearpfn.prior import Dataset, PriorConfig
from linearpfn.reference import fit_exact


def exact_reference(ds: Dataset, cfg: PriorConfig) -> MethodResult:
    t0 = time.perf_counter()
    post = fit_exact(ds.X.astype(float), ds.y.astype(float), cfg,
                     standardization_tol=1e-6)
    pip = post.pip()[1:]
    coef = post.coef_mean()
    fit_seconds = time.perf_counter() - t0
    return MethodResult(
        gamma_hat=pip > 0.5, score=pip.copy(), prob=pip.copy(),
        coef=np.asarray(coef), fit_seconds=fit_seconds,
        meta={"n_models": int(post.gammas.shape[0])},
    )
