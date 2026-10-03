"""Coefficient extraction by probing a fitted predictive surface.

Amortized spike-and-slab regression recovers coefficient estimates by
probing: the posterior-predictive-mean surface m(x) — whether from the exact
enumeration reference or from the trained model — is evaluated once at a
batch of probe points around the context distribution and ridge-regressed
onto the linear-plus-pairwise-interaction basis [1, x_j, x_j * x_k]. The
basis is `linearpfn.prior.build_design`, i.e. exactly the definition the
generator uses (one prior, one implementation). Under the exact posterior
the mean surface is exactly linear in this basis, so extraction is
consistent; the surrogate-fit R2 diagnostic flags surfaces that left the
model class.

Probe design: resample context rows with replacement,
add N(0, 0.1^2) jitter per (standardized) feature, M = 3 d points, one
batched evaluation of m, ridge with lambda = 1e-3.

Why lambda = 1e-3 and NOT a prior-matched penalty: one might argue that
coefficient directions the probes leave unidentified should be set by
prior-strength shrinkage, lambda ~ 1/tau2_eff. On the exact posterior's
surface that degrades recovery at p <= 5. The reason: probing does not fit
raw data — it reads the posterior-MEAN surface, which the posterior has already shrunk toward
zero exactly where the context is uninformative, so adding prior-strength
ridge shrinks twice. At p <= 5 the M = 3d probe cloud identifies the full
basis and lambda is pure numerical stabilization; at d >> n the jitter
gives context-null directions probe-Gram mass ~ M * jitter_sd^2 >> lambda,
so the extraction reads the surface's own shrinkage there for any small
lambda. `prior_matched_ridge_lambda` is available (ridge_lambda="prior").
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from linearpfn.prior import PriorConfig, build_design, expected_active_count, n_effects

__all__ = [
    "ProbeResult",
    "extract_coefficients",
    "prior_matched_ridge_lambda",
    "probe_points",
]

JITTER_SD = 0.1
POINTS_PER_EFFECT = 3
RIDGE_LAMBDA = 1e-3


def prior_matched_ridge_lambda(p: int, cfg: PriorConfig) -> float:
    """Ridge penalty implied by the prior's own slab: lambda = 1/tau2_eff.

    Rationale: the probe cloud spans at most ~n_context (+jitter) dimensions
    while the basis has d = 1 + p + C(p,2) columns, so at large p most
    coefficient directions are unidentified by the probes and the ridge
    penalty decides them. The exact posterior mean also reverts to zero in
    directions the context does not identify, with shrinkage set by the slab
    variance tau2_eff * sigma2 — matching lambda to 1/tau2_eff (in
    standardized units) aligns the two. tau2_eff uses the same per-k scaling
    as the prior, with k the closed-form expected active count at this p.
    """
    if cfg.slab_scaling == "per_k":
        tau2_eff = cfg.tau2 / max(expected_active_count(p, cfg), 1.0)
    else:
        tau2_eff = cfg.tau2
    return 1.0 / tau2_eff


class ProbeResult(NamedTuple):
    """Extracted coefficients plus the surrogate-fit diagnostic."""

    coef: np.ndarray  # (d,) coefficients in the canonical effect order
    r2: float  # R2 of the ridge fit to the probed surface; low values flag
    #            that the surface left the linear-plus-interaction class
    probes: np.ndarray  # (M, p) probe points that were evaluated


def probe_points(
    rng: np.random.Generator,
    X_context: np.ndarray,
    n_points: int | None = None,
    jitter_sd: float = JITTER_SD,
    cfg: PriorConfig | None = None,
) -> np.ndarray:
    """Probe locations: context rows resampled with replacement plus
    N(0, jitter_sd^2) jitter per standardized feature. Default M = 3 d,
    with d the basis size of the prior variant (mains-only: d = 1 + p)."""
    n, p = X_context.shape
    if n_points is None:
        n_points = POINTS_PER_EFFECT * n_effects(p, cfg)
    rows = rng.integers(0, n, size=n_points)
    return X_context[rows] + jitter_sd * rng.standard_normal((n_points, p))


def extract_coefficients(
    rng: np.random.Generator,
    X_context: np.ndarray,
    mean_fn: Callable[[np.ndarray], np.ndarray],
    ridge_lambda: float | str = RIDGE_LAMBDA,
    jitter_sd: float = JITTER_SD,
    n_points: int | None = None,
    cfg: PriorConfig | None = None,
) -> ProbeResult:
    """Probe `mean_fn` once at M points and ridge-regress onto the basis.

    mean_fn maps probe points (M, p) to predictive means (M,) — e.g.
    `ExactPosterior.predictive_mean` or a wrapper around the trained model's
    bar-distribution mean. ridge_lambda is either a float or the string
    "prior" (requires cfg), which uses `prior_matched_ridge_lambda`. The
    penalty applies to all coordinates including the intercept. Pass the
    PriorConfig of a mains-only run so the probing basis matches the prior's
    design (build_design with the same cfg — one prior, one implementation).
    """
    probes = probe_points(rng, X_context, n_points, jitter_sd=jitter_sd, cfg=cfg)
    m = np.asarray(mean_fn(probes), dtype=float)
    Z = build_design(probes, cfg)
    d = Z.shape[1]
    if ridge_lambda == "prior":
        if cfg is None:
            raise ValueError('ridge_lambda="prior" requires cfg')
        lam = prior_matched_ridge_lambda(X_context.shape[1], cfg)
    else:
        lam = float(ridge_lambda)
    coef = np.linalg.solve(Z.T @ Z + lam * np.eye(d), Z.T @ m)
    resid = m - Z @ coef
    total = float(((m - m.mean()) ** 2).sum())
    if total < 1e-12:  # (near-)constant surface: R2 is ill-defined
        r2 = 1.0 if float(resid @ resid) < 1e-12 else 0.0
    else:
        r2 = 1.0 - float(resid @ resid) / total
    return ProbeResult(coef=coef, r2=r2, probes=probes)
