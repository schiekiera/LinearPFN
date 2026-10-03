from __future__ import annotations

import math

import numpy as np

from linearpfn.prior import Dataset, PriorConfig, _pairs_array, build_design, prior_scale_diag

KNOBS: dict[str, str] = {
    "coef": "fixed-magnitude coefficients: |beta_e| = slab SD, random sign",
    "int": "interaction-heavy: each hereditary pair active w.p. 1/2, rho = 1",
    "noise": "Student-t(3) noise, variance-matched to sigma2",
}
PANEL_OF: dict[str, str] = {"coef": "kc", "int": "ki", "noise": "kn"}
KNOB_ID: dict[str, int] = {"coef": 1, "int": 2, "noise": 3}   # RNG stream component
INT_SHARE = 0.5      # `int`: inclusion probability of a hereditary pair
INT_RHO = 1.0        # `int`: interaction slab variance ratio
NOISE_DF = 3.0       # `noise`: Student-t degrees of freedom


def _rebuild(rng: np.random.Generator, ds: Dataset, X: np.ndarray, cfg: PriorConfig,
             gamma: np.ndarray, beta: np.ndarray, rho: float, knob: str,
             noise: np.ndarray | None = None) -> Dataset:
    n = X.shape[0]
    if noise is None:
        noise = rng.standard_normal(n)
    y = build_design(X, cfg) @ beta + math.sqrt(ds.sigma2) * noise
    return Dataset(X=X, y=y, gamma=gamma, beta=beta, sigma2=float(ds.sigma2), c=float(ds.c),
                   rho=float(rho), heredity_mode="strong")


def knob_coef(rng: np.random.Generator, ds: Dataset, X: np.ndarray, cfg: PriorConfig) -> Dataset:
    """|beta_e| = sqrt(v_e sigma2) on every active non-intercept effect, sign
    from the rng; the intercept keeps its prior draw."""
    scales = prior_scale_diag(ds.gamma, cfg, c=ds.c, rho=ds.rho)
    beta = ds.beta.copy()
    act = np.flatnonzero(ds.gamma[1:]) + 1
    signs = np.where(rng.random(act.size) < 0.5, -1.0, 1.0)
    beta[act] = signs * np.sqrt(scales[act] * ds.sigma2)
    return _rebuild(rng, ds, X, cfg, ds.gamma.copy(), beta, ds.rho, "coef")


def knob_int(rng: np.random.Generator, ds: Dataset, X: np.ndarray, cfg: PriorConfig) -> Dataset:
    """Interactions: every pair with both parents active is included w.p.
    INT_SHARE; their coefficients ~ N(0, c * INT_RHO * sigma2). Mains and the
    intercept keep the prior's gamma and beta."""
    p = X.shape[1]
    pairs = _pairs_array(p, cfg)
    mains = ds.gamma[1: 1 + p]
    elig = mains[pairs[:, 0]] & mains[pairs[:, 1]]
    ints = np.zeros(len(pairs), dtype=bool)
    if elig.any():
        ints[elig] = rng.random(int(elig.sum())) < INT_SHARE
    gamma = ds.gamma.copy()
    gamma[1 + p:] = ints
    beta = ds.beta.copy()
    beta[1 + p:] = 0.0
    scales = prior_scale_diag(gamma, cfg, c=ds.c, rho=INT_RHO)
    idx = np.flatnonzero(ints) + 1 + p
    beta[idx] = rng.standard_normal(idx.size) * np.sqrt(scales[idx] * ds.sigma2)
    return _rebuild(rng, ds, X, cfg, gamma, beta, INT_RHO, "int")


def knob_noise(rng: np.random.Generator, ds: Dataset, X: np.ndarray, cfg: PriorConfig) -> Dataset:
    """eps = sigma * t_NOISE_DF / sqrt(NOISE_DF / (NOISE_DF - 2)), so Var(eps) = sigma2."""
    n = X.shape[0]
    t = rng.standard_t(NOISE_DF, size=n) / math.sqrt(NOISE_DF / (NOISE_DF - 2.0))
    return _rebuild(rng, ds, X, cfg, ds.gamma.copy(), ds.beta.copy(), ds.rho, "noise", noise=t)


_FN = {"coef": knob_coef, "int": knob_int, "noise": knob_noise}


def apply_knob(rng: np.random.Generator, ds: Dataset, X: np.ndarray, cfg: PriorConfig,
               knob: str) -> Dataset:
    """One knob on one prior draw; raises on an unknown knob name."""
    if knob not in _FN:
        raise ValueError(f"unknown knob {knob!r}; one of {sorted(_FN)}")
    return _FN[knob](rng, ds, X, cfg)
