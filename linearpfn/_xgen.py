from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy import stats
from scipy.special import ndtr

if TYPE_CHECKING:
    from linearpfn.prior import PriorConfig

__all__ = ["MARGINAL_TYPES", "XMeta", "sample_X", "sample_correlation", "standardize"]

# Canonical order of column-marginal types; `PriorConfig.marginal_weights`
# and `marginal_probs()` follow this order everywhere.
MARGINAL_TYPES = ("gaussian", "likert", "binary", "count", "skewed")


class XMeta(NamedTuple):
    """Ground-truth structure of one X draw, for diagnostics and validation
    strata. Per-column arrays are in FINAL (post-shuffle) column order
    except h2 and cross_w, which are pooled-distribution diagnostics in
    pre-shuffle block order."""

    latent: np.ndarray  # (n, p) latent Gaussian columns before the copula
    column_types: np.ndarray  # (p,) marginal type name per column
    is_factor: np.ndarray  # (p,) bool: column belongs to the factor block
    f: float  # drawn factor share (0.0 = pure generic)
    p_fac: int  # factor-block size round(f * p)
    m: int  # number of factors (0 when p_fac == 0)
    h2: np.ndarray  # (p_fac,) communalities
    cross_w: np.ndarray  # (p_gen,) signed cross-block loadings
    h2_high: bool  # factor block drew the high-communality (near-duplicate) regime


def standardize(X: np.ndarray) -> np.ndarray:
    """Z-score each column with ddof=0 (population sd)."""
    return (X - X.mean(axis=0)) / X.std(axis=0)


def sample_correlation(rng: np.random.Generator, p: int) -> np.ndarray:
    """Random correlation matrix: eigenvalues p * Dirichlet(1_p), then scipy's
    `random_correlation` (Davies & Higham Givens-rotation construction).

    Tiny eigenvalues are valid (a near-singular R is fine — columns remain
    non-degenerate a.s.); the rare numerical failure is retried with fresh
    eigenvalues from the same rng, preserving determinism given the seed.
    """
    if p == 1:
        return np.ones((1, 1))
    last_err: Exception | None = None
    for _ in range(8):
        eigs = p * rng.dirichlet(np.ones(p))
        eigs *= p / eigs.sum()  # guard float drift; must sum to exactly p
        try:
            return stats.random_correlation.rvs(eigs, random_state=rng)
        except (ValueError, RuntimeError) as err:
            last_err = err
    raise RuntimeError(f"random_correlation failed after 8 attempts (p={p})") from last_err


def _generic_scores(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    """The generic latent mechanism: rows iid N(0, R) with R from
    `sample_correlation`; Cholesky with eigen fallback for near-singular R."""
    R = sample_correlation(rng, p)
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        # Near-singular R (tiny Dirichlet eigenvalue): eigen factor L L' = R.
        w, Q = np.linalg.eigh(R)
        L = Q * np.sqrt(np.clip(w, 0.0, None))
    return rng.standard_normal((n, p)) @ L.T


def _factor_block(
    rng: np.random.Generator, n: int, p_fac: int, cfg: PriorConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    """Factor-analytic latent block: (block (n, p_fac), F (n, m), h2, m,
    high) where `high` flags the high-communality regime."""
    m_max = max(1, math.ceil(p_fac * cfg.factor_count_frac_max))
    m = int(rng.integers(1, m_max + 1))
    primary = rng.integers(0, m, size=p_fac)
    # Communality regime, drawn per dataset. Two mixture components: the
    # standard band, and (with h2_high_prob) a HIGH band whose blocks are
    # near-duplicate column groups, the collinearity regime real designs
    # occupy routinely and the standard band alone reaches in ~0.8% of p=5
    # draws. Gated on the config BEFORE the rng is touched, so
    # h2_high_prob = 0.0 consumes no extra random numbers.
    if cfg.h2_high_prob <= 0.0:
        high = False
    else:
        high = bool(rng.random() < cfg.h2_high_prob)
    if high:
        mean_lo, mean_hi, cap = cfg.h2_high_mean_low, cfg.h2_high_cap, cfg.h2_high_cap
    else:
        mean_lo, mean_hi, cap = cfg.h2_mean_low, cfg.h2_mean_high, cfg.h2_cap
    h2_mean = rng.uniform(mean_lo, mean_hi)
    conc = rng.uniform(cfg.h2_concentration_low, cfg.h2_concentration_high)
    h2 = np.minimum(rng.beta(h2_mean * conc, (1.0 - h2_mean) * conc, size=p_fac), cap)
    lam = np.zeros((p_fac, m))
    rows = np.arange(p_fac)
    if m == 1:
        # One factor: fold the cross-loading remainder back into the primary,
        # keeping sum_f lam^2 = h2 (and a 1-column block exactly N(0, 1)).
        lam[:, 0] = np.sqrt(h2)
    else:
        lam[rows, primary] = np.sqrt((1.0 - cfg.cross_loading_share) * h2)
        split = rng.dirichlet(np.ones(m - 1), size=p_fac)  # (p_fac, m-1)
        cross = np.sqrt(cfg.cross_loading_share * h2[:, None] * split)
        mask = np.ones((p_fac, m), dtype=bool)
        mask[rows, primary] = False
        lam[mask] = cross.ravel()  # row-major: fills each row's m-1 free slots
    flip = rng.random((p_fac, m)) < cfg.neg_loading_prob
    lam = np.where(flip, -lam, lam)
    F = rng.standard_normal((n, m))
    E = rng.standard_normal((n, p_fac))
    block = F @ lam.T + np.sqrt(1.0 - h2) * E
    return block, F, h2, m, high


def _apply_marginal(
    rng: np.random.Generator,
    latent_col: np.ndarray,
    u_col: np.ndarray,
    type_index: int,
    cfg: PriorConfig,
) -> np.ndarray:
    """One column's NORTA transform; draws the column's own parameters."""
    kind = MARGINAL_TYPES[type_index]
    if kind == "gaussian":
        return latent_col
    if kind == "likert":
        levels = int(rng.integers(cfg.likert_levels_min, cfg.likert_levels_max + 1))
        return np.floor(u_col * levels)  # equal-probability bins, values 0..levels-1
    if kind == "binary":
        rate = rng.uniform(cfg.binary_rate_low, cfg.binary_rate_high)
        return (u_col >= 1.0 - rate).astype(float)
    if kind == "count":
        poisson = rng.random() < 0.5
        lam = math.exp(rng.uniform(math.log(0.5), math.log(10.0)))
        if poisson:
            return stats.poisson.ppf(u_col, lam)
        r = rng.uniform(1.0, 5.0)
        return stats.nbinom.ppf(u_col, r, r / (r + lam))  # mean lam, dispersion r
    if kind == "skewed":
        lognormal = rng.random() < 0.5
        if lognormal:
            s = rng.uniform(0.4, 1.0)
            return stats.lognorm.ppf(u_col, s)
        return stats.expon.ppf(u_col)
    raise ValueError(f"unknown marginal type index {type_index}")


def _copula_column(
    rng: np.random.Generator,
    latent_col: np.ndarray,
    u_col: np.ndarray,
    type_index: int,
    cfg: PriorConfig,
    probs: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Transform one column, redrawing type + parameters on degeneracy.

    The latent column (and hence the copula structure) is never redrawn;
    only the marginal side is. 8 retries, then RuntimeError."""
    t = type_index
    for _ in range(9):  # initial draw + 8 retries
        col = _apply_marginal(rng, latent_col, u_col, t, cfg)
        if float(col.std()) >= 1e-8:
            return col, t
        t = int(rng.choice(len(MARGINAL_TYPES), p=probs))
    raise RuntimeError(
        f"marginal transform degenerate after 8 redraws (n={latent_col.size})"
    )


def sample_X(
    rng: np.random.Generator, n: int, p: int, cfg: PriorConfig
) -> tuple[np.ndarray, XMeta]:
    """Draw one standardized (ddof=0) feature matrix; see module docstring.

    Every stage short-circuits on the config BEFORE touching the rng, so a
    disabled stage consumes no random numbers."""
    if cfg.f_zero_prob >= 1.0:
        f = 0.0  # config-gated: no rng touched
    elif rng.random() < cfg.f_zero_prob:
        f = 0.0
    else:
        f = float(rng.beta(cfg.f_beta_a, cfg.f_beta_b))
    p_fac = int(round(f * p))
    p_gen = p - p_fac

    gen = np.empty((n, 0))
    fac = np.empty((n, 0))
    F = np.empty((n, 0))
    h2 = np.zeros(0)
    m = 0
    h2_high = False
    if p_gen:
        gen = _generic_scores(rng, n, p_gen)
    if p_fac:
        fac, F, h2, m, h2_high = _factor_block(rng, n, p_fac, cfg)
    cross_w = np.zeros(p_gen)
    if p_fac and p_gen:
        w_abs = rng.uniform(0.0, cfg.cross_block_w_max, size=p_gen)
        sign = np.where(rng.random(p_gen) < 0.5, -1.0, 1.0)
        U = rng.standard_normal((m, p_gen))
        U /= np.linalg.norm(U, axis=0)  # random unit vector per generic column
        cross_w = sign * w_abs
        gen = np.sqrt(1.0 - cross_w**2) * gen + (F @ U) * cross_w

    latent = np.concatenate([gen, fac], axis=1)
    is_factor = np.concatenate([np.zeros(p_gen, bool), np.ones(p_fac, bool)])
    if cfg.f_zero_prob < 1.0:  # config-gated with the factor mechanism itself
        perm = rng.permutation(p)
        latent = latent[:, perm]
        is_factor = is_factor[perm]

    if cfg.is_gaussian_only():
        X = latent  # config-gated: whole copula stage skipped
        types = np.full(p, "gaussian")
    else:
        probs = cfg.marginal_probs()
        type_idx = rng.choice(len(MARGINAL_TYPES), size=p, p=probs)
        u = ndtr(latent)
        X = np.empty_like(latent)
        names: list[str] = []
        for j in range(p):
            X[:, j], t = _copula_column(rng, latent[:, j], u[:, j], int(type_idx[j]), cfg, probs)
            names.append(MARGINAL_TYPES[t])
        types = np.asarray(names)

    meta = XMeta(
        latent=latent,
        column_types=types,
        is_factor=is_factor,
        f=f,
        p_fac=p_fac,
        m=m,
        h2=h2,
        cross_w=cross_w,
        h2_high=h2_high,
    )
    return standardize(X), meta
