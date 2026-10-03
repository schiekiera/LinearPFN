"""Zero-inflated truncated-geometric count law (private helper of prior).

The count prior draws HOW MANY effects are active directly, instead of a
per-candidate rate: P(k = 0) = zero_prob; conditional on k >= 1, k
follows a geometric law with untruncated conditional mean `mean`
(success probability q = 1/mean), truncated to {1, ..., m} and
renormalized. m = 0 collapses all mass onto k = 0. The active subset is
uniform given the count, so log P(specific vector) =
logpmf(k) - log C(m, k).

Substantive content: the number of real causes of a phenomenon is small and
does NOT scale with the number of variables a researcher happens to measure;
the rate prior asserts the opposite (E[#actives] ~ p, E[#interactions] ~ C(p, 2)).

One implementation serves the sampler (`zt_geom_draw`), the exact reference
(`zt_geom_logpmf`) and the closed-form expectation (`zt_geom_mean`).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

__all__ = ["zt_geom_logpmf", "zt_geom_draw", "zt_geom_mean", "log_choose"]


def log_choose(m: np.ndarray | int, k: np.ndarray | int) -> np.ndarray:
    """log C(m, k), vectorized; 0 for the (0, 0) placeholder rows."""
    m_arr = np.asarray(m, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    return gammaln(m_arr + 1.0) - gammaln(k_arr + 1.0) - gammaln(m_arr - k_arr + 1.0)


def zt_geom_logpmf(
    k: np.ndarray | int,
    m: np.ndarray | int,
    zero_prob: float,
    mean: float,
) -> np.ndarray:
    """log P(k) under the zero-inflated geometric truncated to {0, ..., m}.

    Vectorized over k and m (broadcast). Out-of-support k (k < 0, k > m,
    or non-zero k when m = 0) gets -inf. mean = 1 is the degenerate
    point mass at k = 1 (given non-zero)."""
    k_arr = np.asarray(k, dtype=float)
    m_arr = np.asarray(m, dtype=float)
    k_b, m_b = np.broadcast_arrays(k_arr, m_arr)
    out = np.full(k_b.shape, -np.inf)
    m_zero = m_b <= 0.0
    out = np.where(m_zero & (k_b == 0.0), 0.0, out)
    if zero_prob > 0.0:
        out = np.where(~m_zero & (k_b == 0.0), math.log(zero_prob), out)
    pos = ~m_zero & (k_b >= 1.0) & (k_b <= m_b)
    q = 1.0 / mean
    if q >= 1.0:  # point mass at k = 1 given non-zero
        val = np.where(k_b == 1.0, math.log1p(-zero_prob), -np.inf)
    else:
        log_omq = math.log1p(-q)
        m_safe = np.maximum(m_b, 1.0)
        log_norm = np.log1p(-np.exp(m_safe * log_omq))  # log(1 - (1-q)^m)
        val = (math.log1p(-zero_prob) + (k_b - 1.0) * log_omq
               + math.log(q) - log_norm)
    return np.where(pos, val, out)


def zt_geom_draw(
    rng: np.random.Generator,
    m: int,
    zero_prob: float,
    mean: float,
) -> int:
    """One draw by inverse CDF. Always consumes exactly ONE rng.random()
    (also when m = 0), so the stream shape does not depend on the data."""
    u = float(rng.random())
    if m <= 0 or u < zero_prob:
        return 0
    q = 1.0 / mean
    if q >= 1.0:
        return 1
    v = (u - zero_prob) / (1.0 - zero_prob)
    log_omq = math.log1p(-q)
    trunc_mass = -math.expm1(m * log_omq)  # 1 - (1-q)^m
    k = math.ceil(math.log1p(-v * trunc_mass) / log_omq)
    return int(min(max(k, 1), m))


def zt_geom_mean(m: int, zero_prob: float, mean: float) -> float:
    """E[k] in closed form: (1-z) * [1 - (1-q)^m (1 + m q)] / (q (1 - (1-q)^m))."""
    if m <= 0:
        return 0.0
    q = 1.0 / mean
    if q >= 1.0:
        return 1.0 - zero_prob
    omq_m = (1.0 - q) ** m
    return (1.0 - zero_prob) * (1.0 - omq_m * (1.0 + m * q)) / (q * (1.0 - omq_m))
