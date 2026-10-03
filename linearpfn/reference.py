"""Exact enumeration Bayesian-model-averaging reference for the LinearPFN prior.

numpy/scipy only — no neural network. This is the ground truth that the
amortized spike-and-slab regression model is validated against: because the
slab is Gaussian scaled by sigma2 and sigma2 is inverse-gamma, the marginal
likelihood of every heredity-valid active set gamma is available in closed
form, and for p <= `MAX_ENUM_P` the model space is small enough to enumerate
exactly (sum_{a=1..p} C(p,a) 2^{C(a,2)} non-empty models; 1449 at p = 5).
Under the mains-only prior variant the model space is the 2^p - 1 non-empty
main subsets, so enumeration reaches p <= `MAX_ENUM_P_MAINS_ONLY` — that
reach is the variant's purpose (exact validation far beyond p = 5).

Conjugacy. For a fixed gamma with design Z (intercept + active effects):

    y | beta, sigma2 ~ N(Z beta, sigma2 I_n)
    beta | sigma2    ~ N(0, sigma2 V0),
    V0 = diag(tau2_intercept, tau2_eff, ..., tau2_eff)
    sigma2           ~ InvGamma(a0, b0)

with tau2_eff the model's per-effect slab variance from
`linearpfn.prior.slab_variance` — a constant within each model (tau2 under
slab_scaling="fixed"; tau2 / k under "per_k" with k the model's active
non-intercept count), so conjugacy holds unchanged in both modes.

    A = V0^{-1} + Z'Z,   V_n = A^{-1},   m_n = V_n Z'y,
    a_n = a0 + n/2,      b_n = b0 + (y'y - m_n'A m_n)/2 = b0 + (y'y - m_n.Z'y)/2
    log p(y|gamma) = -(n/2) log(2 pi) + (log|V_n| - log|V0|)/2
                     + a0 log b0 - a_n log b_n + lnG(a_n) - lnG(a0)

Positivity of b_n - b0 (Woodbury): I - Z V_n Z' = (I + Z V0 Z')^{-1} is
positive definite, so y'y - m_n'A m_n = y'(I + Z V0 Z')^{-1} y > 0. The
evidence equals the density of y ~ Student-t_{2 a0}(0, (b0/a0)(I + Z V0 Z')),
which serves as an independent check of the closed form.

Quadrature (random slab scale). When the slab scale c (and the interaction variance
ratio rho) are random, the per-model evidence marginalizes over them on a
Gauss-Legendre grid in log space (`linearpfn._quadrature`); the posterior
is then a discrete distribution over (model, node) COMPONENTS. Model-level
weights and PIPs are computed over ALL models before any pruning;
coefficient moments and predictives run over the components that carry
1 - comp_prune_tol of the mass (capped, loud failure if the cap cannot
reach 1 - 5e-4, half the ~1e-3 metric precision).
With the random gates off, every axis is a single node
with mass one and the machinery reduces exactly to the fixed-(c, rho)
computation below — component i IS model i.

Predictives. Per component at query row z*:

    y* | y, comp ~ t_{2 a_n}( z*'m_n, (b_n/a_n)(1 + z*'V_n z*) )

The exact posterior predictive is the weight mixture of these Student-t
components — they share df = 2 a_n because a_n depends on neither gamma nor
(c, rho). Mixture quantiles are computed by bisection on the mixture CDF,
never by a normal approximation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp, stdtr, stdtrit

from linearpfn._quadrature import evidence_pass, moment_pass, node_grid
from linearpfn.prior import (
    PriorConfig,
    build_design,
    log_prior_gamma,
    n_effects,
    slab_variance,
)

__all__ = [
    "MAX_ENUM_P",
    "MAX_ENUM_P_MAINS_ONLY",
    "QUAD_NODES_C",
    "QUAD_NODES_RHO",
    "ExactPosterior",
    "enumerate_models",
    "fit_exact",
    "max_enum_p",
]

MAX_ENUM_P = 5
# Mains-only variant: the model space is all 2^p main subsets (2^p - 1 when
# the prior rejects the empty set), so enumeration reaches much larger p.
# The cap is set by the measured wall-clock and memory of the batched fit.
MAX_ENUM_P_MAINS_ONLY = 20

# Quadrature node counts under random_c / random_rho_int. Convergence check:
# scripts/38_compute_quadrature.py (20 datasets/cell vs a 30x15 truth
# grid; the stage writes the full table next to its results.json):
# at (20, 11), max |dPIP| <= 1.3e-8 in every cell (p in {3,5}, n in
# {32, 1024}) and max per-model |dlogev| <= 1.1e-7 except the n=1024/p=5
# cell's 1.6e-5 — one weight-negligible model out of 32768 (that fit's own
# dPIP is 1.3e-8), rho-axis limited and saturated in c. The mains-only
# 1-D c rule at 20 nodes measures dlogev <= 4.8e-7 at p in {10, 15}.
# Overridable per call via fit_exact(n_nodes_c=..., n_nodes_rho=...).
QUAD_NODES_C = 20
QUAD_NODES_RHO = 11


def max_enum_p(cfg: PriorConfig) -> int:
    """Largest p the exact enumeration reference supports under this prior."""
    return MAX_ENUM_P if cfg.include_interactions else MAX_ENUM_P_MAINS_ONLY


def enumerate_models(p: int, cfg: PriorConfig) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate all prior-supported active sets for p <= max_enum_p.

    Builds ALL 2^(d-1) intercept-anchored bit patterns vectorized, evaluates
    the exact log prior on the stack, and keeps the rows with finite mass —
    the support is whatever `log_prior_gamma` says it is, so enumeration can
    never disagree with the prior. Under strong heredity with the empty set
    rejected the filter yields 1449 models at p = 5 (2^p - 1 mains-only);
    under the heredity mixture every pattern survives, including the
    intercept-only model. Returns the boolean model
    matrix (M, d) and the exact log prior (M,).
    """
    cap = max_enum_p(cfg)
    if not 1 <= p <= cap:
        raise ValueError(f"exact enumeration supports 1 <= p <= {cap}, got p={p}")
    d = n_effects(p, cfg)
    bits = np.arange(2 ** (d - 1), dtype=np.int64)
    patterns = (bits[:, None] >> np.arange(d - 1)) & 1
    gammas = np.concatenate(
        [np.ones((bits.size, 1), dtype=bool), patterns.astype(bool)], axis=1
    )
    log_prior = np.asarray(log_prior_gamma(gammas, p, cfg))
    keep = np.isfinite(log_prior)
    return gammas[keep], log_prior[keep]


def _design_stats(Z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Sufficient statistics for evidence evaluation: (Z'Z, Z'y, y'y, n).

    Shared with the MC3 sampler (linearpfn.mc3), which precomputes these
    once per dataset and evaluates thousands of single-model evidences.
    """
    return Z.T @ Z, Z.T @ y, float(y @ y), Z.shape[0]


def _fit_models_from_stats(
    stats: tuple[np.ndarray, np.ndarray, float, int],
    gammas: np.ndarray,
    cfg: PriorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched single-(c, rho) NIG fit of every model, grouped by size.

    The FIXED-scale fit (c = tau2, rho = rho_int_fixed): raises under
    random_c/random_rho_int via `slab_variance`, which is correct — its only
    consumers are the MC3 sampler (guarded to fixed-scale priors) and the
    single-node path. Gathers per-model submatrices of the
    precomputed Gram matrix with fancy indexing and runs stacked
    Cholesky/solve per size group; interaction slots (canonical index > p)
    get the rho-scaled slab. A = V0^{-1} + Z'Z is positive definite by
    construction, so Cholesky never needs jitter.
    """
    G, c, yty, n = stats
    m_count, d = gammas.shape
    p = _p_from_shape(d, cfg)
    a_n = cfg.a0 + n / 2.0
    const = (
        -0.5 * n * math.log(2.0 * math.pi)
        + cfg.a0 * math.log(cfg.b0)
        + gammaln(a_n)
        - gammaln(cfg.a0)
    )
    log_ev = np.empty(m_count)
    b_n = np.empty(m_count)
    m_full = np.zeros((m_count, d))
    v_full = np.zeros((m_count, d, d))
    sizes = gammas.sum(axis=1)
    for s in np.unique(sizes):
        rows = np.flatnonzero(sizes == s)
        idx = np.nonzero(gammas[rows])[1].reshape(len(rows), s)
        tau2_eff = slab_variance(int(s) - 1, cfg)  # intercept is not a selected effect
        tau2_int = slab_variance(int(s) - 1, cfg, is_interaction=True)
        v0_sub = np.where(
            idx == 0, cfg.tau2_intercept, np.where(idx > p, tau2_int, tau2_eff)
        )
        a_mat = G[idx[:, :, None], idx[:, None, :]].copy()
        ar = np.arange(s)
        a_mat[:, ar, ar] += 1.0 / v0_sub
        chol = np.linalg.cholesky(a_mat)
        logdet_vn = -2.0 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
        csub = c[idx]
        m_n = np.linalg.solve(a_mat, csub[..., None])[..., 0]
        quad = np.einsum("ms,ms->m", m_n, csub)
        b_group = cfg.b0 + 0.5 * (yty - quad)
        log_v0 = np.log(v0_sub).sum(axis=1)
        log_ev[rows] = const + 0.5 * (logdet_vn - log_v0) - a_n * np.log(b_group)
        b_n[rows] = b_group
        v_n = np.linalg.inv(a_mat)
        m_full[rows[:, None], idx] = m_n
        v_full[rows[:, None, None], idx[:, :, None], idx[:, None, :]] = v_n
    return log_ev, b_n, m_full, v_full


def _p_from_shape(d: int, cfg: PriorConfig) -> int:
    """p implied by the effect-vector length d (exact inversion)."""
    if not cfg.include_interactions:
        return d - 1
    return (math.isqrt(1 + 8 * (d - 1)) - 1) // 2


@dataclass(frozen=True, eq=False)
class ExactPosterior:
    """Exact posterior over an explicit model space for one dataset.

    All quantities are exact under the LinearPFN prior: model probabilities,
    per-effect posterior inclusion probabilities, spike-and-slab coefficient
    moments, and the Student-t mixture posterior predictive.

    Model-level arrays (gammas, log_prior, log_evidence, log_weights) cover
    ALL M models with (c, rho) marginalized out — PIPs are never pruned.
    Component-level arrays (b_n, m_full, v_full, comp_*) cover the C kept
    (model, quadrature-node) components carrying comp_mass of the posterior;
    with the random scales off, C == M in model-major order and component i
    IS model i.
    """

    gammas: np.ndarray  # (M, d) bool model matrix
    log_prior: np.ndarray  # (M,)
    log_evidence: np.ndarray  # (M,) c/rho-marginalized
    log_weights: np.ndarray  # (M,) normalized log posterior model probabilities
    a_n: float  # shared posterior shape; predictive df = 2 a_n
    b_n: np.ndarray  # (C,) per-component posterior scales
    m_full: np.ndarray  # (C, d) component means scattered to the full basis
    v_full: np.ndarray  # (C, d, d) component covariance factors (dense scatter)
    p: int
    cfg: PriorConfig
    comp_model: np.ndarray  # (C,) model index of each component
    comp_log_weight: np.ndarray  # (C,) joint (model, node) log weights (same norm)
    comp_mass: float  # total posterior mass the kept components carry

    @classmethod
    def from_single_node(
        cls,
        gammas: np.ndarray,
        log_prior: np.ndarray,
        log_evidence: np.ndarray,
        log_weights: np.ndarray,
        a_n: float,
        b_n: np.ndarray,
        m_full: np.ndarray,
        v_full: np.ndarray,
        p: int,
        cfg: PriorConfig,
    ) -> ExactPosterior:
        """Posterior at a single fixed (c, rho): every model is its own
        component in model-major order (MC3 and fixed-model fits)."""
        return cls(
            gammas=gammas,
            log_prior=log_prior,
            log_evidence=log_evidence,
            log_weights=log_weights,
            a_n=a_n,
            b_n=b_n,
            m_full=m_full,
            v_full=v_full,
            p=p,
            cfg=cfg,
            comp_model=np.arange(len(gammas)),
            comp_log_weight=log_weights,
            comp_mass=1.0,
        )

    def weights(self) -> np.ndarray:
        """Exact posterior model probabilities."""
        return np.exp(self.log_weights)

    def pip(self) -> np.ndarray:
        """Exact posterior inclusion probability per effect (intercept = 1).

        Computed from the model-level weights over ALL models — pruning
        never touches this."""
        return self.weights() @ self.gammas

    def _comp_weights(self) -> np.ndarray:
        """Normalized component weights (kept components sum to 1)."""
        w = np.exp(self.comp_log_weight)
        return w / w.sum()

    def coef_mean(self) -> np.ndarray:
        """Exact posterior mean of every coefficient (spike-and-slab mixture)."""
        return self._comp_weights() @ self.m_full

    def coef_var(self) -> np.ndarray:
        """Exact posterior variance of every coefficient.

        Mixes the within-component marginal-t variance (b_n/(a_n-1)) V_n[jj]
        with the point mass at zero from components excluding the effect.
        """
        w = self._comp_weights()
        diag_v = np.diagonal(self.v_full, axis1=1, axis2=2)
        second = w @ (self.b_n[:, None] / (self.a_n - 1.0) * diag_v + self.m_full**2)
        return second - self.coef_mean() ** 2

    # ---- Student-t mixture predictive ------------------------------------

    def _components(
        self, Xq: np.ndarray, idx: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mixture components at query rows: (loc, scale), each (nq, C').

        Query rows are taken in the same standardized coordinates as the
        context X the posterior was fitted on; the basis is rebuilt with the
        shared `build_design`. Common df = 2 a_n. `idx` restricts to a
        component subset (the weight-pruned prefix) BEFORE any per-component
        work — at mains-only large p computing components for all of them
        dominates everything else.
        """
        zq = build_design(np.asarray(Xq, dtype=float), self.cfg)
        m = self.m_full if idx is None else self.m_full[idx]
        v = self.v_full if idx is None else self.v_full[idx]
        b = self.b_n if idx is None else self.b_n[idx]
        loc = zq @ m.T
        quad = np.einsum("qi,mij,qj->qm", zq, v, zq)
        scale = np.sqrt(b / self.a_n * (1.0 + quad))
        return loc, scale

    def _pruned(self, prune_tol: float) -> tuple[np.ndarray, np.ndarray]:
        """Indices + renormalized weights of the smallest weight-sorted
        COMPONENT prefix reaching mass 1 - prune_tol (posteriors concentrate
        on few components; this keeps predictive evaluations cheap)."""
        w = self._comp_weights()
        if prune_tol <= 0.0:
            return np.arange(w.size), w
        order = np.argsort(w)[::-1]
        keep = int(np.searchsorted(np.cumsum(w[order]), 1.0 - prune_tol)) + 1
        idx = order[: min(keep, w.size)]
        wk = w[idx]
        return idx, wk / wk.sum()

    def predictive_mean(self, Xq: np.ndarray) -> np.ndarray:
        """Exact posterior predictive mean at query rows (loc only — no
        per-component covariance work, so it stays cheap at ~2^p spaces)."""
        zq = build_design(np.asarray(Xq, dtype=float), self.cfg)
        return (zq @ self.m_full.T) @ self._comp_weights()

    def predictive_var(self, Xq: np.ndarray) -> np.ndarray:
        """Exact posterior predictive variance (second-moment mixing)."""
        w = self._comp_weights()
        loc, scale = self._components(Xq)
        comp_var = scale**2 * (self.a_n / (self.a_n - 1.0))  # t: df/(df-2) scale^2
        mean = loc @ w
        return (comp_var + loc**2) @ w - mean**2

    def predictive_cdf(
        self, Xq: np.ndarray, yq: np.ndarray, prune_tol: float = 1e-12
    ) -> np.ndarray:
        """Exact posterior predictive CDF evaluated at (query row, value) pairs."""
        idx, w = self._pruned(prune_tol)
        loc, scale = self._components(Xq, idx)
        z = (np.asarray(yq, dtype=float)[:, None] - loc) / scale
        return stdtr(2.0 * self.a_n, z) @ w

    def predictive_logpdf(
        self, Xq: np.ndarray, yq: np.ndarray, prune_tol: float = 1e-12
    ) -> np.ndarray:
        """Exact posterior predictive log density at (query row, value) pairs.

        This is the exact-BMA NLL reference of the validation's predictive
        criterion.
        """
        idx, w = self._pruned(prune_tol)
        loc, scale = self._components(Xq, idx)
        df = 2.0 * self.a_n
        z = (np.asarray(yq, dtype=float)[:, None] - loc) / scale
        log_norm = gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0) - 0.5 * math.log(df * math.pi)
        log_comp = log_norm - 0.5 * (df + 1.0) * np.log1p(z**2 / df) - np.log(scale)
        return logsumexp(log_comp + np.log(w), axis=1)

    def predictive_quantiles(
        self, Xq: np.ndarray, levels: Sequence[float], prune_tol: float = 1e-12
    ) -> np.ndarray:
        """Exact predictive quantiles of the Student-t mixture, shape (nq, L).

        Bracket lemma: with F = sum_g w_g F_g and t_lo = min_g F_g^{-1}(q),
        F(t_lo) = sum_g w_g F_g(t_lo) <= sum_g w_g q = q; symmetrically
        F(t_hi) >= q at t_hi = max_g F_g^{-1}(q). The mixture quantile
        therefore lies in [t_lo, t_hi]; 100 bisection iterations shrink the
        bracket to floating-point resolution (measured round-trip ~4e-16).
        """
        idx, w = self._pruned(prune_tol)
        loc, scale = self._components(Xq, idx)
        df = 2.0 * self.a_n
        levels_arr = np.asarray(list(levels), dtype=float)
        tq = stdtrit(df, levels_arr)  # (L,)
        comp_q = loc[:, :, None] + scale[:, :, None] * tq  # (nq, M, L)
        lo = comp_q.min(axis=1)
        hi = comp_q.max(axis=1)
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            z = (mid[:, None, :] - loc[:, :, None]) / scale[:, :, None]
            cdf = np.einsum("qml,m->ql", stdtr(df, z), w)
            above = cdf >= levels_arr
            hi = np.where(above, mid, hi)
            lo = np.where(above, lo, mid)
        return 0.5 * (lo + hi)


def fit_exact(
    X: np.ndarray,
    y: np.ndarray,
    cfg: PriorConfig,
    models: tuple[np.ndarray, np.ndarray] | None = None,
    standardization_tol: float = 1e-6,
    n_nodes_c: int | None = None,
    n_nodes_rho: int | None = None,
    comp_prune_tol: float = 1e-9,
    comp_cap: int = 5_000_000,
) -> ExactPosterior:
    """Exact BMA posterior for one dataset under the LinearPFN prior.

    X must be z-standardized (ddof=0) in the same way the generator
    standardizes features — asserted rather than silently re-standardized so
    the "one prior, one implementation" contract cannot drift. When fitting
    on a context SLICE of a standardized dataset (whose own column stats
    deviate from 0/1 by O(1/sqrt(n))), pass a looser `standardization_tol`;
    the guard then still catches genuinely raw data. `models` overrides the
    enumerated model space with explicit (gammas, log_prior) arrays, e.g.
    prior mass 1 on a single fixed gamma.

    Under random_c / random_rho_int the evidence marginalizes over the
    (c, rho) quadrature grid (n_nodes_c x n_nodes_rho, defaulting to the
    module constants QUAD_NODES_*); model weights and PIPs are computed over
    everything, while coefficient/predictive moments keep the smallest set
    of (model, node) components carrying 1 - comp_prune_tol posterior mass,
    capped at comp_cap. If the cap cannot reach 1 - 5e-4 mass (half the
    ~1e-3 metric precision) the fit RAISES — loud beats silently biased.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    if y.shape != (n,):
        raise ValueError(f"y must have shape ({n},), got {y.shape}")
    tol = standardization_tol
    if np.abs(X.mean(axis=0)).max() > tol or np.abs(X.std(axis=0) - 1.0).max() > tol:
        raise ValueError(
            "X must be z-standardized (ddof=0); the reference shares the "
            "generator's standardization contract"
        )
    if models is None:
        gammas, log_prior = enumerate_models(p, cfg)
    else:
        gammas = np.asarray(models[0], dtype=bool)
        log_prior = np.asarray(models[1], dtype=float)
    if not gammas[:, 0].all():
        raise ValueError("every model must include the intercept (gamma[:, 0] all True)")
    stats = _design_stats(build_design(X, cfg), y)
    c_nodes, c_logmass, rho_nodes, rho_logmass = node_grid(
        cfg, n_nodes_c or QUAD_NODES_C, n_nodes_rho or QUAD_NODES_RHO
    )
    log_ev, cand_model, cand_c, cand_rho, cand_logjoint, cand_bn = evidence_pass(
        stats, gammas, log_prior, cfg, p, c_nodes, c_logmass, rho_nodes, rho_logmass
    )
    norm = logsumexp(cand_logjoint)
    log_weights = log_prior + log_ev - norm
    comp_logw_all = cand_logjoint - norm
    order = np.argsort(comp_logw_all)[::-1]
    csum = np.cumsum(np.exp(comp_logw_all[order]))
    if cand_logjoint.size == len(gammas):
        # Single-node grid (both random gates off): keep every model, so
        # component i IS model i, no pruning at all.
        keep_n = order.size
    else:
        keep_n = min(int(np.searchsorted(csum, 1.0 - comp_prune_tol)) + 1, order.size)
        if keep_n > comp_cap:
            keep_n = comp_cap
            # Sizing: a diffuse near-null p=5 draw spreads mass so widely
            # that 1 - 1e-4 needs ~43% of the 20x11 grid's 7.2M (model, node)
            # candidates (1.6M reach only 1 - 5.5e-4). The 5M cap gives ~50x
            # margin to the floor on that worst case; dense covariances take ~10 GB
            # at d=16, so the full validation panel needs a large-memory
            # machine. The floor sits at half the ~1e-3
            # metric precision. Model weights and PIPs are computed
            # pre-pruning and stay exact regardless; comp_mass records the
            # kept fraction.
            if csum[keep_n - 1] < 1.0 - 5e-4:
                raise RuntimeError(
                    f"comp_cap={comp_cap} components reach only {csum[keep_n - 1]:.9f} "
                    "posterior mass (< 1 - 5e-4); raise comp_cap or comp_prune_tol"
                )
    kept = np.sort(order[:keep_n])  # (model, c, rho) lex order; K=1 => model-major
    b_n, m_full, v_full = moment_pass(
        stats, gammas, cfg, p, cand_model[kept], cand_c[kept], cand_rho[kept],
        c_nodes, rho_nodes,
    )
    return ExactPosterior(
        gammas=gammas,
        log_prior=log_prior,
        log_evidence=log_ev,
        log_weights=log_weights,
        a_n=cfg.a0 + n / 2.0,
        b_n=b_n,
        m_full=m_full,
        v_full=v_full,
        p=p,
        cfg=cfg,
        comp_model=cand_model[kept],
        comp_log_weight=comp_logw_all[kept],
        comp_mass=float(min(csum[keep_n - 1], 1.0)),
    )
