"""Gauss-Legendre quadrature over (c, rho) for the exact reference.

Private helper of `linearpfn.reference`. When the prior draws the slab
scale c (and the interaction variance ratio rho) per
dataset from truncated lognormals, the per-model evidence becomes

    p(y | X, gamma) = ∫∫ p(y | X, gamma, c, rho) p(c) p(rho) dc drho

with the inner term the closed-form NIG evidence at a fixed diagonal V0.
The integral is discretized on Gauss-Legendre nodes over log c (and a
tensor grid over log rho for the interactions variant); node masses are
quadrature weight x Jacobian x `trunc_lognormal_logpdf` — the SAME density
the sampler draws from (one law, one implementation) — and are
renormalized so each axis' masses sum to exactly one. Renormalizing keeps
the discrete (model, node) posterior an exact probability distribution and
absorbs the O(quadrature error) mass defect; with a single node the mass
is exactly 1, so the fixed-c/fixed-rho configuration is LITERALLY the
single-node computation of the same code path.

Two passes over the model space (memory is the binding constraint — dense
per-(model, node) covariance storage would be (M K, d, d)):

- `evidence_pass`: per size group, gather the Gram submatrices ONCE and
  loop over nodes adding diag(1/V0(node)); emits per-model marginalized
  log evidence (model-level weights and PIPs are therefore NEVER pruned)
  plus the flat candidate arrays of per-(model, node) joint log weights
  and b_n. Models with no active interaction are rho-inert: their
  evidence is constant along the rho axis, so they are evaluated on the
  1-D c grid only (rho node index -1) with the c-axis masses —
  mathematically identical to the full grid because the rho masses sum to
  one. Under the strong-weighted heredity mixture this is a large share
  of the model space, and the mains-only variant is entirely 1-D.
- `moment_pass`: coefficient moments (m_n, V_n, b_n) only for the
  components that survive the caller's mass-based selection, grouped by
  (model size, c node, rho node).

Candidates are emitted in (model, c, rho) lexicographic order so that the
single-node keep-all case has component i == model i.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gammaln, logsumexp

from linearpfn.prior import slab_variance, trunc_lognormal_logpdf

if TYPE_CHECKING:
    from linearpfn.prior import PriorConfig

__all__ = ["evidence_pass", "gl_log_nodes", "moment_pass", "node_grid"]

# stats tuple: (Z'Z, Z'y, y'y, n) — see reference._design_stats
Stats = tuple[np.ndarray, np.ndarray, float, int]


def gl_log_nodes(
    logmean: float, logsd: float, lo: float, hi: float, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes on t = log(value) over [log lo, log hi] and log
    masses, renormalized to logsumexp(masses) = 0.

    mass_i ∝ w_i * value_i * p(value_i) with p the truncated-lognormal
    density (the value_i factor is the dvalue = e^t dt Jacobian)."""
    x, w = np.polynomial.legendre.leggauss(k)
    a, b = math.log(lo), math.log(hi)
    t = 0.5 * (a + b) + 0.5 * (b - a) * x
    values = np.exp(t)
    logmass = (
        np.log(w * (0.5 * (b - a)))
        + t
        + np.asarray(trunc_lognormal_logpdf(values, logmean, logsd, lo, hi))
    )
    return values, logmass - logsumexp(logmass)


def node_grid(
    cfg: PriorConfig, n_c: int, n_rho: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quadrature axes (c_nodes, c_logmass, rho_nodes, rho_logmass).

    A degenerate axis (random gate off) is a single node at the fixed value
    with mass exactly 1, reducing the whole machinery to the fixed-scale path."""
    if cfg.random_c:
        c_nodes, c_logmass = gl_log_nodes(
            cfg.c_logmean, cfg.c_logsd, cfg.c_min, cfg.c_max, n_c
        )
    else:
        c_nodes, c_logmass = np.array([float(cfg.tau2)]), np.zeros(1)
    if cfg.random_rho_int and cfg.include_interactions:
        rho_nodes, rho_logmass = gl_log_nodes(
            cfg.rho_logmean, cfg.rho_logsd, cfg.rho_min, cfg.rho_max, n_rho
        )
    else:
        rho_nodes, rho_logmass = np.array([float(cfg.rho_int_fixed)]), np.zeros(1)
    return c_nodes, c_logmass, rho_nodes, rho_logmass


def _nig_logev(
    gram: np.ndarray,
    csub: np.ndarray,
    v0_sub: np.ndarray,
    yty: float,
    a_n: float,
    const: float,
    b0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched NIG log evidence and b_n for a stack of same-size models at
    one fixed V0. A = V0^{-1} + Z'Z is PD by construction: no jitter."""
    a_mat = gram.copy()
    ar = np.arange(gram.shape[1])
    a_mat[:, ar, ar] += 1.0 / v0_sub
    chol = np.linalg.cholesky(a_mat)
    logdet_vn = -2.0 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
    m_n = np.linalg.solve(a_mat, csub[..., None])[..., 0]
    quad = np.einsum("ms,ms->m", m_n, csub)
    b_n = b0 + 0.5 * (yty - quad)
    log_ev = const + 0.5 * (logdet_vn - np.log(v0_sub).sum(axis=1)) - a_n * np.log(b_n)
    return log_ev, b_n


def evidence_pass(
    stats: Stats,
    gammas: np.ndarray,
    log_prior: np.ndarray,
    cfg: PriorConfig,
    p: int,
    c_nodes: np.ndarray,
    c_logmass: np.ndarray,
    rho_nodes: np.ndarray,
    rho_logmass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pass 1: (log_evidence (M,), cand_model, cand_c, cand_rho,
    cand_logjoint, cand_bn), candidates lex-sorted by (model, c, rho).

    cand_logjoint = log_prior + node mass + node evidence, UNNORMALIZED;
    cand_rho = -1 marks rho-inert entries (evidence constant in rho)."""
    G, cvec, yty, n = stats
    M, d = gammas.shape
    a_n = cfg.a0 + n / 2.0
    const = (
        -0.5 * n * math.log(2.0 * math.pi)
        + cfg.a0 * math.log(cfg.b0)
        + gammaln(a_n)
        - gammaln(cfg.a0)
    )
    Kc, Kr = c_nodes.size, rho_nodes.size
    sizes = gammas.sum(axis=1)
    has_int = gammas[:, 1 + p :].any(axis=1) if d > 1 + p else np.zeros(M, dtype=bool)
    log_evidence = np.empty(M)
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for s in np.unique(sizes):
        s = int(s)
        for inert in (True, False):
            rows = np.flatnonzero((sizes == s) & (has_int != inert))
            if rows.size == 0:
                continue
            idx = np.nonzero(gammas[rows])[1].reshape(rows.size, s)
            gram = G[idx[:, :, None], idx[:, None, :]]
            csub = cvec[idx]
            if inert:
                log_ev = np.empty((rows.size, Kc))
                b_n = np.empty((rows.size, Kc))
                for kc in range(Kc):
                    tau2_main = slab_variance(s - 1, cfg, c=float(c_nodes[kc]), rho=1.0)
                    v0_sub = np.where(idx == 0, cfg.tau2_intercept, tau2_main)
                    log_ev[:, kc], b_n[:, kc] = _nig_logev(
                        gram, csub, v0_sub, yty, a_n, const, cfg.b0
                    )
                log_evidence[rows] = logsumexp(c_logmass[None, :] + log_ev, axis=1)
                logjoint = log_prior[rows, None] + c_logmass[None, :] + log_ev
                parts.append((
                    np.repeat(rows, Kc),
                    np.tile(np.arange(Kc), rows.size),
                    np.full(rows.size * Kc, -1, dtype=np.int64),
                    logjoint.ravel(),
                    b_n.ravel(),
                ))
            else:
                int_slot = idx > p
                log_ev = np.empty((rows.size, Kc, Kr))
                b_n = np.empty((rows.size, Kc, Kr))
                for kc in range(Kc):
                    for kr in range(Kr):
                        c_val, rho_val = float(c_nodes[kc]), float(rho_nodes[kr])
                        tau2_main = slab_variance(s - 1, cfg, c=c_val, rho=rho_val)
                        tau2_int = slab_variance(
                            s - 1, cfg, c=c_val, rho=rho_val, is_interaction=True
                        )
                        v0_sub = np.where(
                            idx == 0,
                            cfg.tau2_intercept,
                            np.where(int_slot, tau2_int, tau2_main),
                        )
                        log_ev[:, kc, kr], b_n[:, kc, kr] = _nig_logev(
                            gram, csub, v0_sub, yty, a_n, const, cfg.b0
                        )
                mass = c_logmass[:, None] + rho_logmass[None, :]
                log_evidence[rows] = logsumexp(
                    (mass[None, :, :] + log_ev).reshape(rows.size, -1), axis=1
                )
                logjoint = log_prior[rows, None, None] + mass[None, :, :] + log_ev
                grid_c = np.repeat(np.arange(Kc), Kr)
                grid_r = np.tile(np.arange(Kr), Kc)
                parts.append((
                    np.repeat(rows, Kc * Kr),
                    np.tile(grid_c, rows.size),
                    np.tile(grid_r, rows.size),
                    logjoint.reshape(rows.size, -1).ravel(),
                    b_n.reshape(rows.size, -1).ravel(),
                ))

    cand_model = np.concatenate([q[0] for q in parts])
    cand_c = np.concatenate([q[1] for q in parts])
    cand_rho = np.concatenate([q[2] for q in parts])
    cand_logjoint = np.concatenate([q[3] for q in parts])
    cand_bn = np.concatenate([q[4] for q in parts])
    order = np.lexsort((cand_rho, cand_c, cand_model))
    return (
        log_evidence,
        cand_model[order],
        cand_c[order],
        cand_rho[order],
        cand_logjoint[order],
        cand_bn[order],
    )


def moment_pass(
    stats: Stats,
    gammas: np.ndarray,
    cfg: PriorConfig,
    p: int,
    comp_model: np.ndarray,
    comp_c: np.ndarray,
    comp_rho: np.ndarray,
    c_nodes: np.ndarray,
    rho_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pass 2: (b_n (C,), m_full (C, d), v_full (C, d, d)) for the kept
    components only, grouped by (model size, c node, rho node)."""
    G, cvec, yty, _ = stats
    C = comp_model.size
    d = gammas.shape[1]
    sizes = gammas.sum(axis=1)[comp_model]
    b_n = np.empty(C)
    m_full = np.zeros((C, d))
    v_full = np.zeros((C, d, d))
    keys = np.stack([sizes, comp_c, comp_rho], axis=1)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)  # numpy 2.0 returns (C, 1) for axis-based unique
    for gi, (s, kc, kr) in enumerate(uniq):
        s = int(s)
        comps = np.flatnonzero(inverse == gi)
        idx = np.nonzero(gammas[comp_model[comps]])[1].reshape(comps.size, s)
        c_val = float(c_nodes[int(kc)])
        rho_val = float(rho_nodes[int(kr)]) if kr >= 0 else float(rho_nodes[0])
        tau2_main = slab_variance(s - 1, cfg, c=c_val, rho=rho_val)
        tau2_int = slab_variance(s - 1, cfg, c=c_val, rho=rho_val, is_interaction=True)
        v0_sub = np.where(
            idx == 0, cfg.tau2_intercept, np.where(idx > p, tau2_int, tau2_main)
        )
        a_mat = G[idx[:, :, None], idx[:, None, :]].copy()
        ar = np.arange(s)
        a_mat[:, ar, ar] += 1.0 / v0_sub
        csub = cvec[idx]
        m_n = np.linalg.solve(a_mat, csub[..., None])[..., 0]
        quad = np.einsum("ms,ms->m", m_n, csub)
        b_n[comps] = cfg.b0 + 0.5 * (yty - quad)
        v_n = np.linalg.inv(a_mat)
        m_full[comps[:, None], idx] = m_n
        v_full[comps[:, None, None], idx[:, :, None], idx[:, None, :]] = v_n
    return b_n, m_full, v_full
