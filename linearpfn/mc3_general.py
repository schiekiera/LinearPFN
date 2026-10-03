"""MC3 for the general prior: Metropolis-Hastings over ALL active sets,
targeting exactly the posterior `fit_exact` enumerates at small p.

`linearpfn.mc3` serves the fixed-scale configuration only (fixed c, strong
heredity enforced by its proposal sets, non-empty models, dense per-model
covariances in the result). This module is the sampler for the general prior
(heredity mixture, count / rate / mixture priors, random slab scale) and for
mains-only priors:

- Target. pi(gamma) proportional to exp(log_prior_gamma(gamma)) x
  sum_k w_k p(y | X, gamma, c_k, rho_fixed), the c-marginalized NIG evidence
  on the reference's Gauss-Legendre grid (`_quadrature.node_grid`,
  `reference.QUAD_NODES_C`). The support is whatever `log_prior_gamma`
  gives finite mass — under the heredity mixture every gamma, the empty
  model included. Heredity therefore lives in the TARGET, not in the moves.
  The rho axis must be degenerate (random_rho_int=False): a tensor grid is
  refused rather than approximated.
- Moves. P_add = P_drop = 0.4, P_swap = 0.2. Add: uniform over inactive
  effects; drop: uniform over active effects; swap: one uniform drop then
  one uniform add. Hastings terms: add log|I| - log(|A|+1), drop
  log|A| - log(|I|+1), swap 0 (both counts preserved).
- Chains. n_chains independent prior draws (`sample_gamma`) as starts, one
  shared generator; burn-in states discarded; no thinning.
- Estimators over the FULL kept trace: frequency PIPs (pooled and per
  chain); the coefficient posterior mean accumulated online (the c-node-
  weighted NIG mean of each visited state, computed once per state change
  and weighted by the run length); the predictive log density on optional
  query rows accumulated online in log space (mixture of Student-t over
  states x nodes). Nothing is truncated to a model subset and no d x d
  matrix is ever stored, so d = 466 (p = 30 with interactions) is fine.
- Diagnostics. Per-indicator split R-hat and Geyer initial-positive-
  sequence ESS (summed over chains) on the packed kept trace, the same
  definitions as `linearpfn.mc3`; acceptance by move;
  unique visited models (saturates at cache_cap); wall seconds for
  sampling and for diagnostics separately.

"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp

from linearpfn._quadrature import node_grid
from linearpfn.mc3 import MC3Diagnostics
from linearpfn.prior import (
    PriorConfig,
    build_design,
    interaction_pairs,
    log_prior_gamma,
    n_effects,
    sample_gamma,
    slab_variance,
)
from linearpfn.reference import QUAD_NODES_C, QUAD_NODES_RHO, _design_stats

__all__ = ["MC3GeneralResult", "P_SWAP", "fit_mc3_general"]

P_SWAP = 0.2  # P_add = P_drop = (1 - P_SWAP) / 2


@dataclass(frozen=True)
class MC3GeneralResult:
    pip: np.ndarray  # (d,) pooled frequency PIPs, intercept slot 1.0
    pip_by_chain: np.ndarray  # (n_chains, d-1)
    coef_mean: np.ndarray  # (d,) posterior mean, intercept included
    nll_query: float | None  # mean predictive NLL on the query rows (None without)
    diagnostics: MC3Diagnostics
    seconds: float  # sampling wall clock
    seconds_diag: float  # R-hat / ESS wall clock


class _NodeEvidence:
    """c-marginalized NIG evidence and per-state summaries for one dataset.

    Sufficient statistics are computed once (`reference._design_stats`); per
    state only the active s x s block is factorized, batched over the K
    c-nodes. Per-effect prior variances come from `slab_variance` per node
    and per active count k (so `per_k` scaling stays correct)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, cfg: PriorConfig, X_query=None):
        Z = build_design(X, cfg)
        self.G, self.cvec, self.yty, self.n = _design_stats(Z, y)
        self.p = X.shape[1]
        self.d = Z.shape[1]
        self.cfg = cfg
        c_nodes, c_logmass, rho_nodes, _ = node_grid(cfg, QUAD_NODES_C, QUAD_NODES_RHO)
        if rho_nodes.size != 1:
            raise ValueError("mc3_general: a rho quadrature axis is not supported")
        self.c_nodes, self.c_logmass = c_nodes, c_logmass
        self.rho = float(rho_nodes[0])
        self.K = int(c_nodes.size)
        self.a_n = cfg.a0 + self.n / 2.0
        self.const = (
            -0.5 * self.n * math.log(2.0 * math.pi)
            + cfg.a0 * math.log(cfg.b0)
            + gammaln(self.a_n)
            - gammaln(cfg.a0)
        )
        self.Zq = None if X_query is None else build_design(np.asarray(X_query, float), cfg)
        self._v0: dict[int, np.ndarray] = {}

    def v0(self, k: int) -> np.ndarray:
        """(K, d) per-effect prior variances for a model with k active effects."""
        v = self._v0.get(k)
        if v is None:
            cfg, p, d = self.cfg, self.p, self.d
            v = np.empty((self.K, d))
            for kk, c in enumerate(self.c_nodes):
                v[kk, 0] = cfg.tau2_intercept
                v[kk, 1 : 1 + p] = slab_variance(k, cfg, c=float(c), rho=self.rho)
                if d > 1 + p:
                    v[kk, 1 + p :] = slab_variance(
                        k, cfg, c=float(c), rho=self.rho, is_interaction=True
                    )
            self._v0[k] = v
        return v

    def solve(self, gamma: np.ndarray):
        """(idx, log_ev (K,), m (K, s), A (K, s, s), b_n (K,)) at every node."""
        idx = np.flatnonzero(gamma)
        s = idx.size
        v0 = self.v0(s - 1)[:, idx]
        A = np.broadcast_to(self.G[np.ix_(idx, idx)], (self.K, s, s)).copy()
        ar = np.arange(s)
        A[:, ar, ar] += 1.0 / v0
        chol = np.linalg.cholesky(A)
        logdet_vn = -2.0 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
        csub = self.cvec[idx]
        m = np.linalg.solve(A, np.broadcast_to(csub, (self.K, s))[..., None])[..., 0]
        b_n = self.cfg.b0 + 0.5 * (self.yty - m @ csub)
        log_ev = self.const + 0.5 * (logdet_vn - np.log(v0).sum(axis=1)) - self.a_n * np.log(b_n)
        return idx, log_ev, m, A, b_n

    def log_evidence(self, gamma: np.ndarray) -> float:
        return float(logsumexp(self.c_logmass + self.solve(gamma)[1]))

    def summaries(self, gamma: np.ndarray, y_query: np.ndarray | None):
        """(coef_mean (d,), log predictive density at the query rows | None)."""
        idx, log_ev, m, A, b_n = self.solve(gamma)
        logw = self.c_logmass + log_ev
        logw -= logsumexp(logw)  # node posterior given the model
        w = np.exp(logw)
        coef = np.zeros(self.d)
        coef[idx] = w @ m
        if self.Zq is None:
            return coef, None
        Zs = self.Zq[:, idx]  # (nq, s)
        nq = Zs.shape[0]
        mu = Zs @ m.T  # (nq, K)
        sol = np.linalg.solve(A, np.broadcast_to(Zs.T, (self.K, idx.size, nq)))
        h = np.einsum("qs,ksq->qk", Zs, sol)
        scale2 = (b_n / self.a_n)[None, :] * (1.0 + h)
        nu = 2.0 * self.a_n
        t2 = (y_query[:, None] - mu) ** 2 / scale2
        logpdf = (
            gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(nu * math.pi)
            - 0.5 * np.log(scale2) - (nu + 1) / 2 * np.log1p(t2 / nu)
        )
        return coef, logsumexp(logpdf + logw[None, :], axis=1)


class _PriorMemo:
    """`log_prior_gamma` memoized on the statistics it depends on: (k_main,
    k_int, every active interaction strong-eligible, every one weak-eligible).
    Mains are exchangeable and so are pairs within an eligibility class, so
    the key is sufficient (it agrees with direct evaluation, including on
    heredity-violating and empty models)."""

    def __init__(self, p: int, cfg: PriorConfig):
        self.p, self.cfg = p, cfg
        pairs = np.asarray(interaction_pairs(p, cfg), dtype=int).reshape(-1, 2)
        self.pa, self.pb = pairs[:, 0], pairs[:, 1]
        self.memo: dict[tuple, float] = {}

    def __call__(self, gamma: np.ndarray) -> float:
        p = self.p
        mains, ints = gamma[1 : 1 + p], gamma[1 + p :]
        k_int = int(ints.sum())
        if k_int:
            a, b = mains[self.pa[ints]], mains[self.pb[ints]]
            strong, weak = bool(np.all(a & b)), bool(np.all(a | b))
        else:
            strong = weak = True
        key = (int(mains.sum()), k_int, strong, weak)
        v = self.memo.get(key)
        if v is None:
            v = float(log_prior_gamma(gamma, p, self.cfg))
            self.memo[key] = v
        return v


def _propose(rng: np.random.Generator, gamma: np.ndarray, p_swap: float):
    """One add/drop/swap proposal: (proposal | None, log q(rev)/q(fwd), kind)."""
    act = np.flatnonzero(gamma[1:]) + 1
    inact = np.flatnonzero(~gamma[1:]) + 1
    p_add = (1.0 - p_swap) / 2.0
    u = rng.random()
    prop = gamma.copy()
    if u < p_add:
        if inact.size == 0:
            return None, 0.0, "add"
        prop[rng.choice(inact)] = True
        return prop, math.log(inact.size) - math.log(act.size + 1), "add"
    if u < 2 * p_add:
        if act.size == 0:
            return None, 0.0, "drop"
        prop[rng.choice(act)] = False
        return prop, math.log(act.size) - math.log(inact.size + 1), "drop"
    if act.size == 0 or inact.size == 0:
        return None, 0.0, "swap"
    prop[rng.choice(act)] = False
    prop[rng.choice(inact)] = True
    return prop, 0.0, "swap"


def _rhat_from_halves(means: np.ndarray, vars_: np.ndarray, length: int) -> np.ndarray:
    """Vectorized `mc3._rhat_split` from per-half-chain means/variances (2C, J)."""
    w = vars_.mean(axis=0)
    b = length * means.var(axis=0, ddof=1)
    out = np.sqrt((length - 1) / length + b / np.where(w > 1e-12, w, 1.0) / length)
    degenerate = w <= 1e-12
    out[degenerate] = np.where(b[degenerate] <= 1e-12, 1.0, np.inf)
    return out


def _ess_geyer(x: np.ndarray) -> float:
    """`mc3._ess_geyer` with the initial-positive-sequence search vectorized."""
    n = x.size
    x = x - x.mean()
    if float(x @ x) <= 0.0:
        return float(n)
    f = np.fft.rfft(x, 2 * n)
    acov = np.fft.irfft(f * np.conj(f))[:n] / n
    rho = acov / acov[0]
    m = n // 2
    pairs = rho[1 : 2 * m - 2 : 2] + rho[2 : 2 * m - 1 : 2]
    neg = np.flatnonzero(pairs < 0)
    stop = int(neg[0]) if neg.size else pairs.size
    return n / (1.0 + 2.0 * float(pairs[:stop].sum()))


def fit_mc3_general(
    X: np.ndarray,
    y: np.ndarray,
    cfg: PriorConfig,
    *,
    X_query: np.ndarray | None = None,
    y_query: np.ndarray | None = None,
    n_chains: int = 4,
    n_steps: int = 4000,
    burn_in: int | None = None,
    seed: int = 0,
    p_swap: float = P_SWAP,
    cache_cap: int = 2_000_000,
) -> MC3GeneralResult:
    """Sampled posterior over active sets for one dataset (see module doc).

    X is used as given: the caller standardizes it the way `fit_exact`
    expects (context slices of a standardized dataset are fine, as in the
    validation harness). burn_in defaults to n_steps // 4; the kept sample
    per chain is n_steps - burn_in states. X_query/y_query switch on the
    online predictive-NLL estimate."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if (X_query is None) != (y_query is None):
        raise ValueError("pass X_query and y_query together")
    if burn_in is None:
        burn_in = n_steps // 4
    if not 0 <= burn_in < n_steps:
        raise ValueError("need 0 <= burn_in < n_steps")
    n, p = X.shape
    d = n_effects(p, cfg)
    ev = _NodeEvidence(X, y, cfg, X_query)
    prior = _PriorMemo(p, cfg)
    yq = None if y_query is None else np.asarray(y_query, dtype=float)
    cache: dict[bytes, float] = {}

    def log_post(gamma: np.ndarray) -> float:
        key = np.packbits(gamma).tobytes()
        v = cache.get(key)
        if v is None:
            v = prior(gamma) + ev.log_evidence(gamma)
            if len(cache) < cache_cap:
                cache[key] = v
        return v

    rng = np.random.default_rng(seed)
    n_kept = n_steps - burn_in
    n_bytes = (d - 1 + 7) // 8
    trace = np.zeros((n_chains, n_kept, n_bytes), dtype=np.uint8)
    pip_sum = np.zeros((n_chains, d - 1))
    coef_sum = np.zeros(d)
    logdens = None if yq is None else np.full(yq.size, -np.inf)
    proposed = {"add": 0, "drop": 0, "swap": 0}
    accepted = dict.fromkeys(proposed, 0)
    def flush(chain: int, gamma: np.ndarray, run: int, summ):
        """Add the current state's run to every estimator; returns its summaries."""
        if run == 0:
            return summ
        if summ is None:
            summ = ev.summaries(gamma, yq)
        pip_sum[chain] += run * gamma[1:]
        coef_sum[:] += run * summ[0]
        if logdens is not None:
            np.logaddexp(logdens, summ[1] + math.log(run), out=logdens)
        return summ

    t0 = time.perf_counter()
    for chain in range(n_chains):
        gamma = sample_gamma(rng, p, cfg)
        lp = log_post(gamma)
        packed = np.packbits(gamma[1:])
        run = 0  # kept steps spent in the current state
        summ = None  # (coef, logdens) of the current state, computed lazily
        for step in range(n_steps):
            prop, log_q, kind = _propose(rng, gamma, p_swap)
            proposed[kind] += 1
            if prop is not None:
                lp_new = log_post(prop)
                if np.isfinite(lp_new) and math.log(rng.random()) < lp_new - lp + log_q:
                    flush(chain, gamma, run, summ)
                    gamma, lp, run, summ = prop, lp_new, 0, None
                    packed = np.packbits(gamma[1:])
                    accepted[kind] += 1
            if step >= burn_in:
                trace[chain, step - burn_in] = packed
                run += 1
        flush(chain, gamma, run, summ)
    seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    total = float(n_chains * n_kept)
    counts = pip_sum.sum(axis=0)
    has_var = (counts > 0) & (counts < total)
    half = n_kept // 2
    means = np.empty((2 * n_chains, d - 1))
    vars_ = np.empty((2 * n_chains, d - 1))
    ess_vec = np.full(d - 1, total)
    ess_acc = np.zeros(int(has_var.sum()))
    for chain in range(n_chains):
        bits = np.unpackbits(trace[chain], axis=1, count=d - 1)
        for h, sl in enumerate((slice(0, half), slice(half, 2 * half))):
            s = bits[sl].sum(axis=0, dtype=np.int64).astype(float)
            mu = s / half
            means[2 * chain + h] = mu
            vars_[2 * chain + h] = (s - half * mu * mu) / (half - 1)  # ddof=1 for 0/1 data
        for j_out, j in enumerate(np.flatnonzero(has_var)):
            ess_acc[j_out] += _ess_geyer(bits[:, j].astype(float))
    ess_vec[has_var] = ess_acc
    rhat_vec = _rhat_from_halves(means, vars_, half)
    seconds_diag = time.perf_counter() - t1

    diagnostics = MC3Diagnostics(
        rhat_max=float(rhat_vec.max()),
        ess_min=float(ess_vec[has_var].min()) if has_var.any() else total,
        accept_rate=sum(accepted.values()) / (n_chains * n_steps),
        n_chains=n_chains,
        n_steps=n_steps,
        burn_in=burn_in,
        n_unique_models=len(cache),
        rhat=rhat_vec,
        ess=ess_vec,
        accept_by_move={
            k: (accepted[k] / proposed[k] if proposed[k] else float("nan")) for k in proposed
        },
    )
    pip = np.concatenate([[1.0], counts / total])
    return MC3GeneralResult(
        pip=pip,
        pip_by_chain=pip_sum / n_kept,
        coef_mean=coef_sum / total,
        nll_query=None if logdens is None else float(-(logdens - math.log(total)).mean()),
        diagnostics=diagnostics,
        seconds=seconds,
        seconds_diag=seconds_diag,
    )
