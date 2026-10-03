"""MC3-style sampled reference over heredity-valid model space (numpy only).

Metropolis-Hastings on active sets with add / drop / swap proposals that
respect strong heredity, targeting evidence(gamma) * prior(gamma) — the
same estimand the exact enumeration computes at p <= 5, at any p. Both
factors reuse already-tested code: the closed-form NIG evidence from
`linearpfn.reference` and the analytic heredity-aware prior from
`linearpfn.prior`. PIPs and coefficient posterior means are
sample averages; the sampled posterior is packaged as a frequency-weighted
`ExactPosterior`, so every predictive / PIP / coefficient panel works
unchanged — always label results from this reference as approximate.

Proposals from state gamma (P_add = P_drop = 0.4, P_swap = 0.2):
- add: activate a uniform draw from the addable set A(gamma) — any inactive
  main; an inactive interaction with both parents active;
- drop: deactivate a uniform draw from the droppable set D(gamma) — any
  active interaction; an active main with no active child interactions
  (drops never cascade); the last remaining active effect is never
  droppable (the prior places zero mass on the empty set);
- swap: one drop then one add from the intermediate state. The
  intermediate state of the reverse swap is identical, so its addable
  count cancels from the Hastings ratio.

Acceptance in logs: log alpha = delta log[evidence * prior] + log q(rev)
- log q(fwd), with q counts taken from the correct states (|A(gamma)| /
|D(gamma')| for add, mirrored for drop, |D(gamma)| / |D(gamma')| for swap;
P_add = P_drop so the type probabilities cancel).

Diagnostics: split R-hat per effect indicator across chains (max reported)
and Geyer initial-positive-sequence effective sample size (min reported
over effects with variance).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from linearpfn.prior import (
    PriorConfig,
    build_design,
    interaction_pairs,
    log_prior_gamma,
    n_effects,
    sample_gamma,
)
from linearpfn.reference import ExactPosterior, _design_stats, _fit_models_from_stats

__all__ = ["MC3Diagnostics", "MC3Result", "fit_mc3"]

P_SWAP = 0.2  # P_add = P_drop = (1 - P_SWAP) / 2


@dataclass(frozen=True)
class MC3Diagnostics:
    rhat_max: float
    ess_min: float
    accept_rate: float
    n_chains: int
    n_steps: int
    burn_in: int
    n_unique_models: int
    rhat: np.ndarray  # (d-1,) per-effect split R-hat
    ess: np.ndarray  # (d-1,) per-effect ESS (n_kept for zero-variance effects)
    accept_by_move: dict[str, float]  # acceptance rate per proposal type


@dataclass(frozen=True)
class MC3Result:
    posterior: ExactPosterior  # frequency-weighted over unique visited models
    pip: np.ndarray  # (d,) sample-average inclusion probabilities
    coef_mean: np.ndarray  # (d,)
    diagnostics: MC3Diagnostics


def _addable(gamma: np.ndarray, p: int, pairs: np.ndarray) -> np.ndarray:
    mains = gamma[1 : 1 + p]
    ints = gamma[1 + p :]
    add_mains = 1 + np.flatnonzero(~mains)
    elig = mains[pairs[:, 0]] & mains[pairs[:, 1]]
    add_ints = 1 + p + np.flatnonzero(elig & ~ints)
    return np.concatenate([add_mains, add_ints])


def _droppable(gamma: np.ndarray, p: int, pairs: np.ndarray) -> np.ndarray:
    mains = gamma[1 : 1 + p]
    ints = gamma[1 + p :]
    if int(mains.sum()) + int(ints.sum()) <= 1:
        return np.empty(0, dtype=int)  # never empty the active set
    drop_ints = 1 + p + np.flatnonzero(ints)
    has_child = np.zeros(p, dtype=bool)
    for t in np.flatnonzero(ints):
        has_child[pairs[t, 0]] = True
        has_child[pairs[t, 1]] = True
    drop_mains = 1 + np.flatnonzero(mains & ~has_child)
    return np.concatenate([drop_mains, drop_ints])


def _rhat_split(seqs: np.ndarray) -> float:
    """Split R-hat over (chains, kept) indicator sequences for one effect."""
    half = seqs.shape[1] // 2
    x = np.concatenate([seqs[:, :half], seqs[:, half : 2 * half]], axis=0).astype(float)
    length = x.shape[1]
    w = x.var(axis=1, ddof=1).mean()
    b = length * x.mean(axis=1).var(ddof=1)
    if w <= 1e-12:
        return 1.0 if b <= 1e-12 else float("inf")
    return float(np.sqrt((length - 1) / length + b / (w * length)))


def _ess_geyer(x: np.ndarray) -> float:
    """Effective sample size of one chain's sequence (initial positive pairs)."""
    n = x.size
    x = x - x.mean()
    if float(x @ x) <= 0.0:
        return float(n)
    f = np.fft.rfft(x, 2 * n)
    acov = np.fft.irfft(f * np.conj(f))[:n] / n
    rho = acov / acov[0]
    tau = 1.0
    for k in range(1, n // 2):
        pair = rho[2 * k - 1] + rho[2 * k]
        if pair < 0:
            break
        tau += 2.0 * pair
    return n / tau


def _single_move(
    rng: np.random.Generator,
    gamma: np.ndarray,
    p: int,
    pairs: np.ndarray,
    p_add: float,
) -> tuple[np.ndarray | None, float, str]:
    """One add/drop/swap proposal: (proposal | None, log q(rev)/q(fwd), kind)."""
    u = rng.random()
    proposal = gamma.copy()
    if u < p_add:
        cands = _addable(gamma, p, pairs)
        if not cands.size:
            return None, 0.0, "add"
        proposal[rng.choice(cands)] = True
        return proposal, float(
            np.log(cands.size) - np.log(_droppable(proposal, p, pairs).size)
        ), "add"
    if u < 2 * p_add:
        cands = _droppable(gamma, p, pairs)
        if not cands.size:
            return None, 0.0, "drop"
        proposal[rng.choice(cands)] = False
        return proposal, float(
            np.log(cands.size) - np.log(_addable(proposal, p, pairs).size)
        ), "drop"
    drops = _droppable(gamma, p, pairs)
    if not drops.size:
        return None, 0.0, "swap"
    proposal[rng.choice(drops)] = False
    adds = _addable(proposal, p, pairs)
    proposal[rng.choice(adds)] = True
    if np.array_equal(proposal, gamma):
        return None, 0.0, "swap"
    # the reverse swap's intermediate is identical, so its addable count cancels
    return proposal, float(
        np.log(drops.size) - np.log(_droppable(proposal, p, pairs).size)
    ), "swap"


def fit_mc3(
    X: np.ndarray,
    y: np.ndarray,
    cfg: PriorConfig,
    n_chains: int = 4,
    n_steps: int = 4000,
    burn_in: int = 1000,
    seed: int = 0,
    standardization_tol: float = 1e-6,
    multi_flip: float = 0.0,
) -> MC3Result:
    """Sampled posterior over heredity-valid models for one dataset.

    X must be standardized like `fit_exact` expects (same tolerance
    semantics). n_steps counts post-init proposals per chain; the kept
    sample per chain is n_steps - burn_in states (no thinning; rejected
    steps repeat the state, as the MH estimator requires).

    multi_flip: probability of proposing a composite of 2-3 chained single
    moves in one MH step (a mixing aid for modal trapping at larger p). The
    reverse path visits the same intermediate states backwards with mirrored
    move types, so the Hastings factor is the sum of the component factors —
    a standard sequential-proposal construction, valid path-wise.
    """
    if not cfg.include_interactions:
        raise ValueError(
            "MC3 targets the interactions prior; mains-only model spaces "
            "enumerate exactly to p <= MAX_ENUM_P_MAINS_ONLY — use fit_exact"
        )
    if (
        tuple(cfg.heredity_weights) != (1.0, 0.0, 0.0)
        or not cfg.reject_empty
        or cfg.random_c
        or cfg.random_rho_int
    ):
        raise ValueError(
            "fit_mc3 serves the fixed-scale configuration only: its add/drop/swap "
            "kernel enforces strong heredity and non-empty active sets, and its "
            "evidence is a single fixed-(c, rho) point — it cannot target the "
            "heredity mixture or a random slab scale. Use fit_exact (quadrature), "
            "linearpfn.mc3_general, or the fixed-scale prior config."
        )
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    tol = standardization_tol
    if np.abs(X.mean(axis=0)).max() > tol or np.abs(X.std(axis=0) - 1.0).max() > tol:
        raise ValueError("X must be z-standardized (ddof=0); see fit_exact")
    d = n_effects(p)
    pairs = np.asarray(interaction_pairs(p), dtype=int).reshape(-1, 2)
    stats = _design_stats(build_design(X), y)
    rng = np.random.default_rng(seed)

    cache: dict[bytes, float] = {}

    def log_post(gamma: np.ndarray) -> float:
        key = gamma.tobytes()
        if key not in cache:
            log_ev = float(_fit_models_from_stats(stats, gamma[None, :], cfg)[0][0])
            cache[key] = log_ev + float(log_prior_gamma(gamma, p, cfg))
        return cache[key]

    kept = np.empty((n_chains, n_steps - burn_in, d - 1), dtype=np.uint8)
    counts: dict[bytes, int] = {}
    proposed = {"add": 0, "drop": 0, "swap": 0, "multi": 0}
    accepted = dict.fromkeys(proposed, 0)
    p_add = (1.0 - P_SWAP) / 2.0
    for chain in range(n_chains):
        gamma = sample_gamma(rng, p, cfg)
        lp = log_post(gamma)
        for step in range(n_steps):
            if multi_flip > 0.0 and rng.random() < multi_flip:
                kind = "multi"
                length = 2 + int(rng.random() < 0.5)
                cur, log_q = gamma, 0.0
                for _ in range(length):
                    nxt, q, _ = _single_move(rng, cur, p, pairs, p_add)
                    if nxt is None:
                        cur = None
                        break
                    cur, log_q = nxt, log_q + q
                proposal = None if cur is None or np.array_equal(cur, gamma) else cur
            else:
                proposal, log_q, kind = _single_move(rng, gamma, p, pairs, p_add)
            proposed[kind] += 1
            if proposal is not None:
                lp_new = log_post(proposal)
                if np.isfinite(lp_new) and np.log(rng.random()) < lp_new - lp + log_q:
                    gamma, lp = proposal, lp_new
                    accepted[kind] += 1
            if step >= burn_in:
                kept[chain, step - burn_in] = gamma[1:]
                key = gamma.tobytes()
                counts[key] = counts.get(key, 0) + 1
    n_accept = sum(accepted.values())

    unique = np.array([np.frombuffer(k, dtype=bool) for k in counts])
    freqs = np.array([counts[g.tobytes()] for g in unique], dtype=float)
    log_ev, b_n, m_full, v_full = _fit_models_from_stats(stats, unique, cfg)
    log_weights = np.log(freqs) - np.log(freqs.sum())
    posterior = ExactPosterior.from_single_node(
        gammas=unique,
        log_prior=np.asarray(log_prior_gamma(unique, p, cfg)),
        log_evidence=log_ev,
        log_weights=log_weights - logsumexp(log_weights),  # exact renormalization
        a_n=cfg.a0 + n / 2.0,
        b_n=b_n,
        m_full=m_full,
        v_full=v_full,
        p=p,
        cfg=cfg,
    )
    n_kept_total = float(kept.shape[0] * kept.shape[1])
    rhat_vec = np.array([_rhat_split(kept[:, :, j]) for j in range(d - 1)])
    ess_vec = np.full(d - 1, n_kept_total)
    has_var = kept.astype(float).var(axis=(0, 1)) > 0
    for j in np.flatnonzero(has_var):
        ess_vec[j] = sum(_ess_geyer(kept[c, :, j].astype(float)) for c in range(n_chains))
    diagnostics = MC3Diagnostics(
        rhat_max=float(rhat_vec.max()),
        ess_min=float(ess_vec[has_var].min()) if has_var.any() else n_kept_total,
        accept_rate=n_accept / (n_chains * n_steps),
        n_chains=n_chains,
        n_steps=n_steps,
        burn_in=burn_in,
        n_unique_models=len(counts),
        rhat=rhat_vec,
        ess=ess_vec,
        accept_by_move={
            k: (accepted[k] / proposed[k] if proposed[k] else float("nan"))
            for k in proposed
        },
    )
    return MC3Result(
        posterior=posterior,
        pip=posterior.pip(),
        coef_mean=posterior.coef_mean(),
        diagnostics=diagnostics,
    )
