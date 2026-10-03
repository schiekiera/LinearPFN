"""Prior specification for LinearPFN: amortized spike-and-slab regression.

This module is the single source of truth for the data-generating process
underlying LinearPFN — Bayesian variable selection for sparse linear models
with pairwise interactions, without MCMC. The synthetic-data generator
(`linearpfn.generator`) and the exact enumeration reference
(`linearpfn.reference`) must both consume the functions defined here so their
assumptions can never drift apart.

Generative recipe for one dataset (all constants live in `PriorConfig`):

1. n ~ Uniform{n_min..n_max}, p ~ Uniform{p_min..p_max}.
2. X ~ `sample_X` (implemented in `linearpfn._xgen`, re-exported here; the
   DGP spans these two modules and nothing else): a generic block with a
   random correlation matrix R (scipy's `random_correlation` with
   eigenvalues drawn from a Dirichlet(1_p) scaled to sum p; rows iid
   N(0, R), the whole mechanism when the factor share f = 0), optionally
   mixed with a factor-analytic
   block and pushed through a Gaussian copula onto mixed marginals; each
   column is z-standardized (ddof=0) at the end.
3. Candidate effects: p main effects and C(p,2) pairwise interactions, where
   interaction (j, k) is the elementwise product of standardized columns j
   and k (interaction columns are NOT re-standardized; their variance is
   1 + rho_jk^2 by construction). d = 1 + p + C(p,2) including the intercept.
4. Active set. A per-dataset heredity mode h ~ Categorical(heredity_weights)
   over (strong, weak, none) decides pair eligibility: interaction (j, k)
   is eligible iff both parent mains are active (strong), at least one is
   (weak), or unconditionally (none). How many effects are active depends
   on `sparsity_law`:
   - "count" (the count prior, the default): k_main and k_int are drawn directly from
     zero-inflated truncated-geometric laws (`linearpfn._counts`),
     p-independent, and the active subsets are uniform given the counts —
     a phenomenon has a handful of causes regardless of how many
     candidates were measured. The null-study mass is P(k_main = 0) =
     k_main_zero; no rejection machinery.
   - "rate" (the rate prior):
     pi_main ~ U(pi_main_low, pi_main_high), mains iid Bern(pi_main);
     pi_int ~ U(pi_int_low, pi_int_high), eligible interactions iid
     Bern(pi_int); ineligible interactions are never active. Under
     reject_empty=True (which requires pure strong heredity and the rate
     prior) an empty active set rejects the WHOLE step-4
     draw — including (pi_main, pi_int) — which keeps the marginal prior
     over gamma in closed Beta-function form; under reject_empty=False
     the empty (intercept-only) model is a legal draw
     (see `log_prior_gamma`).
   - "mixture" (the mixture prior): each dataset follows the count prior
     with probability `law_count_share`, else the rate prior; everything else
     (mode, slab, c, rho, X) is shared. The marginal prior over gamma is
     the same convex combination of the two pure marginals, so the exact
     reference needs no new machinery. Encodes honest uncertainty about
     whether the number of true causes is p-independent or grows with p.
5. Effect scale: c = tau2 unless random_c, in which case c is drawn from a
   lognormal truncated to [c_min, c_max] (`trunc_lognormal_draw`); rho =
   rho_int_fixed unless random_rho_int (interactions variant only), same
   truncated-lognormal mechanism on [rho_min, rho_max].
6. sigma2 ~ InvGamma(a0, b0), density ∝ x^(-a0-1) exp(-b0/x),
   mean b0/(a0-1).
7. intercept ~ N(0, tau2_intercept * sigma2); active effects iid
   N(0, tau2_eff * sigma2); inactive effects are exactly 0. Under
   slab_scaling="fixed", tau2_eff = c; under "per_k", tau2_eff = c / k
   with k the number of active non-intercept effects INCLUDING
   interactions (a per-model constant, so NIG conjugacy is untouched — see
   `slab_variance`). Active interactions get tau2_eff * rho: rho is a
   VARIANCE ratio.
8. y = Z beta + eps with eps ~ N(0, sigma2 I), Z = [1, X, interactions].
   y is never standardized or rescaled.

Canonical effect ordering used everywhere in this code:
[intercept, mains 0..p-1, pairs in itertools.combinations(range(p), 2)
order]. `gamma` is a length-d boolean vector over ALL d effects with the
intercept slot always True (the intercept is in every model and excluded
from selection).

"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple

import numpy as np
from scipy.special import betainc, betaln, logsumexp, ndtr, ndtri

from linearpfn._xgen import MARGINAL_TYPES, XMeta, sample_correlation, sample_X, standardize

__all__ = [
    "HEREDITY_MODES",
    "MARGINAL_TYPES",
    "Dataset",
    "PriorConfig",
    "XMeta",
    "build_design",
    "canonical_marginal_weights",
    "expected_active_count",
    "interaction_pairs",
    "log_p_empty",
    "log_prior_gamma",
    "n_effects",
    "prior_scale_diag",
    "sample_X",
    "sample_correlation",
    "sample_dataset",
    "sample_dataset_with_meta",
    "sample_gamma",
    "sample_heredity_mode",
    "slab_variance",
    "standardize",
    "trunc_lognormal_draw",
    "trunc_lognormal_logpdf",
]

# Canonical order of heredity modes; `PriorConfig.heredity_weights` follows
# this order (strong, weak, none) everywhere.
HEREDITY_MODES = ("strong", "weak", "none")


def canonical_marginal_weights(weights: dict[str, float]) -> tuple[tuple[str, float], ...]:
    """Frozen-safe canonical form of a marginal-type weight mapping: pairs
    in MARGINAL_TYPES order. Unknown names raise here so a YAML typo fails
    at config load with the offending name in the message."""
    unknown = set(weights) - set(MARGINAL_TYPES)
    if unknown:
        raise ValueError(f"unknown marginal types {sorted(unknown)}; valid: {MARGINAL_TYPES}")
    return tuple((t, float(weights[t])) for t in MARGINAL_TYPES if t in weights)


@dataclass(frozen=True)
class PriorConfig:
    """All constants of the data-generating process. Every field is settable
    from the config's data section; configs/paper_model.yaml holds the values
    of the reported model."""

    n_min: int = 20
    n_max: int = 1024
    p_min: int = 2
    p_max: int = 30
    # Sparsity law. "rate" (the rate prior): mains iid Bern(pi_main),
    # interactions iid Bern(pi_int) over the mode-eligible pairs, so
    # E[#actives] scales with p and E[#interactions] with C(p,2), i.e.
    # measuring more variables creates more true causes.
    # "count" (the count prior): k_main and k_int are drawn DIRECTLY from
    # zero-inflated truncated-geometric laws, p-independent, with a uniform
    # subset given the count: the number of real causes of a phenomenon is
    # small and does not depend on how many candidates were measured. The
    # pi_* fields are inert under "count"; the k_* fields under "rate".
    # "mixture" (the mixture prior): per dataset, the count prior with
    # probability law_count_share, else the rate prior; both parameter sets live.
    sparsity_law: Literal["rate", "count", "mixture"] = "count"
    law_count_share: float = 0.5  # inert unless sparsity_law == "mixture"
    pi_main_low: float = 0.02
    pi_main_high: float = 0.6
    pi_int_low: float = 0.05
    pi_int_high: float = 0.4
    # Count-prior parameters: *_zero is P(k = 0) (k_main_zero carries the
    # null-study mass); *_mean is the conditional-on-nonzero geometric mean
    # before truncation at p (mains) / the mode-eligible pool (interactions).
    k_main_zero: float = 0.09
    k_main_mean: float = 3.5
    k_int_zero: float = 0.5
    k_int_mean: float = 1.5
    a0: float = 3.0
    b0: float = 1.0
    # Slab parameterization.
    #
    # slab_scaling="fixed" (the default): each active effect has its OWN
    # variance c * sigma2 (interactions x rho): a cause's effect size does not
    # depend on how many other causes exist. R2 then grows with the
    # (count-prior-bounded) k; that R2 is informative about sparsity is
    # honest signal the exact posterior also uses, not a shortcut.
    #
    # "per_k": each effect has variance (c / k) * sigma2, keeping total
    # signal variance roughly constant in k (R2-D2 / g-prior style). Cost: at
    # large k every effect drops below the detection floor; even the exact
    # posterior leaves about a third of the true actives under PIP 0.5.
    #
    # tau2 = 0.5 is the per-k calibration (R2 5th-95th percentiles
    # [0.042, 0.546]); under "fixed" the per-effect scale is the random c
    # (see c_* below), calibrated against meta-analytic effect-size norms.
    tau2: float = 0.5
    tau2_intercept: float = 0.25
    slab_scaling: Literal["fixed", "per_k"] = "fixed"
    # Mains-only variant: candidate effects are the p mains only (d = 1 + p),
    # no interaction stage, heredity moot. Exists for validation reach: the
    # 2^p - 1 model space enumerates exactly to p ~ 20. Under per_k slab
    # scaling the R2 distribution is insensitive to the candidate count
    # (total signal variance ~ tau2 * sigma2 regardless of k), so tau2 stays
    # shared with the interactions prior.
    include_interactions: bool = True

    # ---- structure, effect scale and predictors --------------------------
    #
    # Structure. reject_empty=True rejects the whole draw when the active set
    # is empty; False makes the intercept-only model a legal draw: an
    # amortized model must be able to report "nothing here", and the
    # rejection normalizer is exactly what forbids that.
    reject_empty: bool = False
    # Heredity is a per-dataset latent mode drawn from (strong, weak, none)
    # weights, not a hard rule: a hard strong-heredity rule assigns zero
    # prior mass to orphan interactions, which do occur in real data, and
    # makes any benchmark unfair to unstructured baselines. (1, 0, 0)
    # is pure strong heredity. reject_empty=True requires
    # (1, 0, 0): with a mode mixture, "reject the whole draw" is ambiguous
    # (mode redrawn inside vs outside the loop = two different laws).
    heredity_weights: tuple[float, float, float] = (0.6, 0.25, 0.15)
    # Effect scale. c is the slab variance (per effect under
    # slab_scaling="fixed"; the c of c/k under "per_k"), marginalized over
    # a truncated lognormal rather than fixed: a fixed c asserts a typical
    # effect size that varies by subfield, and asking the user to supply it
    # moves a decision they cannot make into their lap.
    #
    # Calibration (fixed slab, count prior): median 0.15, logsd 0.6, trunc
    # [0.01, 1.5] puts the per-active-main standardized effect
    # |beta_j| sd(Z_j)/sd(y) at quartiles 0.09/0.20/0.34, matching the
    # meta-analytic norms for published psychology correlations (Gignac &
    # Szodorai 2016: |r| quartiles .11/.19/.29); interactions land at half
    # that (median 0.15, Aguinis); R2 p5/50/95 = 0.00/0.21/0.69; |y|>8
    # escape rate 7.7e-5 < 5e-4.
    random_c: bool = True
    c_logmean: float = math.log(0.15)
    c_logsd: float = 0.6
    c_min: float = 0.01
    c_max: float = 1.5
    # Interaction slab ratio. NOTE this multiplies a VARIANCE: rho = 0.5
    # means interactions have sd ratio sqrt(0.5) = 0.71 relative to mains,
    # not 0.5. With random_rho_int it is drawn per dataset, so "interactions
    # are smaller" is not asserted by the prior; the support includes
    # rho = 1 and values above it. Inert when include_interactions is
    # False. Identifiability is weak (k_int is often 0 or 1), so the
    # practical effect is a heavier-tailed interaction slab rather than
    # per-dataset adaptivity.
    # Default: rho fixed at 0.5, since interaction effects run smaller than
    # mains (moderated-regression literature, e.g. Aguinis), and a fixed rho
    # removes a quadrature axis from the exact reference.
    random_rho_int: bool = False
    rho_int_fixed: float = 0.5
    rho_logmean: float = math.log(0.5)
    rho_logsd: float = 0.5
    rho_min: float = 0.1
    rho_max: float = 1.5
    # X generation. Columns split between a generic-correlation block and a
    # factor block by a per-dataset share f: real datasets mix DGPs:
    # subscale items with strong factor structure next to sociodemographic
    # columns with none. f = 0 is the generic-only case and carries explicit
    # mass because round(f * p) alone makes pure-generic datasets rare (~2%
    # at p = 30), and that regime is what most benchmark datasets look like.
    f_zero_prob: float = 0.15  # 1.0 = never a factor block
    f_beta_a: float = 1.5  # f ~ Beta(1.5, 3.0): mean 1/3, tilted generic
    f_beta_b: float = 3.0
    cross_block_w_max: float = 0.35  # generic columns' loading on the factors
    # Factor block. Small-support factors are the point: a two-variable
    # factor with high communality IS a near-duplicate pair, so the
    # near-duplicate regime is a corner of this space rather than a
    # separate overlay mechanism (an overlay would make redundancy a
    # deterministic function of p and leave a correlation-magnitude gap
    # below its floor). The communality regime is drawn per dataset, so
    # some datasets have weak factor structure and some have nearly
    # redundant items, with everything in between.
    factor_count_frac_max: float = 1 / 3  # m ~ Uniform{1..max(1, ceil(p_fac * frac))}
    h2_mean_low: float = 0.1
    h2_mean_high: float = 0.92
    h2_concentration_low: float = 4.0
    h2_concentration_high: float = 40.0
    h2_cap: float = 0.98  # keeps the uniqueness Psi >= 0.02
    cross_loading_share: float = 0.15  # fraction of h2 spread over non-primary factors
    neg_loading_prob: float = 0.3  # reverse-coded items
    # High-communality (near-duplicate) REGIME. A second mixture component on
    # the same per-dataset communality regime the block already draws, not
    # an overlay mechanism, so the "redundancy is a corner of the factor
    # space" property above is preserved. Coefficient error grows sharply
    # beyond max |r| ~ 0.95; the standard band alone reaches that in only
    # ~0.8% of p=5 draws and NEVER reaches 0.99 (h2_cap bounds |r| at 0.98),
    # while real designs sit there routinely. h2_high_prob = 0.0 disables
    # the regime (gated before the rng is touched).
    h2_high_prob: float = 0.15  # P(factor block uses the high regime)
    h2_high_mean_low: float = 0.90  # high regime: h2_mean ~ U(this, h2_high_cap)
    h2_high_cap: float = 0.995  # high-regime h2 clip (uniqueness Psi >= 0.005)
    # Column marginals via Gaussian copula (NORTA). Weights over
    # MARGINAL_TYPES in canonical tuple-of-pairs form (frozen-safe); YAML
    # configs write a mapping, coerced via canonical_marginal_weights.
    # Conservative mix: gaussian-dominant, so Gaussian columns stay the mode;
    # (("gaussian", 1.0),) is Gaussian-only.
    marginal_weights: tuple[tuple[str, float], ...] = (
        ("gaussian", 0.60),
        ("likert", 0.20),
        ("binary", 0.10),
        ("count", 0.05),
        ("skewed", 0.05),
    )
    #
    # Binary base rates stay >= 0.2: products of two low-base-rate
    # standardized binaries produce a handful of enormous values, and
    # interaction columns are not re-standardized, so the implied R2
    # blows up.
    binary_rate_low: float = 0.2
    binary_rate_high: float = 0.5
    likert_levels_min: int = 2
    likert_levels_max: int = 7

    def __post_init__(self) -> None:
        """Cross-field consistency; raise, never mutate (dataclass is frozen)."""
        hw = self.heredity_weights
        if len(hw) != 3 or any(w < 0 for w in hw) or abs(sum(hw) - 1.0) > 1e-9:
            raise ValueError(
                f"heredity_weights must be 3 nonnegative weights summing to 1, got {hw}"
            )
        if self.reject_empty and tuple(hw) != (1.0, 0.0, 0.0):
            raise ValueError(
                "reject_empty=True is only defined under pure strong heredity "
                "(heredity_weights == (1, 0, 0)): with a mode mixture, rejecting "
                "the whole draw is ambiguous — the mode redrawn inside vs outside "
                "the rejection loop are two different laws"
            )
        if self.reject_empty and self.sparsity_law != "rate":
            raise ValueError(
                "reject_empty is a rate-law mechanism; under the count law the "
                "null mass is expressed directly as k_main_zero (and the mixture "
                "law's rate component keeps the empty model legal)"
            )
        if self.sparsity_law == "mixture" and not 0.0 < self.law_count_share < 1.0:
            raise ValueError(
                f"law_count_share must lie strictly inside (0, 1) under the mixture "
                f"law (use sparsity_law='count'/'rate' for the pure laws), got "
                f"{self.law_count_share}"
            )
        names = [t for t, _ in self.marginal_weights]
        if set(names) - set(MARGINAL_TYPES) or len(set(names)) != len(names):
            raise ValueError(
                f"marginal_weights names must be unique members of {MARGINAL_TYPES}, "
                f"got {names}"
            )
        probs = [w for _, w in self.marginal_weights]
        if any(w < 0 for w in probs) or abs(sum(probs) - 1.0) > 1e-9:
            raise ValueError(
                f"marginal_weights must be nonnegative and sum to 1, got {self.marginal_weights}"
            )
        checks = [
            (0.0 <= self.pi_main_low < self.pi_main_high <= 1.0, "pi_main range"),
            (0.0 <= self.pi_int_low < self.pi_int_high <= 1.0, "pi_int range"),
            (0.0 <= self.k_main_zero < 1.0 and 0.0 <= self.k_int_zero < 1.0,
             "count-law zero-inflation"),
            (self.k_main_mean >= 1.0 and self.k_int_mean >= 1.0,
             "count-law means"),
            (0.0 < self.c_min < self.c_max and self.c_logsd > 0.0, "c hyperprior"),
            (0.0 < self.rho_min < self.rho_max and self.rho_logsd > 0.0, "rho hyperprior"),
            (self.rho_int_fixed > 0.0, "rho_int_fixed"),
            (0.0 <= self.f_zero_prob <= 1.0, "f_zero_prob"),
            (self.f_beta_a > 0.0 and self.f_beta_b > 0.0, "f beta shape"),
            (0.0 <= self.cross_block_w_max < 1.0, "cross_block_w_max"),
            (0.0 < self.factor_count_frac_max <= 1.0, "factor_count_frac_max"),
            (0.0 < self.h2_mean_low < self.h2_mean_high < 1.0, "h2 mean range"),
            (0.0 < self.h2_concentration_low <= self.h2_concentration_high, "h2 concentration"),
            (self.h2_mean_high <= self.h2_cap < 1.0, "h2_cap"),
            (0.0 <= self.h2_high_prob <= 1.0, "h2_high_prob"),
            (
                self.h2_high_prob <= 0.0
                or self.h2_mean_low < self.h2_high_mean_low <= self.h2_high_cap < 1.0,
                "h2 high regime range",
            ),
            (0.0 <= self.cross_loading_share < 1.0, "cross_loading_share"),
            (0.0 <= self.neg_loading_prob <= 1.0, "neg_loading_prob"),
            (0.0 < self.binary_rate_low <= self.binary_rate_high < 1.0, "binary rate range"),
            (2 <= self.likert_levels_min <= self.likert_levels_max, "likert levels"),
        ]
        for ok, name in checks:
            if not ok:
                raise ValueError(f"invalid PriorConfig: {name} out of range")

    def marginal_probs(self) -> np.ndarray:
        """Marginal-type probabilities as a vector over MARGINAL_TYPES."""
        w = dict(self.marginal_weights)
        return np.array([w.get(t, 0.0) for t in MARGINAL_TYPES], dtype=float)

    def is_gaussian_only(self) -> bool:
        """True iff all marginal mass is on the gaussian type;
        the copula stage is skipped entirely in that case."""
        return float(self.marginal_probs()[1:].sum()) == 0.0


class Dataset(NamedTuple):
    """One synthetic dataset with its ground truth.

    The trailing fields default to the fixed-scale configuration (c = tau2 =
    0.5, rho = 1, strong heredity), so a Dataset can be built from (X, y,
    gamma, beta, sigma2) alone; `sample_dataset` always records the drawn values.
    """

    X: np.ndarray  # (n, p) z-standardized features
    y: np.ndarray  # (n,) responses, never standardized
    gamma: np.ndarray  # (d,) bool active-set vector; gamma[0] (intercept) always True
    beta: np.ndarray  # (d,) coefficients; inactive entries exactly 0.0
    sigma2: float  # noise variance
    c: float = 0.5  # slab scale actually used (== cfg.tau2 unless random_c)
    rho: float = 1.0  # interaction slab variance ratio actually used
    heredity_mode: str = "strong"  # heredity mode of this dataset's gamma draw


def interaction_pairs(p: int, cfg: PriorConfig | None = None) -> list[tuple[int, int]]:
    """Canonical interaction ordering: itertools.combinations(range(p), 2).
    Empty under the mains-only variant (pass the cfg to get variant-aware
    structure; omitting it means the full interaction basis)."""
    if cfg is not None and not cfg.include_interactions:
        return []
    return list(itertools.combinations(range(p), 2))


def n_effects(p: int, cfg: PriorConfig | None = None) -> int:
    """d = 1 + p + C(p, 2): intercept, mains, pairwise interactions
    (1 + p under the mains-only variant)."""
    if cfg is not None and not cfg.include_interactions:
        return 1 + p
    return 1 + p + p * (p - 1) // 2


def _pairs_array(p: int, cfg: PriorConfig | None = None) -> np.ndarray:
    return np.asarray(interaction_pairs(p, cfg), dtype=int).reshape(-1, 2)


def build_design(X: np.ndarray, cfg: PriorConfig | None = None) -> np.ndarray:
    """Full design [1, X, interactions] in canonical order ([1, X] under the
    mains-only variant).

    X must already be standardized; interaction columns are the exact
    elementwise products of the given columns and are NOT re-standardized.
    This definition is shared by the generator, the exact reference, and the
    probing basis (linearpfn.probe): one prior, one implementation.
    """
    n, p = X.shape
    pairs = _pairs_array(p, cfg)
    cols = [np.ones((n, 1)), X]
    if len(pairs):
        cols.append(X[:, pairs[:, 0]] * X[:, pairs[:, 1]])
    return np.concatenate(cols, axis=1)


def trunc_lognormal_draw(
    rng: np.random.Generator, logmean: float, logsd: float, lo: float, hi: float
) -> float:
    """One draw from a lognormal truncated to [lo, hi], via the inverse CDF
    (ndtri) — exactly one rng.random() call, no rejection loop, so the RNG
    stream cost is config-independent. This is the single definition of the
    c/rho law: the exact reference integrates the SAME density
    (`trunc_lognormal_logpdf`).
    """
    a = ndtr((math.log(lo) - logmean) / logsd)
    b = ndtr((math.log(hi) - logmean) / logsd)
    u = rng.random()
    return float(math.exp(logmean + logsd * ndtri(a + u * (b - a))))


def trunc_lognormal_logpdf(
    x: np.ndarray | float, logmean: float, logsd: float, lo: float, hi: float
) -> np.ndarray | float:
    """Log density (in x, 1/x Jacobian included) of the truncated lognormal;
    -inf outside [lo, hi]. Vectorized over x."""
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    a = ndtr((math.log(lo) - logmean) / logsd)
    b = ndtr((math.log(hi) - logmean) / logsd)
    safe = np.clip(x_arr, 1e-300, None)
    z = (np.log(safe) - logmean) / logsd
    lp = -np.log(safe * logsd) - 0.5 * math.log(2.0 * math.pi) - 0.5 * z**2 - math.log(b - a)
    out = np.where((x_arr >= lo) & (x_arr <= hi), lp, -np.inf)
    return float(out[0]) if np.ndim(x) == 0 else out


def slab_variance(
    k: int,
    cfg: PriorConfig,
    c: float | None = None,
    rho: float | None = None,
    is_interaction: bool = False,
) -> float:
    """Per-effect slab variance for a model with k active non-intercept
    effects (k INCLUDES active interactions).

    c is the dataset-level slab scale: it defaults to cfg.tau2 when
    random_c=False; under random_c=True the caller MUST pass the drawn (or
    quadrature-node) value: a silent default would evaluate a different
    prior than the one sampled. Mains: c under "fixed", c / k under "per_k" (a
    per-model constant, so NIG conjugacy is untouched). Interactions:
    additionally multiplied by the VARIANCE ratio rho (defaults to
    cfg.rho_int_fixed when random_rho_int=False, same must-pass contract
    otherwise). The intercept never goes through here (tau2_intercept).
    """
    if c is None:
        if cfg.random_c:
            raise ValueError("random_c=True: pass the drawn/node c explicitly")
        c = cfg.tau2
    if cfg.slab_scaling == "fixed":
        base = c
    elif cfg.slab_scaling == "per_k":
        base = c / max(k, 1)
    else:
        raise ValueError(f"unknown slab_scaling {cfg.slab_scaling!r}")
    if is_interaction:
        if rho is None:
            if cfg.random_rho_int:
                raise ValueError("random_rho_int=True: pass the drawn/node rho explicitly")
            rho = cfg.rho_int_fixed
        base = base * rho
    return base


def _p_from_d(d: int, cfg: PriorConfig) -> int:
    """Invert d = n_effects(p, cfg), exactly, for both variants."""
    if not cfg.include_interactions:
        return d - 1
    p = (math.isqrt(1 + 8 * (d - 1)) - 1) // 2  # d - 1 = p (p + 1) / 2
    if n_effects(p, cfg) != d:
        raise ValueError(f"no valid p with n_effects(p) == {d}")
    return p


def prior_scale_diag(
    gamma: np.ndarray,
    cfg: PriorConfig,
    c: float | None = None,
    rho: float | None = None,
) -> np.ndarray:
    """Diagonal of V0 for the model gamma: [tau2_intercept, tau2_eff for
    mains, tau2_eff * rho for interaction slots].

    The slab is beta | sigma2 ~ N(0, sigma2 * diag(V0)) on included effects;
    tau2_eff comes from `slab_variance` with k = active non-intercept count
    (interactions included). Entries at inactive slots are the variances
    that would apply if included. c/rho contracts as in `slab_variance`.
    """
    g = np.asarray(gamma, dtype=bool)
    k = int(g[1:].sum())
    v = np.full(g.size, slab_variance(k, cfg, c=c, rho=rho))
    p = _p_from_d(g.size, cfg)
    if g.size > 1 + p:  # interaction slots exist
        v[1 + p :] = slab_variance(k, cfg, c=c, rho=rho, is_interaction=True)
    v[0] = cfg.tau2_intercept
    return v


def sample_heredity_mode(rng: np.random.Generator, cfg: PriorConfig) -> str:
    """Draw the per-dataset heredity mode from `cfg.heredity_weights`.

    A degenerate weight vector (all mass on one mode, e.g. pure strong
    heredity (1, 0, 0)) returns that mode WITHOUT touching the rng, so a
    fixed mode consumes no random numbers.
    """
    for weight, mode in zip(cfg.heredity_weights, HEREDITY_MODES, strict=True):
        if weight >= 1.0:
            return mode
    u = rng.random()
    edge = 0.0
    for weight, mode in zip(cfg.heredity_weights, HEREDITY_MODES, strict=True):
        edge += weight
        if u < edge:
            return mode
    return HEREDITY_MODES[-1]  # float-sum slack: weights sum to 1 within 1e-9


def _mode_eligibility(mains: np.ndarray, pairs: np.ndarray, mode: str) -> np.ndarray:
    """Eligible-pair mask under a heredity mode; mains (..., p) bool."""
    a = mains[..., pairs[:, 0]]
    b = mains[..., pairs[:, 1]]
    if mode == "strong":
        return a & b
    if mode == "weak":
        return a | b
    if mode == "none":
        return np.ones_like(a)
    raise ValueError(f"unknown heredity mode {mode!r}")


def sample_gamma(
    rng: np.random.Generator,
    p: int,
    cfg: PriorConfig,
    mode: str | None = None,
) -> np.ndarray:
    """Sample the active-set vector gamma (step 4), length d, intercept True.

    Heredity: an interaction is eligible iff its parents satisfy the
    dataset's heredity mode — both active (strong), at least one (weak), or
    unconditionally (none). The mode is drawn here from
    `cfg.heredity_weights` unless the caller passes one in (`sample_dataset`
    does, so the draw can be recorded on the Dataset).

    Rejection: under reject_empty=True (which requires pure strong
    heredity and the rate prior) a draw with no active main — under strong heredity
    exactly the empty active set — is rejected WHOLE, (pi_main, pi_int) and
    all Bernoullis, and redrawn; under reject_empty=False the empty active
    set is a legal draw. `log_prior_gamma` computes the marginal law of the
    returned gamma in closed form.
    Mains-only variant: the interaction stage is dropped (d = 1 + p) and
    the mode is moot.
    """
    pairs = _pairs_array(p, cfg)
    d = n_effects(p, cfg)
    if mode is None:
        mode = sample_heredity_mode(rng, cfg) if cfg.include_interactions else "strong"
    law = cfg.sparsity_law
    if law == "mixture":
        # One uniform picks the component; the chosen pure law then runs
        # verbatim. Gated so the pure laws never see this extra draw.
        law = "count" if rng.random() < cfg.law_count_share else "rate"
    if law == "count":
        # Count prior: draw the counts directly (each count consumes exactly
        # one rng.random()), then a uniform subset. No rejection loop: the
        # null mass is k_main_zero.
        from linearpfn._counts import zt_geom_draw

        k_m = zt_geom_draw(rng, p, cfg.k_main_zero, cfg.k_main_mean)
        mains = np.zeros(p, dtype=bool)
        if k_m:
            mains[rng.choice(p, size=k_m, replace=False)] = True
        ints = np.zeros(len(pairs), dtype=bool)
        if cfg.include_interactions:
            elig = _mode_eligibility(mains, pairs, mode)
            n_elig = int(elig.sum())
            k_i = zt_geom_draw(rng, n_elig, cfg.k_int_zero, cfg.k_int_mean)
            if k_i:
                idx = np.flatnonzero(elig)
                ints[idx[rng.choice(n_elig, size=k_i, replace=False)]] = True
        gamma = np.empty(d, dtype=bool)
        gamma[0] = True
        gamma[1 : 1 + p] = mains
        gamma[1 + p :] = ints
        return gamma
    while True:
        pi_main = rng.uniform(cfg.pi_main_low, cfg.pi_main_high)
        mains = rng.random(p) < pi_main
        if cfg.reject_empty and not mains.any():
            continue  # empty active set: reject the whole step-4 draw incl. the pis
        ints = np.zeros(len(pairs), dtype=bool)
        if cfg.include_interactions:
            # pi_int is drawn even when no pair is eligible, so the stream
            # shape does not depend on the mains draw.
            pi_int = rng.uniform(cfg.pi_int_low, cfg.pi_int_high)
            elig = _mode_eligibility(mains, pairs, mode)
            n_elig = int(elig.sum())
            if n_elig:
                ints[elig] = rng.random(n_elig) < pi_int
        gamma = np.empty(d, dtype=bool)
        gamma[0] = True
        gamma[1 : 1 + p] = mains
        gamma[1 + p :] = ints
        return gamma


def place_prior_params(
    rng: np.random.Generator,
    X: np.ndarray,
    cfg: PriorConfig,
) -> Dataset:
    """Steps 4-7 of the prior on a GIVEN standardized X: draw (mode, gamma,
    c, rho, sigma2, beta) and y exactly as `sample_dataset` does after its X
    stage — the one implementation of the parameter law, shared by the
    synthetic sampler and the semi-synthetic benchmark (prior-consistent
    placement on real designs). The rng continues whatever stream the caller
    is on; `sample_dataset_with_meta` calls this right after `sample_X`."""
    n, p = X.shape
    mode = sample_heredity_mode(rng, cfg) if cfg.include_interactions else "strong"
    gamma = sample_gamma(rng, p, cfg, mode=mode)
    if cfg.random_c:  # config-gated before the rng
        c = trunc_lognormal_draw(rng, cfg.c_logmean, cfg.c_logsd, cfg.c_min, cfg.c_max)
    else:
        c = float(cfg.tau2)
    if cfg.random_rho_int and cfg.include_interactions:
        rho = trunc_lognormal_draw(rng, cfg.rho_logmean, cfg.rho_logsd, cfg.rho_min, cfg.rho_max)
    else:
        rho = float(cfg.rho_int_fixed)
    sigma2 = 1.0 / rng.gamma(cfg.a0, 1.0 / cfg.b0)  # InvGamma(a0, b0)
    scales = prior_scale_diag(gamma, cfg, c=c, rho=rho)
    beta = np.zeros(n_effects(p, cfg))
    beta[gamma] = rng.standard_normal(int(gamma.sum())) * np.sqrt(scales[gamma] * sigma2)
    y = build_design(X, cfg) @ beta + math.sqrt(sigma2) * rng.standard_normal(n)
    return Dataset(
        X=X,
        y=y,
        gamma=gamma,
        beta=beta,
        sigma2=float(sigma2),
        c=c,
        rho=rho,
        heredity_mode=mode,
    )


def sample_dataset_with_meta(
    rng: np.random.Generator,
    cfg: PriorConfig,
    n: int | None = None,
    p: int | None = None,
) -> tuple[Dataset, XMeta]:
    """`sample_dataset` plus the X-structure ground truth (XMeta), for the
    preflight diagnostics and validation strata. Same implementation, same
    RNG stream — the meta is assembled from values drawn anyway."""
    if n is None:
        n = int(rng.integers(cfg.n_min, cfg.n_max + 1))
    if p is None:
        p = int(rng.integers(cfg.p_min, cfg.p_max + 1))
    X, meta = sample_X(rng, n, p, cfg)
    return place_prior_params(rng, X, cfg), meta


def sample_dataset(
    rng: np.random.Generator,
    cfg: PriorConfig,
    n: int | None = None,
    p: int | None = None,
) -> Dataset:
    """Draw one dataset from the prior (steps 1-7 of the module docstring).

    n and p can be overridden for tests and calibration sweeps; by default
    they are drawn uniformly from the configured ranges.
    """
    return sample_dataset_with_meta(rng, cfg, n=n, p=p)[0]


def _log_uniform_bernoulli_marginal(
    k: np.ndarray | int,
    m: np.ndarray | int,
    low: float,
    high: float,
) -> np.ndarray:
    """log P(specific k-of-m binary vector), actives iid Bern(pi), pi ~ U(low, high).

        P = (1/(high-low)) * ∫ pi^k (1-pi)^(m-k) dpi   over [low, high]
          = B(k+1, m-k+1) * [I_high(k+1, m-k+1) - I_low(k+1, m-k+1)] / (high-low)

    with B the beta function and I_x the regularized incomplete beta
    (∫_0^x t^(a-1) (1-t)^(b-1) dt = B(a, b) I_x(a, b)). There is no C(m, k)
    factor because this is the probability of one SPECIFIC vector, not of the
    count. The m = 0 edge degrades gracefully to P = 1.

    Numerics: when I_low > 0.5 both regularized values sit near 1 and their
    difference cancels; the complement identity I_x(a,b) = 1 - I_{1-x}(b,a)
    rewrites the difference in the well-conditioned tail. True underflow is
    unreachable for the p <= 30 regime used here, so it raises.
    """
    k_arr = np.asarray(k, dtype=float)
    m_arr = np.asarray(m, dtype=float)
    a = k_arr + 1.0
    b = m_arr - k_arr + 1.0
    i_low = betainc(a, b, low)
    i_high = betainc(a, b, high)
    diff = i_high - i_low
    use_comp = i_low > 0.5
    if np.any(use_comp):
        comp = betainc(b, a, 1.0 - low) - betainc(b, a, 1.0 - high)
        diff = np.where(use_comp, comp, diff)
    if np.any(diff <= 0.0):
        raise FloatingPointError(
            f"betainc difference underflow for k={k}, m={m}, interval [{low}, {high}]"
        )
    return betaln(a, b) + np.log(diff) - math.log(high - low)


def log_p_empty(p: int, cfg: PriorConfig) -> float:
    """log P(no main effect active) before rejection; closed form
    [(1-l)^(p+1) - (1-u)^(p+1)] / ((p+1)(u-l)) with (l, u) the pi_main range.

    Under strong heredity the empty active set is exactly this event, so it
    is the rejection normalizer in `log_prior_gamma`.
    """
    return float(_log_uniform_bernoulli_marginal(0, p, cfg.pi_main_low, cfg.pi_main_high))


def expected_active_count(p: int, cfg: PriorConfig) -> float:
    """Expected #active non-intercept effects, closed form over the heredity
    mixture — conditioned on a non-empty active set when reject_empty=True
    (matching the sampler's rejection law), unconditional otherwise.

    Given pi_main, mains are iid Bern(pi_main) so E[k_main] = p E[pi]. A
    pair is eligible with probability g_h(pi) per mode — pi^2 (strong),
    2 pi - pi^2 (weak), 1 (none) — so E[k_int] = C(p,2) E[g_h(pi)] E[pi_int]
    averaged over the mode weights. The empty event contributes zero
    actives, hence conditioning on non-empty divides by 1 - P_empty. For
    pi ~ U(l, u): E[pi] = (l+u)/2, E[pi^2] = (u^3 - l^3) / (3(u - l)).
    Mixture law: the share-weighted average of the two pure expectations.
    """
    if cfg.sparsity_law == "mixture":
        w = cfg.law_count_share
        return w * expected_active_count(p, replace(cfg, sparsity_law="count")) + (
            1.0 - w
        ) * expected_active_count(p, replace(cfg, sparsity_law="rate"))
    if cfg.sparsity_law == "count":
        # Exact finite sum over k_main (the interaction pool is a
        # deterministic function of k_main per mode).
        from linearpfn._counts import zt_geom_logpmf, zt_geom_mean

        raw = zt_geom_mean(p, cfg.k_main_zero, cfg.k_main_mean)
        if cfg.include_interactions:
            pools = {
                "strong": lambda k: k * (k - 1) // 2,
                "weak": lambda k: p * (p - 1) // 2 - (p - k) * (p - k - 1) // 2,
                "none": lambda k: p * (p - 1) // 2,
            }
            for k in range(p + 1):
                pk = math.exp(
                    float(zt_geom_logpmf(k, p, cfg.k_main_zero, cfg.k_main_mean))
                )
                for w, mode in zip(cfg.heredity_weights, HEREDITY_MODES, strict=True):
                    if w:
                        raw += pk * w * zt_geom_mean(
                            pools[mode](k), cfg.k_int_zero, cfg.k_int_mean
                        )
        return raw
    lo, hi = cfg.pi_main_low, cfg.pi_main_high
    e_pi = (lo + hi) / 2.0
    raw = p * e_pi
    if cfg.include_interactions:
        e_pi2 = (hi**3 - lo**3) / (3.0 * (hi - lo))
        e_elig = {"strong": e_pi2, "weak": 2.0 * e_pi - e_pi2, "none": 1.0}
        mix = sum(
            w * e_elig[mode]
            for w, mode in zip(cfg.heredity_weights, HEREDITY_MODES, strict=True)
        )
        e_int = (cfg.pi_int_low + cfg.pi_int_high) / 2.0
        raw += p * (p - 1) / 2.0 * mix * e_int
    if cfg.reject_empty:
        raw /= 1.0 - math.exp(log_p_empty(p, cfg))
    return raw


def log_prior_gamma(
    gammas: np.ndarray,
    p: int,
    cfg: PriorConfig,
) -> np.ndarray | float:
    """Exact log marginal prior of active-set vectors, with (pi_main,
    pi_int) and the heredity mode integrated out. Vectorized: accepts one
    (d,) gamma or a stack (M, d); this is the enumeration hot path.

    Derivation. Write gamma = (gamma_main, gamma_int), k = #active mains,
    k_int = #active interactions.

    1. Mains are iid Bern(pi_main) given pi_main, so a SPECIFIC vector with
       k ones has probability pi^k (1-pi)^(p-k); marginalizing
       pi_main ~ U(l, u) gives the closed Beta form of
       `_log_uniform_bernoulli_marginal` (no C(p, k) factor).
    2. Given gamma_main and the heredity mode h, the eligible-pair count is
       a deterministic function of k: m_strong = C(k, 2) (both parents
       active), m_weak = C(p, 2) - C(p - k, 2) (at least one),
       m_none = C(p, 2). Interactions outside the eligible set have prior
       mass zero under h; inside they are iid Bern(pi_int) with
       pi_int ~ U(l', u') independent of pi_main — the same Beta form with
       (k_int, m_h, l', u').
    3. The mode is latent, so the marginal law is the mixture
       logsumexp_h [log w_h + log p(gamma | h)]. A gamma that violates
       strong heredity gets -inf from the strong branch but finite mass
       from the weak or none branches — the network can be moved by
       evidence instead of refusing outright. Zero-weight modes are never
       evaluated, so pure strong heredity (1, 0, 0) stays a single strong term.
    4. reject_empty=True (requires pure strong heredity and the rate prior): the
       sampler rejects the whole step-4 draw until the active set is
       non-empty. Because strong heredity forces interactions off whenever
       no main is active, "empty" is exactly {gamma_main = 0}; conditioning
       divides by 1 - P_empty (`log_p_empty`) and the empty gamma gets
       -inf. Under reject_empty=False the empty model is in-support with
       its unadjusted mixture mass.

    Raises if the intercept slot is not set (the intercept is in every
    model). Mains-only variant: steps 2-3 disappear (d = 1 + p).
    """
    arr = np.asarray(gammas, dtype=bool)
    single = arr.ndim == 1
    g = np.atleast_2d(arr)
    d = n_effects(p, cfg)
    if g.shape[1] != d:
        raise ValueError(f"gamma must have length {d} for p={p}, got {g.shape[1]}")
    if not g[:, 0].all():
        raise ValueError("the intercept slot gamma[0] must always be active")
    mains = g[:, 1 : 1 + p]
    k_main = mains.sum(axis=1)
    nonempty = mains.any(axis=1)
    if cfg.sparsity_law == "mixture":
        # Mixture law: the marginal over gamma is the convex combination of
        # the two pure marginals (each with its own mode mixture inside).
        # Recurse on the pure-law configs so the branches below stay the
        # ONLY implementation of each law.
        w = cfg.law_count_share
        lp_count = log_prior_gamma(g, p, replace(cfg, sparsity_law="count"))
        lp_rate = log_prior_gamma(g, p, replace(cfg, sparsity_law="rate"))
        out = np.logaddexp(math.log(w) + lp_count, math.log1p(-w) + lp_rate)
        return float(out[0]) if single else out
    if cfg.sparsity_law == "count":
        # Count law: P(specific mains vector) = P(k_main) / C(p, k_main);
        # per mode, P(specific ints vector) = P(k_int | pool) / C(pool, k_int).
        from linearpfn._counts import log_choose, zt_geom_logpmf

        lp_main = zt_geom_logpmf(
            k_main, p, cfg.k_main_zero, cfg.k_main_mean
        ) - log_choose(p, k_main)

        def lp_int_fn(k_eff: np.ndarray, m_elig: np.ndarray) -> np.ndarray:
            return zt_geom_logpmf(
                k_eff, m_elig, cfg.k_int_zero, cfg.k_int_mean
            ) - log_choose(m_elig, k_eff)
    else:
        lp_main = _log_uniform_bernoulli_marginal(
            k_main, p, cfg.pi_main_low, cfg.pi_main_high
        )
        if cfg.reject_empty:
            lp_main = lp_main - math.log1p(-math.exp(log_p_empty(p, cfg)))

        def lp_int_fn(k_eff: np.ndarray, m_elig: np.ndarray) -> np.ndarray:
            return _log_uniform_bernoulli_marginal(
                k_eff, m_elig, cfg.pi_int_low, cfg.pi_int_high
            )
    if not cfg.include_interactions:
        out = np.where(nonempty, lp_main, -np.inf) if cfg.reject_empty else lp_main
        return float(out[0]) if single else out
    pairs = _pairs_array(p, cfg)
    ints = g[:, 1 + p :]
    k_int = ints.sum(axis=1)
    branches: list[np.ndarray] = []
    for weight, mode in zip(cfg.heredity_weights, HEREDITY_MODES, strict=True):
        if weight == 0.0:
            continue
        elig = _mode_eligibility(mains, pairs, mode)
        valid = ~(ints & ~elig).any(axis=1)
        if cfg.reject_empty:
            valid = valid & nonempty
        # For invalid rows use (0, 0) placeholders to stay inside each law's
        # domain; their result is overwritten with -inf below.
        m_elig = np.where(valid, elig.sum(axis=1), 0)
        k_eff = np.where(valid, k_int, 0)
        lp = lp_main + lp_int_fn(k_eff, m_elig) + math.log(weight)
        branches.append(np.where(valid, lp, -np.inf))
    out = branches[0] if len(branches) == 1 else logsumexp(np.stack(branches), axis=0)
    return float(out[0]) if single else out
