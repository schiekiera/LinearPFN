"""Per-effect heads: selection (PIPs) and coefficients.

Private helper of `linearpfn.model`; both classes are re-exported from
`linearpfn.model`, the public entry point. The attribute names main_mlp /
interaction_mlp are part of the checkpoint contract.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from linearpfn._pairstats import N_PAIR_STATS
from linearpfn.prior import interaction_pairs

__all__ = ["CoefficientHead", "SelectionHead"]


class _PerEffectHead(nn.Module):
    """Shared structure of the per-effect heads.

    The backbone is per-cell, so a clean per-feature stream exists: column j
    of the (batch, rows, columns, embedding) tensor after the final
    transformer block (columns 0..p-1 are features; the last column is the
    target). Each feature's stream is mean-pooled over CONTEXT rows into
    h_j. Main effect j: MLP_m(h_j). Interaction (j, k):
    MLP_i(concat(h_j + h_k, h_j * h_k, ...)) — invariant to swapping j and k
    by construction. Output order matches gamma[1:]: mains 0..p-1, then
    `interaction_pairs(p)`; the intercept is excluded. Batches share one p
    (generator contract), so no padding mask is needed.

    `n_out` numbers are emitted per candidate effect: SelectionHead reads
    one as an inclusion logit, CoefficientHead reads one as a coefficient
    value or n_bins as bucket logits. The attribute names main_mlp /
    interaction_mlp are part of the checkpoint contract — renaming them
    invalidates every stored state_dict.

    `pair_stats` widens the interaction MLP's input by N_PAIR_STATS exact
    sufficient statistics per pair, computed in `linearpfn._pairstats` from
    the raw columns. Off (the default) adds no modules, shapes or state keys.
    """

    N_STREAMS = {"column": 2, "row": 3, "y_aware": 4}

    def __init__(
        self,
        embedding_size: int,
        mlp_hidden_size: int,
        pair_pooling: str = "y_aware",
        include_interactions: bool = True,
        n_out: int = 1,
        pair_stats: bool = False,
    ) -> None:
        super().__init__()
        if pair_pooling not in self.N_STREAMS:
            raise ValueError(f"unknown pair_pooling {pair_pooling!r}")
        if pair_stats and not include_interactions:
            raise ValueError("pair_stats requires include_interactions")
        self.pair_pooling = pair_pooling
        self.include_interactions = include_interactions
        self.n_out = n_out
        self.pair_stats = pair_stats
        self.main_mlp = nn.Sequential(
            nn.Linear(embedding_size, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, n_out),
        )
        if include_interactions:
            in_dim = self.N_STREAMS[pair_pooling] * embedding_size
            if pair_stats:
                in_dim += N_PAIR_STATS
            self.interaction_mlp = nn.Sequential(
                nn.Linear(in_dim, mlp_hidden_size),
                nn.GELU(),
                nn.Linear(mlp_hidden_size, n_out),
            )

    def _effects(
        self,
        h: Tensor,
        h_pair: Tensor | None = None,
        h_pair_y: Tensor | None = None,
        h_stats: Tensor | None = None,
    ) -> Tensor:
        """h: (B, p, E) pooled feature streams; h_pair / h_pair_y:
        (B, C(p,2), E) row-paired and y-aware pair streams (required by the
        "row" / "y_aware" modes); h_stats: (B, C(p,2), N_PAIR_STATS) exact
        pair statistics (required when pair_stats) -> (B, p + C(p,2), n_out);
        (B, p, n_out) under the mains-only variant."""
        p = h.shape[1]
        main = self.main_mlp(h)
        if not self.include_interactions:
            return main
        pairs = torch.as_tensor(interaction_pairs(p), device=h.device)
        h_j, h_k = h[:, pairs[:, 0]], h[:, pairs[:, 1]]
        parts = [h_j + h_k, h_j * h_k]
        if self.pair_pooling in ("row", "y_aware"):
            if h_pair is None:
                raise ValueError(f'pair_pooling="{self.pair_pooling}" requires h_pair')
            parts.append(h_pair)
        if self.pair_pooling == "y_aware":
            if h_pair_y is None:
                raise ValueError('pair_pooling="y_aware" requires h_pair_y')
            parts.append(h_pair_y)
        if self.pair_stats:
            if h_stats is None:
                raise ValueError("pair_stats=True requires h_stats")
            parts.append(h_stats.to(h.dtype))
        sym = torch.cat(parts, dim=-1)
        return torch.cat([main, self.interaction_mlp(sym)], dim=1)


class SelectionHead(_PerEffectHead):
    """Per-effect inclusion-probability head — the amortized spike-and-slab.

    One inclusion logit per candidate effect, in gamma[1:] order. No
    heredity mask is applied at the output: PIPs are marginal quantities,
    and under strong heredity the exact PIP of an interaction with unlikely
    parents is small but not zero — the head must learn this from data (the
    exact reference is the check).
    """

    def __init__(
        self,
        embedding_size: int,
        mlp_hidden_size: int,
        pair_pooling: str = "y_aware",
        include_interactions: bool = True,
        pair_stats: bool = False,
    ) -> None:
        super().__init__(
            embedding_size, mlp_hidden_size, pair_pooling, include_interactions,
            n_out=1, pair_stats=pair_stats,
        )

    def forward(
        self,
        h: Tensor,
        h_pair: Tensor | None = None,
        h_pair_y: Tensor | None = None,
        h_stats: Tensor | None = None,
    ) -> Tensor:
        """-> inclusion logits (B, p + C(p,2)); (B, p) mains-only."""
        return self._effects(h, h_pair, h_pair_y, h_stats).squeeze(-1)


class CoefficientHead(_PerEffectHead):
    """Per-effect coefficient head.

    Motivation: the predictive and selection losses carry no term in
    coefficient space, so under collinearity (where many coefficient vectors
    predict y almost identically) they give the model little gradient signal
    about which one is right, exactly where a coefficient read-out amplifies
    error by ~1/(1-r). Ground truth is free: the generator draws beta before
    building y and carries it on every batch as `Batch.beta`, aligned
    slot-for-slot with gamma.

    Two modes:

    - "mse": one value per effect, trained by MSE against the DRAWN beta
      including its zeros. Squared error is minimized by the conditional
      mean, so this converges to E[beta | data], the posterior mean.
    - "spike_slab": the true coefficient posterior is a spike-and-slab
      mixture, so a single Gaussian-style value is structurally wrong. Its
      NLL splits into a binary "is this coefficient zero" term (the
      selection BCE, reused rather than duplicated) plus a value term
      evaluated only on ACTIVE draws, emitted here as bar-distribution
      bucket logits over the coefficient support. The posterior mean is
      then read out as pip * E[beta | active].
    """

    def __init__(
        self,
        embedding_size: int,
        mlp_hidden_size: int,
        pair_pooling: str = "y_aware",
        include_interactions: bool = True,
        n_out: int = 1,
        pair_stats: bool = False,
    ) -> None:
        super().__init__(
            embedding_size, mlp_hidden_size, pair_pooling, include_interactions,
            n_out=n_out, pair_stats=pair_stats,
        )

    def forward(
        self,
        h: Tensor,
        h_pair: Tensor | None = None,
        h_pair_y: Tensor | None = None,
        h_stats: Tensor | None = None,
    ) -> Tensor:
        """-> (B, n_effects) values when n_out == 1, else bucket logits
        (B, n_effects, n_out)."""
        out = self._effects(h, h_pair, h_pair_y, h_stats)
        return out.squeeze(-1) if self.n_out == 1 else out
