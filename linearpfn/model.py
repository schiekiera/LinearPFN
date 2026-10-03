"""Transformer backbone + bar-distribution regression head for LinearPFN.

This is the neural half of amortized spike-and-slab regression: the model is
pretrained on synthetic datasets from `linearpfn.prior` and approximates the
posterior predictive distribution in one forward pass.

Attribution: the backbone is a fork of nanoTabPFN
(https://github.com/automl/nanoTabPFN, Apache License 2.0; see their
CITATION.bib). Architecture preserved from the original: per-cell embeddings
(batch, rows, columns, embedding) with alternating attention between features
and attention between datapoints, where context rows attend to themselves and
query rows attend only to the context. Modifications for LinearPFN:
regression instead of classification (the decoder emits bar-distribution
logits, trained with continuous NLL), a cleaned-up forward signature
(X, y_context, split_index), type hints, and the removal of the sklearn-style
classifier wrapper. The per-feature column streams that the selection and
coefficient heads pool over are exactly the `columns` axis of this backbone.

The bar distribution (Riemann distribution) follows the TabPFN recipe:
fixed support [-8, 8] split into 512 equal bins, plus two half-open tail
bins whose densities are half-normal with scale `tail_scale` beyond the
support borders. y under the prior is approximately unit scale, so the
support covers ~9+ sds.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from linearpfn._bardist import BarDistribution
from linearpfn._heads import CoefficientHead, SelectionHead
from linearpfn._pairstats import pair_sufficient_stats
from linearpfn.prior import interaction_pairs

__all__ = ["BarDistribution", "CoefficientHead", "LinearPFNModel", "SelectionHead"]

# Coefficient-value bar distribution (spike_slab mode). Standardized-X
# coefficients concentrate near zero (the prior's effect-size calibration puts
# the main-effect |r| median near 0.2), so the support is much tighter than
# the y head's (-8, 8) and the tail scale correspondingly smaller.
COEF_SUPPORT = (-4.0, 4.0)
COEF_BINS = 512
COEF_TAIL_SCALE = 0.5


class FeatureEncoder(nn.Module):
    """Embeds each scalar feature cell after normalizing by context statistics."""

    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: Tensor, split_index: int) -> Tensor:
        """(B, rows, cols) -> (B, rows, cols, E); normalization stats come
        from the context rows only, values clipped to [-100, 100]."""
        x = x.unsqueeze(-1)
        mean = torch.mean(x[:, :split_index], dim=1, keepdim=True)
        std = torch.std(x[:, :split_index], dim=1, keepdim=True) + 1e-20
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)


class TargetEncoder(nn.Module):
    """Embeds context targets; query slots are padded with the context mean.

    y is consumed raw (never standardized) per the LinearPFN conventions.
    """

    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_context: Tensor, num_rows: int) -> Tensor:
        """(B, n_ctx, 1) -> (B, num_rows, 1, E)."""
        mean = torch.mean(y_context, dim=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_context.shape[1], 1)
        y = torch.cat([y_context, padding], dim=1)
        return self.linear_layer(y.unsqueeze(-1))


def _sdpa_attention(mha: nn.MultiheadAttention, q_in: Tensor, k_in: Tensor, v_in: Tensor) -> Tensor:
    """Explicit-QKV scaled_dot_product_attention over the SAME parameter
    tensors as `mha` (chunked in_proj -> SDPA -> out_proj), so mha/sdpa
    checkpoints are byte-interchangeable. Mathematically identical to
    mha(q, k, v, need_weights=False)[0] for batch_first 3-D inputs; kernels
    differ, so equivalence is fp32-close, not bitwise."""
    e = q_in.shape[-1]
    heads = mha.num_heads
    w_q, w_k, w_v = mha.in_proj_weight.chunk(3)
    b_q = b_k = b_v = None
    if mha.in_proj_bias is not None:
        b_q, b_k, b_v = mha.in_proj_bias.chunk(3)

    def _split(x: Tensor, w: Tensor, b: Tensor | None) -> Tensor:
        y = F.linear(x, w, b)
        return y.view(y.shape[0], y.shape[1], heads, e // heads).transpose(1, 2)

    out = F.scaled_dot_product_attention(
        _split(q_in, w_q, b_q), _split(k_in, w_k, b_k), _split(v_in, w_v, b_v)
    )
    out = out.transpose(1, 2).reshape(q_in.shape[0], q_in.shape[1], e)
    return mha.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Two-way attention block from nanoTabPFN (itself modified from
    torch.nn.TransformerEncoderLayer): attention between features within each
    row, then attention between datapoints within each column — context rows
    attend to themselves, query rows attend only to the context — then MLP.

    `attention_impl="sdpa"` swaps each nn.MultiheadAttention CALL for an
    explicit-QKV F.scaled_dot_product_attention over the same weights (the
    modules stay, so state_dicts are identical across impls).
    `norm_placement="pre"` reorders each sublayer to x + f(norm(x)) with the
    same norm1/norm2/norm3 parameters; the model-level final LayerNorm that
    pre-norm needs lives in LinearPFNModel, not here.
    """

    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        layer_norm_eps: float = 1e-5,
        attention_impl: str = "mha",
        norm_placement: str = "post",
    ) -> None:
        super().__init__()
        if attention_impl not in ("mha", "sdpa"):
            raise ValueError(f"unknown attention_impl {attention_impl!r}")
        if norm_placement not in ("post", "pre"):
            raise ValueError(f"unknown norm_placement {norm_placement!r}")
        self.attention_impl = attention_impl
        self.norm_placement = norm_placement
        self.self_attention_between_datapoints = nn.MultiheadAttention(
            embedding_size, nhead, batch_first=True
        )
        self.self_attention_between_features = nn.MultiheadAttention(
            embedding_size, nhead, batch_first=True
        )
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, embedding_size)
        self.norm1 = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(embedding_size, eps=layer_norm_eps)

    def _attend(self, mha: nn.MultiheadAttention, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if self.attention_impl == "sdpa":
            return _sdpa_attention(mha, q, k, v)
        return mha(q, k, v, need_weights=False)[0]

    def _datapoint_attention(self, src: Tensor, split_index: int, rows_size: int) -> Tensor:
        """(B*cols, rows, E) -> attention output of the same shape: context
        rows self-attend; query rows cross-attend to the context only."""
        ctx = src[:, :split_index]
        left = self._attend(self.self_attention_between_datapoints, ctx, ctx, ctx)
        if split_index < rows_size:
            right = self._attend(
                self.self_attention_between_datapoints, src[:, split_index:], ctx, ctx
            )
            return torch.cat([left, right], dim=1)
        return left  # no query rows (e.g. selection-only forward in select())

    def forward(self, src: Tensor, split_index: int) -> Tensor:
        """(B, rows, cols, E) -> (B, rows, cols, E)."""
        if self.norm_placement == "pre":
            return self._forward_pre(src, split_index)
        batch_size, rows_size, col_size, embedding_size = src.shape
        # attention between features (within each row)
        src = src.reshape(batch_size * rows_size, col_size, embedding_size)
        src = self._attend(self.self_attention_between_features, src, src, src) + src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        # attention between datapoints (within each column)
        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, embedding_size)
        src = self._datapoint_attention(src, split_index, rows_size) + src
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        # MLP
        src = self.linear2(F.gelu(self.linear1(src))) + src
        return self.norm3(src)

    def _forward_pre(self, src: Tensor, split_index: int) -> Tensor:
        """Pre-norm variant: x + sublayer(norm(x)) per sublayer, same
        parameters as the post-norm path (norm1: feature attention, norm2:
        datapoint attention, norm3: MLP)."""
        batch_size, rows_size, col_size, embedding_size = src.shape
        h = self.norm1(src).reshape(batch_size * rows_size, col_size, embedding_size)
        h = self._attend(self.self_attention_between_features, h, h, h)
        src = src + h.reshape(batch_size, rows_size, col_size, embedding_size)
        h = self.norm2(src).transpose(1, 2).reshape(
            batch_size * col_size, rows_size, embedding_size
        )
        h = self._datapoint_attention(h, split_index, rows_size)
        src = src + h.reshape(batch_size, col_size, rows_size, embedding_size).transpose(2, 1)
        return src + self.linear2(F.gelu(self.linear1(self.norm3(src))))


class Decoder(nn.Module):
    """MLP from target-column embeddings to bar-distribution logits."""

    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(F.gelu(self.linear1(x)))


class LinearPFNModel(nn.Module):
    """nanoTabPFN backbone with bar-distribution, selection, and (optional)
    coefficient heads.

    One forward pass amortizes posterior inference: given a context
    (X_ctx, y_ctx) and query rows X_query, it emits bar-distribution logits
    for each query row's y (predictive head) and one inclusion logit per
    candidate effect (selection head — trained against the simulated
    ground-truth active sets, its converged output approximates the exact
    posterior inclusion probabilities under the prior). With
    `coef_head != "none"` it additionally emits a coefficient estimate per
    candidate effect (see CoefficientHead); the head is created only when
    requested, so a model without it has no coefficient-head state_dict keys.
    """

    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        num_outputs: int,
        pair_pooling: str = "y_aware",
        include_interactions: bool = True,
        attention_impl: str = "mha",
        norm_placement: str = "post",
        coef_head: str = "none",
        pair_stats: bool = False,
    ) -> None:
        super().__init__()
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_blocks = nn.ModuleList(
            TransformerEncoderLayer(
                embedding_size,
                num_attention_heads,
                mlp_hidden_size,
                attention_impl=attention_impl,
                norm_placement=norm_placement,
            )
            for _ in range(num_layers)
        )
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)
        # pair_stats: feed the per-effect heads the exact pair sufficient
        # statistics from the RAW columns (see linearpfn._pairstats). Off
        # adds no modules or state_dict keys.
        self.pair_stats = pair_stats
        self.selection_head = SelectionHead(
            embedding_size, mlp_hidden_size, pair_pooling, include_interactions,
            pair_stats=pair_stats,
        )
        # Created ONLY under pre-norm, so post-norm models carry no extra
        # state_dict keys; both heads (and the pair einsums in
        # forward_with_selection) read the normalized stream.
        self.final_norm: nn.LayerNorm | None = None
        if norm_placement == "pre":
            self.final_norm = nn.LayerNorm(embedding_size)
        if coef_head not in ("none", "mse", "spike_slab"):
            raise ValueError(f"unknown coef_head {coef_head!r}")
        self.coef_head_mode = coef_head
        # Same conditional-creation discipline as final_norm: no head, no
        # state_dict keys.
        self.coefficient_head: CoefficientHead | None = None
        self.coef_bar: BarDistribution | None = None
        if coef_head != "none":
            n_out = 1
            if coef_head == "spike_slab":
                # Its own support: standardized-X coefficients live on a much
                # narrower scale than y (|beta| ~ 0.2 typical under the prior's
                # effect-size calibration), so the y-head's (-8, 8) would spend
                # ~99% of its bins on empty space.
                self.coef_bar = BarDistribution(
                    support=COEF_SUPPORT, n_inner_bins=COEF_BINS, tail_scale=COEF_TAIL_SCALE
                )
                n_out = self.coef_bar.n_bins
            self.coefficient_head = CoefficientHead(
                embedding_size, mlp_hidden_size, pair_pooling, include_interactions,
                n_out=n_out, pair_stats=pair_stats,
            )

    def _backbone(self, X: Tensor, y_context: Tensor, split_index: int) -> Tensor:
        x_emb = self.feature_encoder(X, split_index)
        y_emb = self.target_encoder(y_context.unsqueeze(-1), X.shape[1])
        src = torch.cat([x_emb, y_emb], dim=2)
        for block in self.transformer_blocks:
            src = block(src, split_index=split_index)
        if self.final_norm is not None:
            src = self.final_norm(src)
        return src

    def forward(self, X: Tensor, y_context: Tensor, split_index: int) -> Tensor:
        """X: (B, n, p) all rows; y_context: (B, split_index); returns
        bar-distribution logits (B, n - split_index, num_outputs) for the
        query rows X[:, split_index:]."""
        src = self._backbone(X, y_context, split_index)
        return self.decoder(src[:, split_index:, -1, :])

    def forward_with_selection(
        self, X: Tensor, y_context: Tensor, split_index: int
    ) -> tuple[Tensor, Tensor]:
        """One backbone pass, predictive + selection heads: (predictive
        logits for query rows, selection logits (B, p + C(p,2)) in
        candidate-effect order). The coefficient head, when present, is
        reached through `forward_with_heads`; validate, probe and bench use
        this two-tuple signature.
        """
        pred, sel, _ = self.forward_with_heads(X, y_context, split_index)
        return pred, sel

    def forward_with_heads(
        self, X: Tensor, y_context: Tensor, split_index: int
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """One backbone pass, every head: (predictive logits, selection
        logits, coefficient output or None when no coefficient head).

        pair_pooling="row" additionally feeds the heads the context-mean of
        the per-ROW elementwise product of the two features' cell embeddings;
        "y_aware" adds the third-order stream E_r(e_j ⊙ e_k ⊙ e_target): an
        interaction's evidence lives in the third moment E[x_j x_k y] across
        rows — the sufficient statistic for that coefficient in a linear
        model — and per-column mean pooling destroys the row alignment that
        statistic needs. Context rows only: the target embeddings entering
        the y-aware stream are legitimate conditioning; query y never enters.
        The pooled streams are computed ONCE and shared by both per-effect
        heads — the coefficient head costs one MLP, not a second backbone.
        """
        src = self._backbone(X, y_context, split_index)
        pred = self.decoder(src[:, split_index:, -1, :])
        ctx_f = src[:, :split_index, :-1, :]
        h = ctx_f.mean(dim=1)  # (B, p, E) per-feature column streams
        h_pair = h_pair_y = h_stats = None
        if not self.selection_head.include_interactions:
            coef = None if self.coefficient_head is None else self.coefficient_head(h)
            return pred, self.selection_head(h), coef
        pairs = torch.as_tensor(interaction_pairs(h.shape[1]), device=h.device)
        jj, kk = pairs[:, 0], pairs[:, 1]
        if self.selection_head.pair_pooling in ("row", "y_aware"):
            pair_mean = torch.einsum("brje,brke->bjke", ctx_f, ctx_f) / split_index
            h_pair = pair_mean[:, jj, kk]
            if self.selection_head.pair_pooling == "y_aware":
                # decomposed two-operand form: the 3-operand einsum picks a
                # contraction path that runs out of memory at training scale;
                # (f_j * t) . f_k == f_j . f_k . t, so symmetry is preserved
                ctx_t = src[:, :split_index, -1, :]
                fy = ctx_f * ctx_t.unsqueeze(2)
                triple = torch.einsum("brje,brke->bjke", fy, ctx_f)
                h_pair_y = (triple / split_index)[:, jj, kk]
        if self.pair_stats:
            # exact statistics from the RAW columns, not the embedded stream
            h_stats = pair_sufficient_stats(X[:, :split_index], y_context, jj, kk)
        coef = (
            None if self.coefficient_head is None
            else self.coefficient_head(h, h_pair, h_pair_y, h_stats)
        )
        return pred, self.selection_head(h, h_pair, h_pair_y, h_stats), coef

    def coefficient_posterior_mean(self, sel_logits: Tensor, coef_out: Tensor) -> Tensor:
        """E[beta | data] from raw head outputs, in gamma[1:] order (the
        intercept is not modelled — it is always in the model).

        "mse" mode is trained directly on that expectation, so the head's
        output IS the estimate. "spike_slab" mode factorizes it the way the
        prior does: E[beta] = P(active) * E[beta | active], i.e. the
        selection head's PIP times the value distribution's mean — so the
        two heads combine exactly as the spike-and-slab decomposition
        prescribes, with no third quantity to calibrate.
        """
        if self.coefficient_head is None:
            raise RuntimeError("model has no coefficient head (coef_head='none')")
        if self.coef_head_mode == "mse":
            return coef_out.float()
        assert self.coef_bar is not None
        return torch.sigmoid(sel_logits.float()) * self.coef_bar.mean(coef_out.float())

    @torch.no_grad()
    def coefficient_mean(self, X: Tensor, y: Tensor) -> Tensor:
        """One-shot E[beta | data] for a single dataset. X: (n, p) context
        features, y: (n,) context targets -> (p + C(p,2),)."""
        _, sel, coef = self.forward_with_heads(X.unsqueeze(0), y.unsqueeze(0), X.shape[0])
        return self.coefficient_posterior_mean(sel[0], coef[0])

    @torch.no_grad()
    def select(self, X: Tensor, y: Tensor, threshold: float = 0.5) -> Tensor:
        """Median probability model (Barbieri & Berger): indices of candidate
        effects (0..p-1 mains, then pairs) with PIP > threshold. X: (n, p)
        context features, y: (n,) context targets."""
        _, sel = self.forward_with_selection(
            X.unsqueeze(0), y.unsqueeze(0), X.shape[0]
        )
        return torch.nonzero(torch.sigmoid(sel[0]) > threshold).flatten()
