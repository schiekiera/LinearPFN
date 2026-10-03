"""Pair sufficient statistics from the raw standardized columns.

An optional input of the interaction heads (model option `pair_stats`, off in
the reported configuration). Per-column mean-pooled streams lose the row
alignment that an interaction's evidence lives in: in a linear model the
sufficient statistics for the (j, k) interaction are cross-moments ACROSS rows.
Instead of asking the backbone to preserve them, this module computes them
exactly, from the raw context columns the model already receives, and the
per-effect heads consume them as extra input features.

Per pair (j, k), six numbers:

  1. mean_r(x_j x_k)          — the pair's product first moment
  2. mean_r(x_j x_k y)        — the third moment carrying the interaction
  3. b_0                       ┐ ridge fit of y on (1, x_j, x_k, x_j x_k):
  4. b_j + b_k                 │ (G + lambda I) b = Z'y/n with G = Z'Z/n.
  5. b_j b_k                   │ b_j, b_k enter only through symmetric
  6. b_jk                      ┘ combinations — the head contract requires
                                 invariance to swapping j and k, and
                                 (b_j + b_k, b_j b_k) determines the
                                 unordered pair without information loss.

Everything is computed in fp32 with autocast disabled: the training
forward runs under bf16 autocast, and a batched linear solve neither
supports nor deserves bf16. No RNG is consumed, so the option leaves the
training data stream unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["N_PAIR_STATS", "RIDGE_LAMBDA", "pair_sufficient_stats"]

N_PAIR_STATS = 6
# Matches the probing convention (validate.ridge_lambda); applied to the
# row-normalized Gram, so conditioning is stable across n in [20, 1024].
RIDGE_LAMBDA = 1e-3


def pair_sufficient_stats(X_ctx: Tensor, y_ctx: Tensor, jj: Tensor, kk: Tensor) -> Tensor:
    """X_ctx: (B, n_ctx, p) raw standardized context columns; y_ctx:
    (B, n_ctx) raw context targets; jj, kk: (K,) pair indices in
    `interaction_pairs` order -> (B, K, N_PAIR_STATS), fp32."""
    autocast_device = X_ctx.device.type if X_ctx.device.type in ("cuda", "cpu") else "cpu"
    with torch.autocast(autocast_device, enabled=False):
        X = X_ctx.float()
        y = y_ctx.float()
        n = X.shape[1]
        x_j, x_k = X[:, :, jj], X[:, :, kk]  # (B, n, K)
        prod = x_j * x_k
        s_xx = prod.mean(dim=1)  # (B, K)
        s_xxy = (prod * y.unsqueeze(-1)).mean(dim=1)
        basis = torch.stack([torch.ones_like(x_j), x_j, x_k, prod], dim=-1)  # (B, n, K, 4)
        gram = torch.einsum("bnki,bnkj->bkij", basis, basis) / n
        rhs = torch.einsum("bnki,bn->bki", basis, y) / n
        eye = torch.eye(4, device=gram.device, dtype=gram.dtype)
        b = torch.linalg.solve(gram + RIDGE_LAMBDA * eye, rhs.unsqueeze(-1)).squeeze(-1)
        b0, b_j, b_k, b_jk = b.unbind(dim=-1)
        return torch.stack([s_xx, s_xxy, b0, b_j + b_k, b_j * b_k, b_jk], dim=-1)
