"""Batched synthetic dataset sampling for training LinearPFN.

Thin batching layer over `linearpfn.prior.sample_dataset` — the prior module
remains the single source of truth for the data-generating process; this
module only assembles torch batches and fixes the context/query split.

Batch layout: all datasets in a batch share one (n, p, split_index) draw, so
tensors are dense (B, n, p) with no padding. Rows [:split_index] are context,
rows [split_index:] are query; the query fraction is ~ Uniform(0.1, 0.5).
Ground truth (gamma, beta, sigma2) is returned with every dataset; it is
the training target of the selection and coefficient heads.

Determinism: batch `index` under `seed` is reproducible in isolation —
`sample_batch(cfg, B, seed, index)` seeds a fresh generator from
(seed, index). `PriorBatches` shards indices round-robin over dataloader
workers, so the multiset of generated batches is independent of
num_workers.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from linearpfn.prior import PriorConfig, n_effects, sample_dataset

__all__ = ["QUERY_FRACTION", "Batch", "PriorBatches", "sample_batch"]

# Share of a batch's rows that are query rows, drawn uniformly per batch.
QUERY_FRACTION = (0.1, 0.5)


class Batch(NamedTuple):
    """One training batch of B datasets with shared (n, p, split)."""

    X: torch.Tensor  # (B, n, p) float32, standardized features
    y: torch.Tensor  # (B, n) float32, raw scale (never standardized)
    gamma: torch.Tensor  # (B, d) bool ground-truth active sets, intercept True
    beta: torch.Tensor  # (B, d) float32 ground-truth coefficients
    sigma2: torch.Tensor  # (B,) float32 ground-truth noise variances
    split_index: int  # rows [:split_index] context, [split_index:] query


def sample_batch(cfg: PriorConfig, batch_size: int, seed: int, index: int) -> Batch:
    """Sample batch `index` of the deterministic stream defined by `seed`."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    n = int(rng.integers(cfg.n_min, cfg.n_max + 1))
    p = int(rng.integers(cfg.p_min, cfg.p_max + 1))
    query_fraction = rng.uniform(*QUERY_FRACTION)
    n_query = int(np.clip(round(query_fraction * n), 1, n - 2))
    d = n_effects(p, cfg)
    X = np.empty((batch_size, n, p), dtype=np.float32)
    y = np.empty((batch_size, n), dtype=np.float32)
    gamma = np.empty((batch_size, d), dtype=bool)
    beta = np.empty((batch_size, d), dtype=np.float32)
    sigma2 = np.empty(batch_size, dtype=np.float32)
    for b in range(batch_size):
        ds = sample_dataset(rng, cfg, n=n, p=p)
        X[b] = ds.X
        y[b] = ds.y
        gamma[b] = ds.gamma
        beta[b] = ds.beta
        sigma2[b] = ds.sigma2
    return Batch(
        X=torch.from_numpy(X),
        y=torch.from_numpy(y),
        gamma=torch.from_numpy(gamma),
        beta=torch.from_numpy(beta),
        sigma2=torch.from_numpy(sigma2),
        split_index=n - n_query,
    )


class PriorBatches(IterableDataset[Batch]):
    """Deterministic stream of `num_steps` freshly sampled batches.

    Batch i is `sample_batch(cfg, batch_size, seed, start_index + i)`.
    Use with DataLoader(batch_size=None); with multiple workers the index
    range is sharded round-robin, so the union over workers is exactly the
    serial stream (order may interleave). `start_index` supports resuming a
    training run at its saved step count without replaying data.

    `rank`/`world_size` add a SECOND round-robin level for data-parallel
    training: the global shard is `rank * num_workers + worker_id` with
    stride `world_size * num_workers`, so the union over (rank, worker) is
    again exactly the serial stream. Defaults (0, 1) reproduce the
    single-process iteration index for index, bit-for-bit.
    """

    def __init__(
        self,
        cfg: PriorConfig,
        batch_size: int,
        seed: int,
        num_steps: int,
        start_index: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} outside [0, {world_size})")
        self.cfg = cfg
        self.batch_size = batch_size
        self.seed = seed
        self.num_steps = num_steps
        self.start_index = start_index
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[Batch]:
        info = get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info else (0, 1)
        shard = self.rank * num_workers + worker_id
        stride = self.world_size * num_workers
        stop = self.start_index + self.num_steps
        for index in range(self.start_index + shard, stop, stride):
            yield sample_batch(self.cfg, self.batch_size, self.seed, index)

    def __len__(self) -> int:
        return self.num_steps
