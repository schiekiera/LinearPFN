"""Single-node data-parallel training: process group setup and rank helpers.

Off by default and inert: without the launcher's environment variables
(`WORLD_SIZE` > 1) every function here reports a world of one and the trainer
runs as a single process.

The split is EXACT, not merely equivalent in distribution. One optimizer step
consumes a contiguous block of `accum_steps` stream indices; with W ranks each
rank takes every W-th index of that block (`PriorBatches(rank=, world_size=)`),
so the union over ranks is the same block the single-process run consumes. Each
rank divides its loss by its LOCAL micro-batch count, and the process group
AVERAGES gradients across ranks, so the assembled gradient equals the
single-process mean over the whole block:

    (1/W) * sum_r sum_{i in r} grad_i / (accum/W)  =  (1/accum) * sum_i grad_i

Muon sees that averaged gradient, identical on every rank, so optimizer state
and parameters stay in lockstep without any extra synchronisation.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch

__all__ = ["is_distributed", "world_size", "rank", "local_rank", "setup", "cleanup",
           "is_main", "reduce_mean", "no_sync", "HeadsAdapter", "heads_of", "unwrap"]


class HeadsAdapter(torch.nn.Module):
    """Routes `forward_with_heads` through a plain `forward`.

    DistributedDataParallel installs its gradient-synchronisation hooks on the
    WRAPPER's forward; calling a model method directly bypasses them, so the
    ranks would silently never exchange gradients. Wrapping this adapter
    instead makes the training forward the one DDP sees.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, X, y_context, split_index):
        return self.model.forward_with_heads(X, y_context, split_index)


def unwrap(model):
    """The underlying LinearPFNModel. DDP proxies `forward` and nothing else,
    so every ATTRIBUTE read (coef_head_mode, coef_bar, ...) must go through
    this; reading them off the wrapper raises AttributeError at best and would
    silently diverge at worst."""
    inner = getattr(model, "module", model)          # DDP -> HeadsAdapter
    return getattr(inner, "model", inner)            # HeadsAdapter -> the model


def heads_of(model):
    """The callable that runs the three heads: the model's own method when
    single-process, the DDP wrapper itself when distributed."""
    return getattr(model, "forward_with_heads", model)


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))


def is_distributed() -> bool:
    return world_size() > 1


def is_main() -> bool:
    """Only this rank writes checkpoints, the CSV log, tensorboard and stdout."""
    return rank() == 0


def setup() -> torch.device:
    """Join the process group and return this rank's device. No-op when off."""
    if not is_distributed():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("distributed training requires CUDA")
    torch.cuda.set_device(local_rank())
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    return torch.device("cuda", local_rank())


def cleanup() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """Average a tensor across ranks so the LOGGED loss covers the whole block,
    not just this rank's share. Inert off-distributed."""
    if not is_distributed():
        return t
    out = t.detach().clone()
    torch.distributed.all_reduce(out, op=torch.distributed.ReduceOp.SUM)
    return out / world_size()


def no_sync(model, last: bool):
    """Gradient all-reduce costs a full parameter exchange, so skip it on every
    micro-batch but the last: the local accumulations are reduced once, at the
    end, which is what the single-process run sums anyway."""
    if last or not is_distributed() or not hasattr(model, "no_sync"):
        return nullcontext()
    return model.no_sync()
