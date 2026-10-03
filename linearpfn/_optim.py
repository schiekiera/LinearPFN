"""Optimizer factory + LR schedule for the training loop (private helper).

The default path (`train.optimizer: adamw`) returns plain
`torch.optim.AdamW(model.parameters(), lr=...)`: a single param group with
torch defaults.

`train.optimizer: muon` builds a single hybrid optimizer: Muon (orthogonalized
momentum via Newton-Schulz zeropower) on the transformer trunk's 2-D weight
matrices, AdamW on everything else (encoders, both heads, norm gains, biases),
with weight decay excluded from norms/biases on the AdamW side. The packed
`nn.MultiheadAttention` in_proj_weight (3E, E) is orthogonalized as a batched
(3, E, E) zeropower — Q, K, V separately — while the parameter itself stays
packed, so state_dict keys never change. Heads and embeddings stay on AdamW
per the modded-nanotabpfn precedent (the 514-logit bar decoder is
scale-sensitive; Muon's spectral update normalization is the wrong bias
there).

Vendored-code attribution: the Newton-Schulz quintic iteration and the Muon
update rule follow Keller Jordan's Muon (https://github.com/KellerJordan/Muon,
MIT License; see NOTICE). Rewritten here for the single-GPU case with the
batched-QKV variant; distributed code paths removed.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["build_optimizer", "build_scheduler", "muon_param_groups", "zeropower_via_newtonschulz5"]


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def zeropower_via_newtonschulz5(grad: Tensor, steps: int = 5) -> Tensor:
    """Approximate UV^T of the SVD of `grad` via the quintic Newton-Schulz
    iteration, batched over leading dims (each trailing (m, n) matrix is
    orthogonalized independently). Runs in fp32 for device-portable numerics
    (our matrices are at most 1024x512 — the cost is negligible next to the
    backward pass)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.float()
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    if transposed:
        x = x.mT
    return x.to(grad.dtype)


class MuonAdamW(torch.optim.Optimizer):
    """Hybrid optimizer: per-group dispatch between Muon and AdamW math.

    Groups carry `use_muon`; Muon groups also carry `qkv` (packed (3E, E)
    parameters, orthogonalized as (3, E, E)). One optimizer object, so the
    checkpoint payload, the resume path and the single LambdaLR match AdamW's.
    """

    def __init__(self, param_groups: list[dict]) -> None:
        defaults = {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "use_muon": False,
            "qkv": False,
            "momentum": 0.95,
            "nesterov": True,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        }
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> None:  # noqa: ANN001 - torch signature
        assert closure is None, "closures are not supported"
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)

    def _muon_step(self, group: dict) -> None:
        lr, momentum, wd = group["lr"], group["momentum"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)
            buf: Tensor = state["momentum_buffer"]
            buf.lerp_(grad, 1.0 - momentum)
            update = grad.lerp(buf, momentum) if group["nesterov"] else buf
            if group["qkv"]:
                e = p.shape[-1]
                ortho = zeropower_via_newtonschulz5(update.view(3, e, e)).view_as(p)
                scale = 1.0  # per-chunk dims are (E, E)
            else:
                ortho = zeropower_via_newtonschulz5(update)
                scale = max(1.0, p.shape[-2] / p.shape[-1]) ** 0.5
            if wd:
                p.mul_(1.0 - lr * wd)
            p.add_(ortho, alpha=-lr * scale)

    def _adamw_step(self, group: dict) -> None:
        lr, wd = group["lr"], group["weight_decay"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            t = state["step"]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.lerp_(grad, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            if wd:
                p.mul_(1.0 - lr * wd)
            bias1 = 1.0 - beta1**t
            bias2 = 1.0 - beta2**t
            denom = (exp_avg_sq / bias2).sqrt_().add_(eps)
            p.addcdiv_(exp_avg, denom, value=-lr / bias1)


def muon_param_groups(model: nn.Module, train_cfg: dict) -> list[dict]:
    """Partition parameters for the hybrid optimizer.

    Muon <=> name starts with "transformer_blocks." AND ndim == 2; the packed
    in_proj_weight goes to its own qkv group. Everything else is AdamW, split
    into decay (ndim >= 2) and no-decay (norm gains, biases). The FIRST group
    is the AdamW decay group so `scheduler.get_last_lr()[0]` — the CSV `lr`
    column — means "the AdamW lr".
    """
    adamw_lr = float(train_cfg["lr"])
    muon_lr = float(train_cfg.get("muon_lr", 0.02))
    momentum = float(train_cfg.get("muon_momentum", 0.95))
    muon_wd = float(train_cfg.get("muon_wd", 0.0))
    adamw_decay: list[Tensor] = []
    adamw_nodecay: list[Tensor] = []
    muon: list[Tensor] = []
    muon_qkv: list[Tensor] = []
    for name, p in model.named_parameters():
        if name.startswith("transformer_blocks.") and p.ndim == 2:
            if name.endswith("in_proj_weight") and p.shape[0] == 3 * p.shape[1]:
                muon_qkv.append(p)
            else:
                muon.append(p)
        elif p.ndim >= 2:
            adamw_decay.append(p)
        else:
            adamw_nodecay.append(p)
    groups = [
        {"params": adamw_decay, "lr": adamw_lr, "weight_decay": 0.01, "use_muon": False},
        {"params": adamw_nodecay, "lr": adamw_lr, "weight_decay": 0.0, "use_muon": False},
        {"params": muon, "lr": muon_lr, "weight_decay": muon_wd, "use_muon": True,
         "momentum": momentum},
        {"params": muon_qkv, "lr": muon_lr, "weight_decay": muon_wd, "use_muon": True,
         "momentum": momentum, "qkv": True},
    ]
    return [g for g in groups if g["params"]]


def build_optimizer(model: nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    """`optimizer: adamw` (default) is plain AdamW with torch defaults;
    `optimizer: muon` builds the hybrid MuonAdamW."""
    kind = train_cfg.get("optimizer", "adamw")
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"])
    if kind == "muon":
        return MuonAdamW(muon_param_groups(model, train_cfg))
    raise ValueError(f"unknown train.optimizer {kind!r} (expected 'adamw' or 'muon')")


def build_scheduler(
    optimizer: torch.optim.Optimizer, train_cfg: dict, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup + cosine to 0 over `total_steps` OPTIMIZER steps. The
    single multiplicative lambda scales every param group's base lr, so the
    Muon groups track muon_lr * lambda and the AdamW groups lr * lambda."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_lambda(s, train_cfg["warmup_steps"], total_steps)
    )
