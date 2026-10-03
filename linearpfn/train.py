"""Training loop for LinearPFN: amortized spike-and-slab regression.

On-the-fly data generation in dataloader workers (`linearpfn.generator`),
bar-distribution NLL on query rows only, AdamW or Muon with linear warmup +
cosine decay, bf16 autocast where available, periodic checkpointing,
resumable. Logging: CSV always; tensorboard if the package is importable.

The loss is assembled in `compute_loss` as a dict of named terms summed into
"total": predictive NLL, selection BCE and the coefficient term.

Run: python -m linearpfn.train --config configs/smoke.yaml [--resume CKPT]

After training, a predictive check is evaluated and printed: mean query NLL
of the model vs the per-dataset Gaussian fit to context y (the marginal
baseline) on freshly sampled small-p datasets.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from linearpfn import _ddp
from linearpfn._evalgate import _binary_auc as _binary_auc  # re-export (validate/bench)
from linearpfn._evalgate import evaluate_marginal_gate
from linearpfn._optim import build_optimizer, build_scheduler
from linearpfn._trainutil import (
    CSV_COLUMNS,
    IntervalMeter,
    check_resume_compat,
    save_checkpoint,
    snapshot_due,
    stream_extent,
)
from linearpfn.generator import Batch, PriorBatches
from linearpfn.model import BarDistribution, LinearPFNModel
from linearpfn.prior import PriorConfig, canonical_marginal_weights

__all__ = ["compute_loss", "evaluate_marginal_gate", "load_config", "train"]

_save_checkpoint = save_checkpoint  # private alias


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def make_prior_config(data_cfg: dict) -> PriorConfig:
    """PriorConfig from the config's data section.

    Driven by the dataclass fields, so every prior field is settable from
    YAML; unknown keys raise instead of silently no-opping (a dropped key
    would train under a different prior than the config claims). YAML lists
    become tuples (heredity_weights) and the marginal_weights mapping is
    coerced to the canonical tuple-of-pairs form.
    """
    field_names = {f.name for f in fields(PriorConfig)}
    unknown = set(data_cfg) - field_names - {"batch_size", "num_workers"}
    if unknown:
        raise ValueError(f"unknown data config keys: {sorted(unknown)}")
    overrides: dict[str, object] = {}
    for key in field_names & set(data_cfg):
        value = data_cfg[key]
        if key == "marginal_weights" and isinstance(value, dict):
            value = canonical_marginal_weights(value)
        elif isinstance(value, list):
            value = tuple(value)
        overrides[key] = value
    return PriorConfig(**overrides)


def build_model(
    model_cfg: dict, include_interactions: bool = True
) -> tuple[LinearPFNModel, BarDistribution]:
    """Model + bar distribution. `include_interactions` comes from the PRIOR
    (data section of the config), never from the model section — the
    selection head's candidate set is defined by the data-generating process,
    so deriving it from one place makes a head/prior mismatch impossible
    (and a stale checkpoint fails loudly in load_state_dict)."""
    bar = BarDistribution()
    model = LinearPFNModel(
        embedding_size=model_cfg["embedding_size"],
        num_attention_heads=model_cfg["num_attention_heads"],
        mlp_hidden_size=model_cfg["mlp_hidden_size"],
        num_layers=model_cfg["num_layers"],
        num_outputs=bar.n_bins,
        pair_pooling=model_cfg.get("pair_pooling", "y_aware"),
        include_interactions=include_interactions,
        attention_impl=model_cfg.get("attention_impl", "mha"),
        norm_placement=model_cfg.get("norm_placement", "post"),
        coef_head=model_cfg.get("coef_head", "none"),
        pair_stats=model_cfg.get("pair_stats", False),
    )
    return model, bar


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_loss(
    model: LinearPFNModel,
    bar: BarDistribution,
    batch: Batch,
    device: torch.device,
    autocast_device: str = "cpu",
    use_bf16: bool = False,
    lambda_sel: float = 1.0,
    sel_loss_balance: str = "none",
    lambda_sel_int: float = 1.0,
    lambda_coef: float = 0.0,
    lambda_nll: float = 1.0,
) -> dict[str, Tensor]:
    """Named loss terms; "total" is optimized. "bce_mains"/"bce_int" are
    per-slot diagnostic means (no_grad — never part of the training graph;
    "bce_int" is NaN under the mains-only variant).

    total = lambda_nll * predictive NLL (query rows only) + lambda_sel * selection BCE
    (lambda_nll = 0 drops the predictive signal from the trunk, the ablation
    without the predictive loss; the NLL is still computed and logged)
    against the simulated ground-truth gamma. sel_loss_balance="none" (the
    default, used by the reported configuration) is the pooled per-slot
    mean; "subset_mean" uses
    0.5 * mean(mains) + 0.5 * lambda_sel_int * mean(interactions) —
    deliberately NOT renormalized, so a lambda_sel_int sweep leaves the mains
    gradient invariant. Under mains-only models "subset_mean" reduces exactly
    to the pooled mean. Gradients of the predictive head are NOT stopped —
    the shared representations help both heads. Autocast covers the
    transformer forward pass only: log-softmax/NLL/BCE are always computed in
    fp32 (bf16 log-softmax over 514 logits introduces a silent loss floor of
    about 0.007 nats at initialization under the CPU autocast policy, which
    unlike CUDA's does not force log_softmax to fp32).
    """
    if sel_loss_balance not in ("none", "subset_mean"):
        raise ValueError(f"unknown sel_loss_balance {sel_loss_balance!r}")
    if sel_loss_balance == "none" and lambda_sel_int != 1.0:
        raise ValueError(
            "lambda_sel_int has no meaning under sel_loss_balance='none' — "
            "set sel_loss_balance='subset_mean' or drop lambda_sel_int"
        )
    split = batch.split_index
    p = batch.X.shape[2]
    X = batch.X.to(device)
    y = batch.y.to(device)
    with torch.autocast(autocast_device, dtype=torch.bfloat16, enabled=use_bf16):
        pred_logits, sel_logits, coef_out = _ddp.heads_of(model)(X, y[:, :split], split)
    with torch.autocast(autocast_device, enabled=False):
        nll = bar.nll(pred_logits.float(), y[:, split:]).mean()
        target = batch.gamma[:, 1:].to(device=device, dtype=torch.float32)
        if sel_loss_balance == "none":
            bce = F.binary_cross_entropy_with_logits(sel_logits.float(), target)
        else:
            per = F.binary_cross_entropy_with_logits(
                sel_logits.float(), target, reduction="none"
            )
            if per.shape[1] > p:
                bce = 0.5 * per[:, :p].mean() + 0.5 * lambda_sel_int * per[:, p:].mean()
            else:  # mains-only: the subset mean IS the pooled mean
                bce = per.mean()
        total = lambda_nll * nll + lambda_sel * bce
        coef_loss = torch.zeros((), device=total.device)
        if coef_out is not None and lambda_coef:
            beta = batch.beta[:, 1:].to(device=device, dtype=torch.float32)
            raw = _ddp.unwrap(model)
            if raw.coef_head_mode == "mse":
                # squared error is minimized by the conditional mean, so this
                # converges to E[beta | data], the posterior mean. Zeros are
                # kept: they carry the spike's mass.
                coef_loss = ((coef_out.float() - beta) ** 2).mean()
            else:  # spike_slab: value term on ACTIVE slots; the spike is the BCE
                active = target > 0.5
                if bool(active.any()):
                    assert raw.coef_bar is not None
                    coef_loss = raw.coef_bar.nll(
                        coef_out.float()[active], beta[active]
                    ).mean()
            total = total + lambda_coef * coef_loss
        with torch.no_grad():
            per_diag = F.binary_cross_entropy_with_logits(
                sel_logits.float(), target, reduction="none"
            )
            bce_mains = per_diag[:, :p].mean()
            if per_diag.shape[1] > p:
                bce_int = per_diag[:, p:].mean()
            else:
                bce_int = torch.full((), float("nan"), device=per_diag.device)
    return {"nll": nll, "bce": bce, "bce_mains": bce_mains, "bce_int": bce_int,
            "coef": coef_loss, "total": total}


# Strict train-section schema: a typo'd ablation knob (e.g. "acum_steps")
# must fail loudly, not silently train the baseline regime.
_KNOWN_TRAIN_KEYS = {
    "steps", "lr", "warmup_steps", "grad_clip", "lambda_sel", "bf16",
    "checkpoint_dir", "checkpoint_every_minutes", "log_every",
    "accum_steps", "optimizer", "muon_lr", "muon_momentum", "muon_wd",
    "tf32", "compile", "sel_loss_balance", "lambda_sel_int",
    "snapshot_every_steps", "snapshot_final_every", "snapshot_final_window",
    "lambda_coef", "lambda_nll",
    "post_eval_gate",
}

_LOSS_KEYS = ("total", "nll", "bce", "bce_mains", "bce_int", "coef")


def train(config: dict, resume: str | None = None) -> dict:
    """Run the training loop; returns model, bar, losses, and start step.

    `train.steps` counts OPTIMIZER steps. With `train.accum_steps` = A > 1,
    each optimizer step accumulates gradients over A consecutive stream
    batches (effective batch A * data.batch_size, loss averaged over the
    group); the deterministic stream then consumes steps * A batch indices.
    """
    unknown = set(config["train"]) - _KNOWN_TRAIN_KEYS
    if unknown:
        raise ValueError(f"unknown train config keys: {sorted(unknown)}")
    torch.manual_seed(config["seed"])
    # Data-parallel: the process group owns this rank's device.
    # Off-distributed this is exactly resolve_device().
    device = (_ddp.setup() if _ddp.is_distributed()
              else resolve_device(config.get("device", "auto")))
    is_main_rank = _ddp.is_main()
    # Always log the resolved device: "auto" falls back to CPU when CUDA
    # init fails, and this line makes a silent fallback visible in any log.
    if is_main_rank:
        print(f"device: {device}", flush=True)
    train_cfg = config["train"]
    data_cfg = config["data"]
    if train_cfg.get("tf32", False):
        torch.set_float32_matmul_precision("high")
    prior_cfg = make_prior_config(data_cfg)
    model, bar = build_model(config["model"], prior_cfg.include_interactions)
    model.to(device)
    bar.to(device)
    if train_cfg.get("compile", False):
        # Per-block only: forward_with_selection contains p-dependent Python
        # (interaction_pairs builds a list per p), which would force one
        # graph per p value; the blocks are pure tensor code. dynamic=True
        # because (n, p) and split_index vary every step.
        for block in model.transformer_blocks:
            block.forward = torch.compile(block.forward, dynamic=True)
    total_steps = train_cfg["steps"]
    accum = int(train_cfg.get("accum_steps", 1))
    if accum < 1:
        raise ValueError(f"train.accum_steps must be >= 1, got {accum}")
    # Data-parallel: the block of `accum` stream indices one optimizer step
    # consumes is split across ranks, so each rank runs `accum // W` of them
    # and divides by that local count; the process group's gradient AVERAGE
    # then reassembles exactly the single-process mean over the whole block.
    raw_model = model
    local_accum = accum
    if _ddp.is_distributed():
        w = _ddp.world_size()
        if accum % w:
            raise ValueError(
                f"train.accum_steps {accum} must be divisible by WORLD_SIZE {w} "
                "so the effective batch is unchanged")
        local_accum = accum // w
        model = torch.nn.parallel.DistributedDataParallel(
            _ddp.HeadsAdapter(model), device_ids=[_ddp.local_rank()],
            output_device=_ddp.local_rank())
    # ALWAYS the unwrapped model: muon_param_groups() splits on the parameter
    # NAME ("transformer_blocks." -> Muon), and the DDP wrap prefixes every name
    # with "module.model.", which would silently move the whole trunk to AdamW.
    optimizer = build_optimizer(raw_model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, total_steps)
    start_step = 0
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        check_resume_compat(state["config"], train_cfg)
        raw_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"]
        if is_main_rank:
            print(f"resumed from {resume} at step {start_step}")

    use_bf16 = train_cfg.get("bf16", "auto")
    if use_bf16 == "auto":
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast_device = device.type if device.type in ("cuda", "cpu") else "cpu"

    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ckpt_dir / "train_log.csv"
    new_log = not csv_path.exists()
    writer = None
    try:  # tensorboard is optional (core logging is CSV); rank 0 only
        from torch.utils.tensorboard import SummaryWriter

        if _ddp.is_main():
            writer = SummaryWriter(log_dir=str(ckpt_dir / "tb"))
    except ImportError:
        pass

    num_stream_steps, stream_start = stream_extent(total_steps, start_step, accum)
    dataset = PriorBatches(
        prior_cfg,
        batch_size=data_cfg["batch_size"],
        seed=config["seed"],
        num_steps=num_stream_steps,
        start_index=stream_start,
        rank=_ddp.rank(),
        world_size=_ddp.world_size(),
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=data_cfg.get("num_workers", 0))

    losses: list[float] = []
    bce_losses: list[float] = []
    lambda_sel = float(train_cfg.get("lambda_sel", 1.0))
    sel_loss_balance = train_cfg.get("sel_loss_balance", "none")
    lambda_sel_int = float(train_cfg.get("lambda_sel_int", 1.0))
    lambda_coef = float(train_cfg.get("lambda_coef", 0.0))
    lambda_nll = float(train_cfg.get("lambda_nll", 1.0))
    grad_clip = train_cfg["grad_clip"]
    snap_every = int(train_cfg.get("snapshot_every_steps", 0) or 0)
    snap_final_every = int(train_cfg.get("snapshot_final_every", 0) or 0)
    snap_final_window = int(train_cfg.get("snapshot_final_window", 0) or 0)
    log_every = train_cfg.get("log_every", 50)
    ckpt_interval = train_cfg.get("checkpoint_every_minutes", 30) * 60.0
    last_ckpt = time.monotonic()
    t0 = time.monotonic()
    meter = IntervalMeter()
    data_wait = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    iterator = iter(loader)
    offset = 0
    with open(csv_path if is_main_rank else os.devnull, "a", newline="") as csv_file:
        log = csv.writer(csv_file)
        if new_log and is_main_rank:
            log.writerow(CSV_COLUMNS)
        while True:
            group: list[Tensor] = []
            ended = False
            for micro in range(local_accum):
                fetch_start = time.monotonic()
                try:
                    batch = next(iterator)
                except StopIteration:
                    if micro:  # stream length is exactly steps * accum
                        raise RuntimeError(
                            "data stream ended mid-accumulation group"
                        ) from None
                    ended = True
                    break
                data_wait += time.monotonic() - fetch_start
                with _ddp.no_sync(model, micro == local_accum - 1):
                    terms = compute_loss(model, bar, batch, device, autocast_device,
                                         bool(use_bf16), lambda_sel, sel_loss_balance,
                                         lambda_sel_int, lambda_coef, lambda_nll)
                    # accum == 1 backpropagates the loss itself (no division)
                    (terms["total"] if local_accum == 1
                     else terms["total"] / local_accum).backward()
                group.append(torch.stack([terms[k].detach() for k in _LOSS_KEYS]))
            if ended:
                break
            step = start_step + offset
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip if grad_clip is not None else float("inf")
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            vals = dict(zip(_LOSS_KEYS,
                            _ddp.reduce_mean(torch.stack(group).mean(dim=0)).tolist(),
                            strict=True))
            losses.append(vals["total"])
            bce_losses.append(vals["bce"])
            if is_main_rank and (step % log_every == 0 or step == total_steps - 1):
                elapsed = time.monotonic() - t0
                lr_now = scheduler.get_last_lr()[0]
                grad_norm_val = float(grad_norm)
                steps_per_s, wait_frac = meter.interval(data_wait, offset + 1)
                print(f"step {step:6d} | loss {vals['total']:8.4f} | nll {vals['nll']:7.4f} | "
                      f"bce {vals['bce']:7.4f} (m {vals['bce_mains']:.4f} i "
                      f"{vals['bce_int']:.4f}) | lr {lr_now:.2e} | gnorm {grad_norm_val:7.3f} "
                      f"| {elapsed:6.1f}s")
                log.writerow([step, f"{vals['total']:.6f}", f"{vals['nll']:.6f}",
                              f"{vals['bce']:.6f}", f"{vals['bce_mains']:.6f}",
                              f"{vals['bce_int']:.6f}", f"{vals['coef']:.6f}",
                              f"{lr_now:.6e}",
                              f"{grad_norm_val:.4f}", f"{steps_per_s:.3f}",
                              f"{wait_frac:.4f}", f"{elapsed:.1f}"])
                csv_file.flush()
                if writer is not None:
                    writer.add_scalar("loss/total", vals["total"], step)
                    for name in _LOSS_KEYS[1:]:
                        writer.add_scalar(f"loss/{name}", vals[name], step)
                    writer.add_scalar("train/grad_norm", grad_norm_val, step)
            if is_main_rank and time.monotonic() - last_ckpt > ckpt_interval:
                save_checkpoint(ckpt_dir / "ckpt_latest.pt", raw_model, optimizer, scheduler,
                                step + 1, config)
                last_ckpt = time.monotonic()
            if is_main_rank and snapshot_due(step + 1, total_steps, snap_every, snap_final_every,
                                     snap_final_window):
                save_checkpoint(ckpt_dir / f"ckpt_step_{step + 1:07d}.pt", raw_model,
                                optimizer, scheduler, step + 1, config)
            offset += 1
    if is_main_rank:
        save_checkpoint(ckpt_dir / "ckpt_final.pt", raw_model, optimizer, scheduler,
                        total_steps, config)
    if writer is not None:
        writer.close()
    loop_s = time.monotonic() - t0
    peak_gpu_gb = None
    if offset and is_main_rank:
        print(f"throughput: {offset / loop_s:.2f} steps/s | data-loader wait "
              f"{data_wait:.1f}s of {loop_s:.1f}s loop ({100 * data_wait / loop_s:.1f}%)")
    if device.type == "cuda":
        peak_gpu_gb = torch.cuda.max_memory_allocated() / 2**30
        total_gb = torch.cuda.get_device_properties(device).total_memory / 2**30
        if is_main_rank:
            print(f"peak GPU memory: {peak_gpu_gb:.2f} GiB of {total_gb:.1f} GiB")
    _ddp.cleanup()
    return {"model": raw_model, "bar": bar, "losses": losses, "bce_losses": bce_losses,
            "start_step": start_step, "steps_per_s": (offset / loop_s if offset else 0.0),
            "data_wait_fraction": (data_wait / loop_s if loop_s > 0 else 0.0),
            "peak_gpu_gb": peak_gpu_gb}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train LinearPFN")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    result = train(config, resume=args.resume)
    if not _ddp.is_main():
        # The summary and the post-training gate belong to one process: every
        # rank returns the same averaged losses, and the gate's quadrature fits
        # would otherwise run W times side by side on W resident models.
        return
    losses = result["losses"]
    if losses:
        k = max(1, len(losses) // 10)
        print(f"loss: first-{k}-mean {sum(losses[:k]) / k:.4f} | "
              f"last-{k}-mean {sum(losses[-k:]) / k:.4f}")
    if not config["train"].get("post_eval_gate", True):
        # The check's quadrature fits need host memory next to the resident
        # model; large runs skip it and validate in a separate job instead.
        print("post-training eval gate skipped (train.post_eval_gate: false)")
        return
    device = resolve_device(config.get("device", "auto"))
    gate = evaluate_marginal_gate(result["model"], result["bar"], config, device)
    verdict = "PASS" if gate["model_nll"] < gate["marginal_nll"] else "FAIL"
    print(
        f"predictive check (p <= {config['eval']['p_max']}, "
        f"{config['eval']['n_datasets']} datasets): "
        f"model query NLL {gate['model_nll']:.4f} | marginal baseline "
        f"{gate['marginal_nll']:.4f} | exact BMA {gate['exact_nll']:.4f} | "
        f"gap closed {100 * gate['gap_closed']:.1f}% -> {verdict}"
    )
    bces = result["bce_losses"]
    if bces:
        k = max(1, len(bces) // 10)
        first_bce = sum(bces[:k]) / k
        last_bce = sum(bces[-k:]) / k
        auc = gate["selection_auc"]
        ok = last_bce < first_bce and auc > 0.6
        print(
            f"selection check: BCE first-{k}-mean {first_bce:.4f} -> last-{k}-mean "
            f"{last_bce:.4f} | selection AUC {auc:.4f} (> 0.6) -> "
            f"{'PASS' if ok else 'FAIL'}"
        )


if __name__ == "__main__":
    main()
