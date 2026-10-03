"""Private helpers for the training loop (checkpointing, logging, stream math).

Private helper of `linearpfn.train`, which re-exports what other modules use.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

__all__ = [
    "CSV_COLUMNS",
    "IntervalMeter",
    "check_resume_compat",
    "save_checkpoint",
    "snapshot_due",
    "stream_extent",
]

# One row per optimizer step interval. Loss terms are means over the interval's
# optimizer steps (each itself a mean over that step's accumulation group).
# Resuming appends to the existing file, so start a fresh checkpoint_dir per run.
CSV_COLUMNS = [
    "step",
    "loss",
    "nll",
    "bce",
    "bce_mains",
    "bce_int",
    "coef",
    "lr",
    "grad_norm",
    "steps_per_s",
    "data_wait_frac",
    "seconds",
]


def stream_extent(total_steps: int, start_step: int, accum_steps: int) -> tuple[int, int]:
    """(num_steps, start_index) for PriorBatches when `steps` counts OPTIMIZER
    steps and each optimizer step consumes `accum_steps` batch indices.

    With accum_steps=1 this is (total - start, start). Resuming replays
    nothing: the stream continues at batch index start_step * accum_steps.
    """
    return (total_steps - start_step) * accum_steps, start_step * accum_steps


def snapshot_due(
    step_done: int,
    total_steps: int,
    every: int,
    final_every: int,
    final_window: int,
) -> bool:
    """Whether a retained snapshot is due after completing optimizer step
    `step_done` (1-based). `every` is the base cadence; `final_every` applies
    within the last `final_window` steps (denser, for LAWA). Zero disables."""
    if every and step_done % every == 0:
        return True
    in_window = final_window and step_done > total_steps - final_window
    return bool(final_every and in_window and step_done % final_every == 0)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    config: dict,
) -> None:
    """Atomic checkpoint write (tmp + replace); `config` is the permanent
    record of the run (prior + regime), consumed by validate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": config,
        },
        tmp,
    )
    tmp.replace(path)


def check_resume_compat(ckpt_config: dict, train_cfg: dict) -> None:
    """Refuse resumes that would silently change the regime mid-run.

    A changed accum_steps remaps optimizer steps to dataset indices (skipping
    or replaying data); a changed optimizer cannot load the saved state. Both
    fail loudly here with a readable message instead of a cryptic
    load_state_dict/group error.
    """
    old_train = ckpt_config.get("train", {})
    for key, default in (("accum_steps", 1), ("optimizer", "adamw")):
        old = old_train.get(key, default)
        new = train_cfg.get(key, default)
        if old != new:
            raise ValueError(
                f"resume mismatch on train.{key}: checkpoint has {old!r}, "
                f"config has {new!r} — resuming across a regime change is not "
                "supported (start a fresh checkpoint_dir instead)"
            )


class IntervalMeter:
    """Per-log-interval optimizer-steps/s and data-wait fraction.

    The CSV row needs the deltas since the previous logged row, not run totals.
    """

    def __init__(self) -> None:
        self._t = time.monotonic()
        self._wait = 0.0
        self._steps = 0

    def interval(self, data_wait_total: float, steps_done: int) -> tuple[float, float]:
        now = time.monotonic()
        dt = max(now - self._t, 1e-9)
        steps_per_s = (steps_done - self._steps) / dt
        wait_frac = (data_wait_total - self._wait) / dt
        self._t, self._wait, self._steps = now, data_wait_total, steps_done
        return steps_per_s, wait_frac
