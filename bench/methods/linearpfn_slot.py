"""LinearPFN's slot in the benchmark, gated on a validated checkpoint.

This method raises until the
environment variable LINEARPFN_BENCH_CKPT points at a checkpoint that
passed the validation criteria (linearpfn.validate). Once set, one CPU
forward pass (torch pinned to a single thread, matching the timing
protocol) yields the PIPs;
coefficients come from the model's COEFFICIENT HEAD when the checkpoint
has one, else from predictive-surface probing exactly as in
linearpfn.validate. Set LINEARPFN_BENCH_COEF_SOURCE=probe to force the
probe path even on a coefficient-head checkpoint — that is what separates
"beta supervision improved the representations" from "the head is simply
a better read-out than probing", which the scores alone cannot tell
apart. Timing note: this row is the
single-core CPU figure; the GPU amortized-batch figure is produced
separately and labelled as such.
"""

from __future__ import annotations

import functools
import os
import time

import numpy as np

from bench.harness import MethodResult
from linearpfn.prior import Dataset, PriorConfig
from linearpfn.probe import extract_coefficients


@functools.lru_cache(maxsize=2)
def _load(ckpt_path: str, prior_cfg_path: str):
    import torch

    from linearpfn.train import build_model, load_config, make_prior_config

    torch.set_num_threads(1)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if prior_cfg_path:
        # LINEARPFN_BENCH_PRIOR_CFG: an explicit prior config for checkpoints
        # whose stored config lacks fields whose defaults differ from its training values;
        # rebuilding the prior from such a checkpoint would silently pair the
        # model with a different prior. Point it at the config it trained on.
        prior_cfg = make_prior_config(load_config(prior_cfg_path).get("data", {}))
    else:
        prior_cfg = make_prior_config(state["config"].get("data", {}))
    model, bar = build_model(state["config"]["model"], prior_cfg.include_interactions)
    model.load_state_dict(state["model"])
    model.eval()
    return model, bar, prior_cfg


def linearpfn_slot(
    ds: Dataset,
    cfg: PriorConfig,
    ckpt: str | None = None,
    prior_cfg_path: str | None = None,
) -> MethodResult:
    """None-valued paths fall back to the LINEARPFN_BENCH_* env gate; explicit
    paths (via `linearpfn_slot_for`) exist for paired multi-checkpoint runs —
    they must still point at validated checkpoints."""
    import torch

    if ckpt is None:
        ckpt = os.environ.get("LINEARPFN_BENCH_CKPT", "")
    if not ckpt:
        raise RuntimeError(
            "LinearPFN slot: no blessed checkpoint. Set LINEARPFN_BENCH_CKPT "
            "to a checkpoint that PASSED the hard-gates validation."
        )
    if prior_cfg_path is None:
        prior_cfg_path = os.environ.get("LINEARPFN_BENCH_PRIOR_CFG", "")
    model, bar, prior_cfg = _load(ckpt, prior_cfg_path)
    if prior_cfg.include_interactions != cfg.include_interactions:
        raise RuntimeError("checkpoint prior variant does not match the benchmark prior")
    X_t = torch.from_numpy(ds.X).float().unsqueeze(0)
    y_t = torch.from_numpy(ds.y).float().unsqueeze(0)
    n = ds.X.shape[0]
    t0 = time.perf_counter()
    with torch.no_grad():
        _, sel, coef_out = model.forward_with_heads(X_t, y_t, n)
        prob = torch.sigmoid(sel.float())[0].numpy()
        if coef_out is not None and os.environ.get(
            "LINEARPFN_BENCH_COEF_SOURCE", "head"
        ) != "probe":
            # A checkpoint WITH a coefficient head reports that head — it is
            # the model's coefficient output, and scoring it against the
            # exact anchors is the whole point of training it (the same
            # regret metric the probing rows are scored with). Slot 0 is the
            # intercept, which no head models and no scoring path reads;
            # the context mean of y fills it (X is standardized).
            eff = model.coefficient_posterior_mean(sel[0], coef_out[0]).numpy()
            coef = np.concatenate([[float(ds.y.mean())], eff])
            return MethodResult(
                gamma_hat=prob > 0.5, score=prob.copy(), prob=prob,
                coef=coef, fit_seconds=time.perf_counter() - t0,
                meta={"ckpt": ckpt, "coef_source": f"head:{model.coef_head_mode}"},
            )

    def mean_fn(probes: np.ndarray) -> np.ndarray:
        X_all = torch.cat([X_t[0], torch.from_numpy(probes).float()]).unsqueeze(0)
        with torch.no_grad():
            logits = model(X_all, y_t, n)
        return bar.mean(logits.float())[0].numpy()

    res = extract_coefficients(
        np.random.default_rng(0), ds.X, mean_fn, cfg=cfg
    )
    fit_seconds = time.perf_counter() - t0
    return MethodResult(
        gamma_hat=prob > 0.5, score=prob.copy(), prob=prob,
        coef=res.coef, fit_seconds=fit_seconds,
        meta={"ckpt": ckpt, "surrogate_r2": res.r2, "coef_source": "probe"},
    )


def linearpfn_slot_for(ckpt: str, prior_cfg_path: str):
    """A named slot pinned to one (checkpoint, prior config) pair, for runs
    that compare checkpoints side by side on shared data."""
    import functools as _ft

    return _ft.partial(linearpfn_slot, ckpt=ckpt, prior_cfg_path=prior_cfg_path)
