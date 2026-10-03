"""Quick check of the method on a CPU (about a minute).

1. one dataset drawn from the prior of the reported model (configs/paper_model.yaml);
2. one forward pass of a randomly initialized network with the reported
   architecture: predictive logits, per-effect PIPs and posterior-mean coefficients;
3. the exact posterior of the same dataset (enumeration over models, quadrature
   over the slab scale).

The network is untrained, so its PIPs are arbitrary; the check is that every
stage runs end to end and returns outputs of the right shape and range.

    python smoke_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from linearpfn.prior import interaction_pairs, sample_dataset  # noqa: E402
from linearpfn.reference import fit_exact  # noqa: E402
from linearpfn.train import build_model, load_config, make_prior_config  # noqa: E402

CONFIG = ROOT / "configs" / "paper_model.yaml"


def effect_names(p: int, cfg) -> list[str]:
    return [f"x{j + 1}" for j in range(p)] + [f"x{j + 1}:x{k + 1}"
                                               for j, k in interaction_pairs(p, cfg)]


def main() -> int:
    config = load_config(str(CONFIG))
    cfg = make_prior_config(config["data"])
    rng = np.random.default_rng(0)

    t0 = time.perf_counter()
    ds = sample_dataset(rng, cfg, n=200, p=3)
    n, p = ds.X.shape
    names = effect_names(p, cfg)
    active = [names[j] for j in np.flatnonzero(ds.gamma[1:])]
    print(f"1. prior draw: n = {n}, p = {p}, {len(names)} candidate effects, "
          f"active: {active or 'none'} ({time.perf_counter() - t0:.2f} s)")

    torch.manual_seed(0)
    model, _ = build_model(config["model"], include_interactions=cfg.include_interactions)
    model.eval()
    n_ctx = 150
    X_t = torch.from_numpy(ds.X).float().unsqueeze(0)
    y_t = torch.from_numpy(ds.y[:n_ctx]).float().unsqueeze(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, sel, coef_out = model.forward_with_heads(X_t, y_t, n_ctx)
        pips = torch.sigmoid(sel.float())[0].numpy()
        coef = (model.coefficient_posterior_mean(sel[0], coef_out[0]).numpy()
                if coef_out is not None else None)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"2. forward pass ({n_params:,} parameters, random weights): predictive logits "
          f"{tuple(logits.shape)} for {n - n_ctx} query rows, {pips.size} PIPs, "
          f"{'no' if coef is None else coef.size} coefficients ({time.perf_counter() - t0:.2f} s)")
    assert pips.shape == (len(names),) and np.all((pips >= 0) & (pips <= 1))
    assert coef is None or coef.shape == (len(names),)

    t0 = time.perf_counter()
    post = fit_exact(ds.X, ds.y, cfg)
    pip = post.pip()[1:]
    mean = post.coef_mean()[1:]
    print(f"3. exact posterior ({time.perf_counter() - t0:.2f} s):")
    for name, g, q, b, m in zip(names, ds.gamma[1:], pip, ds.beta[1:], mean, strict=True):
        print(f"   {name:8s} active={bool(g)!s:5s}  PIP {q:.3f}  "
              f"true beta {b:+.3f}  posterior mean {m:+.3f}")
    assert np.all((pip >= -1e-12) & (pip <= 1 + 1e-12))
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
