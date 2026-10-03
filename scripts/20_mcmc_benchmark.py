"""Stage 20 (compute) — the MCMC sampler on a random subset of the main benchmark.

For each panel in `--panels` (default `mx`, the main benchmark), `--per-family`
truth files are drawn uniformly at random, with a seed per (--seed, panel).
The draw is deliberately not stratified by p: the panel's own (n, p)
distribution is what a user of the benchmark faces, so the aggregate cost and
agreement over the subset are directly interpretable, and a per-p breakdown is
read off the same rows.

Each dataset is fitted by `linearpfn.mc3_general` under the prior of
`--config`, the prior that generated y, so neither the sampler nor the model is
misspecified. Chains run for the step ladder below (steps per chain by p); a
run counts as converged when the largest split R-hat is below RHAT_MAX and the
smallest effective sample size is above ESS_MIN. The model's own fit seconds
for the same datasets are already in its rows, measured under the same
single-thread setting (BENCH_PINNED=1) as this stage.

One npz per dataset under <dir>/mc3/, in the benchmark's row layout plus the
convergence diagnostics; datasets are split by `--shard i/N` and a rerun skips
finished files.

Run:  python scripts/20_mcmc_benchmark.py --dir results/main_benchmark \\
          --config configs/paper_model.yaml --per-family 150 --shard 0/16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT))

# MCMC steps per chain by p (the same ladder as the real-outcome stage 18)
STEP_LADDER = ((5, 100_000), (10, 200_000), (20, 500_000), (30, 1_000_000))
RHAT_MAX, ESS_MIN = 1.1, 200.0


def steps_for(p: int) -> int:
    for bound, steps in STEP_LADDER:
        if p <= bound:
            return steps
    return STEP_LADDER[-1][1]


def subset(truth: Path, per_family: int, seed: int, panels: tuple[str, ...]) -> list[Path]:
    out = []
    for panel in panels:
        files = sorted(truth.glob(f"ip{panel}_*.npz"))
        rng = np.random.default_rng(
            [seed, ord(panel[0]), int(panel[1] if panel[1].isdigit() else 0)])
        idx = rng.permutation(len(files))[:per_family]
        out += [files[i] for i in sorted(idx)]
    return out


def main() -> None:
    from linearpfn.mc3_general import fit_mc3_general
    from linearpfn.train import load_config, make_prior_config

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=str(ROOT / "results" / "main_benchmark"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "paper_model.yaml"))
    ap.add_argument("--per-family", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--panels", nargs="+", default=["mx"])
    args = ap.parse_args()

    root = Path(args.dir)
    cfg = make_prior_config(load_config(args.config)["data"])
    files = subset(root / "truth", args.per_family, args.seed, tuple(args.panels))
    i, nsh = (int(v) for v in args.shard.split("/"))
    mine = [f for k, f in enumerate(files) if k % nsh == i]

    if args.dry_run:
        ps = []
        for f in files:
            ps.append(int(np.load(f, allow_pickle=True)["p"]))
        ps = np.array(ps)
        print(f"subset {len(files)} datasets; shard {i}/{nsh} -> {len(mine)}")
        print(f"p: min {ps.min()} med {int(np.median(ps))} max {ps.max()}; "
              f"share p>=26: {100 * (ps >= 26).mean():.1f} %")
        return

    out = root / "mc3"
    out.mkdir(parents=True, exist_ok=True)
    print(f"MC3 cost run: {len(mine)} of {len(files)} datasets, shard {i}/{nsh}", flush=True)
    for f in mine:
        dst = out / f.name
        if dst.exists():
            continue
        z = np.load(f, allow_pickle=True)
        X = z["X"].astype(float)
        y = z["y"].astype(float)
        p = int(z["p"])
        st = steps_for(p)
        t0 = time.perf_counter()
        r = fit_mc3_general(X, y, cfg, n_chains=args.chains, n_steps=st,
                            seed=int(abs(hash(f.name)) % (2 ** 31)))
        wall = time.perf_counter() - t0
        dg = r.diagnostics
        pip = r.pip[1:]
        np.savez_compressed(
            dst, gamma_hat=pip > 0.5, score=pip.copy(), prob=pip, coef=r.coef_mean,
            fit_seconds=r.seconds, wall_seconds=wall, n=z["n"], p=p,
            gamma_true=z["gamma_true"], beta_true=z["beta_true"],
            rhat_max=dg.rhat_max, ess_min=dg.ess_min, n_steps=dg.n_steps,
            converged=bool(dg.rhat_max < RHAT_MAX and dg.ess_min > ESS_MIN),
            meta=str(z["meta"]))
        print(f"{f.name}: p={p} n={int(z['n'])} steps={st} {wall:.1f}s "
              f"rhat {dg.rhat_max:.3f} ess {dg.ess_min:.0f}", flush=True)
    print(f"MC3-COST-DONE shard {i}", flush=True)


if __name__ == "__main__":
    main()
