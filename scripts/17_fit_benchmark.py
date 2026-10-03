"""Stage 17 (compute) — fit methods on the saved truth rows of a benchmark panel.

Every registered method of `bench.methods.available_methods` (lasso, stability
selection, the exact posterior, the R baselines when R and their packages are
installed) and, optionally, a LinearPFN checkpoint as an extra slot
(`--slot-name/--slot-ckpt/--slot-cfg`) is run on the truth files that stages 15
and 16 wrote. The fitting loop is the benchmark's own (`bench.shared._fit`):
the same method wrappers, the same single-thread timing, the same npz row
layout, skip-if-exists and per-dataset failure isolation. `_fit` takes
`(key, make, meta)` entries whose `make()` returns a Dataset; here `make()`
loads one saved truth file.

The datasets are split into `--shard i/N` by a stable (sorted) order, so
shards can run in parallel and a resubmitted shard resumes. `--p-cap 5`
restricts a method to the designs the exact reference can enumerate.

Rows land in `<out-dir>/<method>/<key>.npz`, where the number stages read them.

Run:  python scripts/17_fit_benchmark.py --config configs/paper_model.yaml \\
          --panels mx --truth-dir results/main_benchmark/truth \\
          --out-dir results/main_benchmark --methods susie --shard 0/16
      python scripts/17_fit_benchmark.py --config configs/paper_model.yaml \\
          --panels mx --truth-dir results/main_benchmark/truth \\
          --out-dir results/main_benchmark --slot-name linearpfn_strong \\
          --slot-ckpt checkpoints/paper_model/ckpt_final.pt \\
          --slot-cfg configs/paper_model.yaml --methods linearpfn_strong --shard 0/16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT))


def entries_from(truth_dir: Path, panels: tuple[str, ...], p_cap: int = 10**9):
    """(key, make, meta) triples over the saved truth files, in a STABLE order.

    Sorted by filename so the shard split is reproducible across runs and machines.
    """
    from linearpfn.prior import Dataset

    out = []
    for f in sorted(truth_dir.glob("*.npz")):
        if not any(f.name.startswith(f"ip{p}_") for p in panels):
            continue
        meta = json.loads(str(np.load(f, allow_pickle=False)["meta"]))
        if int(meta["p"]) > p_cap:      # the exact reference enumerates to p <= 5
            continue

        def make(path=f):
            z = np.load(path, allow_pickle=False)
            gamma = np.concatenate([[True], z["gamma_true"].astype(bool)])
            return Dataset(X=z["X"], y=z["y"], gamma=gamma,
                           beta=z["beta_true"], sigma2=1.0), {}

        out.append((f.stem, make, meta))
    return out


def main() -> None:
    from bench.methods import available_methods
    from bench.methods.linearpfn_slot import linearpfn_slot_for
    from bench.shared import PRIOR_CONFIG, _fit
    from linearpfn.train import load_config, make_prior_config

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--truth-dir", default=str(ROOT / "results" / "benchmark" / "truth"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "benchmark"))
    ap.add_argument("--config", default=str(PRIOR_CONFIG),
                    help="prior the METHODS assume; must be the one that generated y")
    ap.add_argument("--panels", nargs="+", default=["1g", "2g"])
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--slot-name", default=None, help="row name for the checkpoint slot")
    ap.add_argument("--slot-ckpt", default=None)
    ap.add_argument("--slot-cfg", default=None)
    ap.add_argument("--shard", default="0/1", help="i/N")
    ap.add_argument("--p-cap", type=int, default=10**9,
                    help="skip designs wider than this (use 5 for the exact row)")
    args = ap.parse_args()

    cfg = make_prior_config(load_config(args.config)["data"])
    entries = entries_from(Path(args.truth_dir), tuple(args.panels), args.p_cap)
    if not entries:
        raise SystemExit(f"no truth files under {args.truth_dir} for panels {args.panels}")

    available, notes = available_methods(include_slot=False)
    for note in notes:
        print(note, flush=True)
    if args.slot_name:
        if not (args.slot_ckpt and args.slot_cfg):
            raise SystemExit("--slot-name needs --slot-ckpt and --slot-cfg")
        available[args.slot_name] = linearpfn_slot_for(args.slot_ckpt, args.slot_cfg)
    missing = [m for m in args.methods if m not in available]
    if missing:
        raise SystemExit(f"methods unavailable: {missing}")

    i, n = (int(v) for v in args.shard.split("/"))
    print(f"panel: {len(entries)} datasets x {len(args.methods)} methods, "
          f"shard {i}/{n}", flush=True)
    _fit(entries, {m: available[m] for m in args.methods}, cfg, Path(args.out_dir), (i, n))
    print(f"FIT-DONE {','.join(args.methods)} shard {i}", flush=True)


if __name__ == "__main__":
    main()
