"""Stage 15 (compute) — generate the MAIN benchmark panel: outcomes drawn from
the reported model's prior on every eligible real design (383 designs, listed
in `data/real_designs/eligible_designs.csv` with their cleaned matrices beside
it, delivered by data_prep/real_designs/10_deliver_eligible.R).

    python scripts/15_gen_main_panel.py --config configs/paper_model.yaml \
        --per-cell 400 --out-dir results/main_benchmark/truth

Acceptance rule: for each of the 5 x 5 cells of realized density k_main/p by
realized R2 (0.1-wide bins, lower edges 0.1 ... 0.9, top bin closed), draw a
design uniformly at random from the pool, draw (gamma, beta, sigma2, y) from
the prior on its standardized X, and keep the draw if it lands in the cell and
every active interaction has both parents active (always true under a
strong-heredity prior), until the cell holds --per-cell datasets. Keys
`ip<panel>_<design>_f<i>g<j>_<k>`, panel default `mx` (main benchmark, real
X), npz layout = the benchmark's truth rows. Every cell has its own RNG stream
seeded by (--seed, cell index), so `--cell i` (one cell, e.g. one task of a
25-task job array) and a serial run write identical rows. Writes
`_grid/cell_<i>.json` with the cell's acceptance beside the rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BENCH = (0.1, 0.3, 0.5, 0.7, 0.9)
BIN = 0.1


def in_cell(value: float, target: float) -> bool:
    top = target + BIN >= 1.0
    return target <= value <= 1.0 if top else target <= value < target + BIN


def strong_heredity(gamma, p: int, cfg) -> bool:
    from linearpfn.prior import interaction_pairs

    active = set(np.flatnonzero(gamma[1: 1 + p]).tolist())
    for t, (a, b) in enumerate(interaction_pairs(p, cfg)):
        if gamma[1 + p + t] and not (a in active and b in active):
            return False
    return True


def load_pool(designs_csv: Path, p_cap: int) -> list[dict]:
    from linearpfn.prior import standardize

    base = designs_csv.parent
    pool = []
    for r in csv.DictReader(open(designs_csv)):
        n, p = int(r["n_delivered"]), int(r["p_delivered"])
        if p > p_cap:
            continue
        raw = np.loadtxt(base / r["clean_file"], delimiter=",", skiprows=1, ndmin=2)
        if raw.shape != (n, p):
            raise RuntimeError(f"{r['clean_file']}: shape {raw.shape} != ({n}, {p})")
        pool.append({"id": Path(r["clean_file"]).stem, "X": standardize(raw), "n": n, "p": p,
                     "domain": r["domain"], "cell": r["cell"]})
    return pool


def main() -> int:
    from linearpfn.prior import build_design, place_prior_params
    from linearpfn.train import load_config, make_prior_config

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="the prior whose DGP generates y")
    ap.add_argument("--designs",
                    default=str(ROOT / "data" / "real_designs" / "eligible_designs.csv"))
    ap.add_argument("--panel", default="mx")
    ap.add_argument("--per-cell", type=int, default=400)
    ap.add_argument("--max-draws", type=int, default=20_000_000)
    ap.add_argument("--p-cap", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "main_benchmark" / "truth"))
    ap.add_argument("--cell", type=int, default=None,
                    help="one cell index 0..24 (row-major over density x signal); default all")
    args = ap.parse_args()

    cfg = make_prior_config(load_config(args.config)["data"])
    pool = load_pool(Path(args.designs), args.p_cap)
    out = Path(args.out_dir)
    (out / "_grid").mkdir(parents=True, exist_ok=True)
    grid = [(ft, rt) for ft in BENCH for rt in BENCH]
    todo = list(range(len(grid))) if args.cell is None else [args.cell]
    cells = []
    for ci in todo:
        ft, rt = grid[ci]
        rng = np.random.default_rng([args.seed, ci])
        if True:
            t0 = time.monotonic()
            kept, draws, used = 0, 0, set()
            while kept < args.per_cell and draws < args.max_draws:
                d = pool[int(rng.integers(len(pool)))]
                X, p = d["X"], d["p"]
                ds = place_prior_params(rng, X, cfg)
                draws += 1
                k_main = int(ds.gamma[1: 1 + p].sum())
                sig = build_design(X, cfg) @ ds.beta
                r2 = float(np.var(sig) / np.var(ds.y))
                if not (in_cell(k_main / p, ft) and in_cell(r2, rt)):
                    continue
                if not strong_heredity(ds.gamma, p, cfg):
                    continue
                kept += 1
                used.add(d["id"])
                key = f"ip{args.panel}_{d['id']}_f{BENCH.index(ft)}g{BENCH.index(rt)}_{kept}"
                meta = {"panel": f"ip{args.panel}", "design": d["id"], "domain": d["domain"],
                        "n": int(X.shape[0]), "p": int(p), "frac_target": ft, "r2_target": rt,
                        "frac_realized": k_main / p, "realized_r2": r2, "k_main": k_main,
                        "hereditary": True, "n_int_realized": int(ds.gamma[1 + p:].sum()),
                        "k_int": int(ds.gamma[1 + p:].sum()),
                        "source": "prior DGP on every eligible real design, strong heredity"}
                np.savez_compressed(out / f"{key}.npz", X=X, y=ds.y, gamma_true=ds.gamma[1:],
                                    beta_true=ds.beta, n=X.shape[0], p=p, meta=json.dumps(meta))
            cells.append({"cell": ci, "f": ft, "r2": rt, "kept": kept, "draws": draws,
                          "rate": kept / draws if draws else 0.0, "designs_used": len(used),
                          "seconds": time.monotonic() - t0, "config": args.config,
                          "designs": args.designs, "pool": len(pool), "panel": args.panel,
                          "per_cell": args.per_cell, "seed": args.seed})
            (out / "_grid" / f"cell_{ci:02d}.json").write_text(json.dumps(cells[-1], indent=1))
            print(f"ip{args.panel} f={ft:g} r2={rt:g}: {kept}/{args.per_cell} kept from {draws} "
                  f"draws ({100 * kept / max(draws, 1):.3f} %), {len(used)} designs, "
                  f"{cells[-1]['seconds']:.0f}s", flush=True)
    print(f"{sum(c['kept'] for c in cells)} datasets -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
