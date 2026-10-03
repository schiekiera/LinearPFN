"""Stage 16 (compute) — generate the MISSPECIFICATION panels on the main
benchmark: every one of the main benchmark's 10,000 datasets with ONE stage of
the outcome model replaced (`bench.knobs`: coef | int | noise).

    python scripts/16_gen_knob_main.py --config configs/paper_model.yaml \
        --designs data/real_designs/eligible_designs.csv \
        --main-dir results/main_benchmark/truth \
        --out-dir results/misspecification/truth [--cell i] [--knobs coef int noise]

The main benchmark (stage 15) stores (X, y, gamma, beta) but not c or
sigma2, which the knobs need. This stage REPLAYS stage 15's per-cell
rejection stream `default_rng([seed15, cell])` draw for draw through stage
15's own acceptance rule, and every accepted draw is checked bit for bit
against the stored row (key, gamma, beta, y); a single mismatch aborts, so a
knob row can only be written from the exact draw the main benchmark scored.
The knob then runs on its OWN stream `default_rng([seed, knob id, cell,
k])`, so the replay is untouched by the knobs and each knob is independent of
which other knobs are generated.

Pairing: knob row `ip<panel>_<design>_f<i>g<j>_<k>` (panel xc / xi / xn,
`PANEL_OF` below) is the main row `ipmx_<design>_f<i>g<j>_<k>` with one stage
replaced — same design, active main effects, c and sigma2 — and its control
is that main row itself, already fitted. `frac_target` / `r2_target` are the
SOURCE draw's cell (density is unchanged by every knob; R2 moves under coef
and int), the knob draw's realized values sit beside the source's in meta.
Strong heredity is asserted on every row. One cell per call (--cell i, e.g.
one task of a job array); a serial run writes identical rows.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PANEL_OF = {"coef": "xc", "int": "xi", "noise": "xn"}   # x: the main benchmark's real-X panel mx
MAIN_PANEL = "mx"


def _stage15():
    spec = importlib.util.spec_from_file_location("gen_main_panel",
                                                  Path(__file__).with_name("15_gen_main_panel.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_row(path: Path, ds, key: str, atol: float = 0.0) -> float:
    """The replayed draw must be the stored main row: the key and gamma
    exactly, beta and y bit for bit unless atol > 0 (a different BLAS than
    the one that generated the main panel rounds them differently in the last
    place). Returns the largest absolute difference in beta or y."""
    if not path.exists():
        raise RuntimeError(f"replay produced {key}, which the main panel does not hold")
    z = np.load(path)
    if not np.array_equal(np.asarray(z["gamma_true"]), ds.gamma[1:]):
        raise RuntimeError(f"replay mismatch in gamma_true at {key}")
    diff = 0.0
    for name, ours in (("beta_true", ds.beta), ("y", ds.y)):
        d = float(np.max(np.abs(np.asarray(z[name], float) - ours), initial=0.0))
        if d > atol:
            raise RuntimeError(f"replay mismatch in {name} at {key}: max |diff| {d:.3g} > {atol:g}")
        diff = max(diff, d)
    return diff


def main() -> int:
    from bench.knobs import KNOB_ID, KNOBS, apply_knob
    from linearpfn.prior import build_design, place_prior_params
    from linearpfn.train import load_config, make_prior_config

    s15 = _stage15()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="the prior that generated the main panel")
    ap.add_argument("--designs",
                    default=str(ROOT / "data" / "real_designs" / "eligible_designs.csv"))
    ap.add_argument("--main-dir", default=str(ROOT / "results" / "main_benchmark" / "truth"))
    ap.add_argument("--knobs", nargs="+", default=list(PANEL_OF), choices=sorted(PANEL_OF))
    ap.add_argument("--p-cap", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260923, help="the knobs' streams")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "misspecification" / "truth"))
    ap.add_argument("--cell", type=int, default=None, help="one cell 0..24; default all")
    ap.add_argument("--max-kept", type=int, default=None,
                    help="stop each cell after this many rows (pilot / test only)")
    ap.add_argument("--atol", type=float, default=0.0,
                    help="tolerance on the stored beta and y (0 = bit for bit; > 0 only when "
                         "replaying under a different BLAS)")
    args = ap.parse_args()

    main_dir = Path(args.main_dir)
    grid_files = sorted((main_dir / "_grid").glob("cell_*.json"))
    rec = {int(json.loads(f.read_text())["cell"]): json.loads(f.read_text()) for f in grid_files}
    if len(rec) != 25:
        raise RuntimeError(f"{main_dir}/_grid holds {len(rec)} cell records, expected 25")
    seed15 = {r["seed"] for r in rec.values()}
    if len(seed15) != 1 or {r["panel"] for r in rec.values()} != {MAIN_PANEL}:
        raise RuntimeError("the main panel's cell records disagree on seed or panel")
    seed15 = seed15.pop()

    cfg = make_prior_config(load_config(args.config)["data"])
    pool = s15.load_pool(Path(args.designs), args.p_cap)
    if len(pool) != rec[0]["pool"]:
        raise RuntimeError(f"pool of {len(pool)} designs, the main panel used {rec[0]['pool']}")
    out = Path(args.out_dir)
    (out / "_grid").mkdir(parents=True, exist_ok=True)
    grid = [(ft, rt) for ft in s15.BENCH for rt in s15.BENCH]
    todo = list(range(len(grid))) if args.cell is None else [args.cell]
    for ci in todo:
        ft, rt = grid[ci]
        target = rec[ci]["kept"] if args.max_kept is None else min(args.max_kept, rec[ci]["kept"])
        rng = np.random.default_rng([seed15, ci])
        t0 = time.monotonic()
        kept, draws, dy_max = 0, 0, 0.0
        written = {k: 0 for k in args.knobs}
        while kept < target:                      # stage 15's loop, draw for draw
            d = pool[int(rng.integers(len(pool)))]
            X, p = d["X"], d["p"]
            ds = place_prior_params(rng, X, cfg)
            draws += 1
            if draws > rec[ci]["draws"]:
                raise RuntimeError(f"cell {ci}: replay needed more draws than stage 15 did")
            k_main = int(ds.gamma[1: 1 + p].sum())
            sig = build_design(X, cfg) @ ds.beta
            r2 = float(np.var(sig) / np.var(ds.y))
            if not (s15.in_cell(k_main / p, ft) and s15.in_cell(r2, rt)):
                continue
            if not s15.strong_heredity(ds.gamma, p, cfg):
                continue
            kept += 1
            tail = f"{d['id']}_f{s15.BENCH.index(ft)}g{s15.BENCH.index(rt)}_{kept}"
            source = f"ip{MAIN_PANEL}_{tail}"
            dy_max = max(dy_max, check_row(main_dir / f"{source}.npz", ds, source, args.atol))
            for knob in args.knobs:
                krng = np.random.default_rng([args.seed, KNOB_ID[knob], ci, kept])
                dk = apply_knob(krng, ds, X, cfg, knob)
                if not s15.strong_heredity(dk.gamma, p, cfg):
                    raise RuntimeError(f"knob {knob} produced a non-hereditary draw at {source}")
                sig_k = build_design(X, cfg) @ dk.beta
                var_y = float(np.var(dk.y))
                r2_k = float(np.var(sig_k) / var_y) if var_y > 0 else 0.0
                panel = PANEL_OF[knob]
                meta = {"panel": f"ip{panel}", "knob": knob, "source": source, "cell": ci,
                        "index": kept, "design": d["id"], "domain": d["domain"],
                        "n": int(X.shape[0]), "p": int(p), "frac_target": ft, "r2_target": rt,
                        "frac_realized": k_main / p, "realized_r2": r2_k,
                        "realized_r2_source": r2, "k_main": k_main, "hereditary": True,
                        "n_int_realized": int(dk.gamma[1 + p:].sum()),
                        "k_int": int(dk.gamma[1 + p:].sum()),
                        "k_int_source": int(ds.gamma[1 + p:].sum()),
                        "description": (f"main benchmark row {source} with knob {knob} "
                                        f"({KNOBS[knob]}), strong heredity")}
                np.savez_compressed(out / f"ip{panel}_{tail}.npz", X=X, y=dk.y,
                                    gamma_true=dk.gamma[1:], beta_true=dk.beta,
                                    n=X.shape[0], p=p, meta=json.dumps(meta))
                written[knob] += 1
        if args.max_kept is None and draws != rec[ci]["draws"]:
            # stage 15 stops at the last acceptance, so the counts must agree exactly
            raise RuntimeError(f"cell {ci}: {draws} draws replayed, stage 15 recorded "
                               f"{rec[ci]['draws']}")
        cell = {"cell": ci, "f": ft, "r2": rt, "kept": kept, "draws": draws,
                "verified": kept, "max_abs_diff": dy_max, "atol": args.atol,
                "written": written, "seconds": time.monotonic() - t0,
                "config": args.config, "designs": args.designs, "main_dir": args.main_dir,
                "seed15": seed15, "seed": args.seed, "panels": PANEL_OF}
        (out / "_grid" / f"cell_{ci:02d}.json").write_text(json.dumps(cell, indent=1))
        print(f"cell {ci} f={ft:g} r2={rt:g}: {kept} rows replayed and verified in {draws} "
              f"draws, written {written}, {cell['seconds']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
