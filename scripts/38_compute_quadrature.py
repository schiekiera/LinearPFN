"""Stage 38 (compute) — pick the quadrature node counts for the exact reference.

For each candidate Gauss-Legendre grid, computes every model's marginalized
log evidence and the PIP vector on a panel of prior-drawn datasets, and
reports the maximum absolute deviation from the FINEST grid (the truth
proxy) — the chosen grid must sit at least two orders of magnitude below
the ~1e-3 metric precision (<= 1e-5 in both log evidence and PIP). Also
times one full fit_exact per cell and records its kept-component count at
the default pruning tolerance (the n=32/p=5 cell is the diffuse-posterior
component-count probe).

Interactions cells (p in {3, 5} x n in {32, 1024}; n=1024 is where the
evidence is sharpest in log c) decide the grid; mains-only cells (p in
{10, 15}, c-axis only) are recorded for a mains-only prior and do not decide
it.

Uses the evidence internals directly (enumerate + evidence_pass) so the
grid sweep does not pay the moment pass per grid.

Run: python scripts/38_compute_quadrature.py --config configs/paper_model.yaml \
     --n-datasets 12 --out reports/quadrature
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import replace

import numpy as np
from scipy.special import logsumexp

ROOT = next(p for p in pathlib.Path(__file__).resolve().parents

            if (p / "pyproject.toml").is_file())  # repo root, any depth

sys.path.insert(0, str(ROOT))

from linearpfn._quadrature import evidence_pass, node_grid  # noqa: E402
from linearpfn.prior import PriorConfig, build_design, sample_dataset  # noqa: E402
from linearpfn.reference import _design_stats, enumerate_models, fit_exact  # noqa: E402
from linearpfn.train import load_config, make_prior_config  # noqa: E402

# candidate (c, rho) node counts: {12, 15} x {5, 7, 9} plus 20-c-node grids,
# because the c-axis is the sharp direction at n=1024 while rho saturates at
# 9 nodes.
INT_GRIDS = [(12, 5), (12, 7), (12, 9), (15, 5), (15, 7), (15, 9), (20, 7), (20, 9)]
INT_TRUTH = (30, 15)
INT_CELLS = [(32, 3), (1024, 3), (32, 5), (1024, 5)]
MAINS_GRIDS = [12, 15, 20, 30]
MAINS_TRUTH = 40
MAINS_CELLS = [(128, 10), (128, 15)]
# the finest grid of each panel, the truth proxy every deviation is measured against
TRUTH = {"interactions": list(INT_TRUTH), "mains_only": [MAINS_TRUTH, 1]}


def _evidence_and_pip(
    stats, gammas, log_prior, cfg, p, n_c: int, n_rho: int
) -> tuple[np.ndarray, np.ndarray]:
    grid = node_grid(cfg, n_c, n_rho)
    log_ev, *_rest = evidence_pass(stats, gammas, log_prior, cfg, p, *grid)
    log_w = log_prior + log_ev
    log_w -= logsumexp(log_w)
    return log_ev, np.exp(log_w) @ gammas


def _sweep(
    cfg: PriorConfig, cells, grids, truth, n_datasets: int, seed: int,
    fit_probe: bool = True, skip: int = 0,
) -> list[dict]:
    rows: list[dict] = []
    for n, p in cells:
        rng = np.random.default_rng([seed, n, p])
        gammas, log_prior = enumerate_models(p, cfg)
        devs = {g: {"dlogev": 0.0, "dpip": 0.0} for g in grids}
        fit_seconds = comp_count = None
        for _ in range(skip):  # burn the deterministic prefix (chunked runs)
            sample_dataset(rng, cfg, n=n, p=p)
        for i in range(n_datasets):
            ds = sample_dataset(rng, cfg, n=n, p=p)
            stats = _design_stats(build_design(ds.X, cfg), ds.y)
            ev_t, pip_t = _evidence_and_pip(stats, gammas, log_prior, cfg, p, *truth)
            for g in grids:
                ev_g, pip_g = _evidence_and_pip(stats, gammas, log_prior, cfg, p, *g)
                devs[g]["dlogev"] = max(devs[g]["dlogev"], float(np.abs(ev_g - ev_t).max()))
                devs[g]["dpip"] = max(devs[g]["dpip"], float(np.abs(pip_g - pip_t).max()))
            print(f"  cell n={n} p={p}: dataset {i + 1}/{n_datasets}", flush=True)
            if i == 0 and fit_probe:  # timing + component probe, first dataset
                t0 = time.perf_counter()
                post = fit_exact(ds.X, ds.y, cfg, standardization_tol=1.0)
                fit_seconds = time.perf_counter() - t0
                comp_count = int(post.comp_model.size)
                del post
        for g in grids:
            rows.append({
                "n": n, "p": p, "grid": list(g), "models": int(gammas.shape[0]),
                "max_dlogev": devs[g]["dlogev"], "max_dpip": devs[g]["dpip"],
                "fit_seconds_default_grid": fit_seconds,
                "components_default_grid": comp_count,
            })
        print(f"cell n={n} p={p} ({gammas.shape[0]} models) done", flush=True)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Quadrature grid convergence check")
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-datasets", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="reports/check_quadrature")
    parser.add_argument(
        "--cells", default=None,
        help="restrict to 'n,p;n,p' cells and write a per-chunk JSON "
             "(chunked runs bound wall-clock and memory per invocation; "
             "use --merge afterwards)",
    )
    parser.add_argument("--panel", choices=["interactions", "mains_only"], default=None,
                        help="with --cells: which panel the cells belong to")
    parser.add_argument("--no-fit-probe", action="store_true",
                        help="skip the full fit_exact timing probe (multi-GB "
                             "on diffuse p=5 cells)")
    parser.add_argument("--skip", type=int, default=0,
                        help="with --cells: skip the first K datasets of the "
                             "deterministic stream (chunked runs; merge takes "
                             "the max over chunks)")
    parser.add_argument("--grids", default=None,
                        help="restrict candidate grids to 'c,r;c,r'")
    parser.add_argument("--merge", action="store_true",
                        help="merge chunk_*.json in --out into results.json + table.md")
    args = parser.parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        _merge(out, args.n_datasets, args.config)
        return
    cfg = (
        make_prior_config(load_config(args.config)["data"])
        if args.config
        else PriorConfig()
    )
    if not (cfg.random_c or cfg.random_rho_int):
        raise SystemExit("prior has no random scales: nothing to check")

    probe = not args.no_fit_probe
    if args.cells:
        cells = [tuple(int(v) for v in c.split(",")) for c in args.cells.split(";")]
        panel = args.panel or "interactions"
        grids = (
            [tuple(int(v) for v in g.split(",")) for g in args.grids.split(";")]
            if args.grids
            else (INT_GRIDS if panel == "interactions" else [(k, 1) for k in MAINS_GRIDS])
        )
        if panel == "interactions":
            rows = _sweep(cfg, cells, grids, INT_TRUTH, args.n_datasets,
                          args.seed, fit_probe=probe, skip=args.skip)
        else:
            mains_cfg = replace(cfg, include_interactions=False)
            rows = _sweep(mains_cfg, cells, grids, (MAINS_TRUTH, 1),
                          args.n_datasets, args.seed, fit_probe=probe,
                          skip=args.skip)
        tag = args.cells.replace(",", "_").replace(";", "-")
        if args.grids:
            tag += "_g" + args.grids.replace(",", "x").replace(";", "-")
        if args.skip:
            tag += f"_s{args.skip}"
        (out / f"chunk_{panel}_{tag}.json").write_text(
            json.dumps({"panel": panel, "rows": rows}, indent=1)
        )
        print(f"chunk written: {out}/chunk_{panel}_{tag}.json")
        return

    results: dict = {"config_is_default": args.config is None, "n_datasets": args.n_datasets,
                     "config": args.config, "truth": TRUTH}
    if cfg.include_interactions:
        results["interactions"] = _sweep(
            cfg, INT_CELLS, INT_GRIDS, INT_TRUTH, args.n_datasets, args.seed,
            fit_probe=probe,
        )
        mains_cfg = replace(cfg, include_interactions=False)
    else:
        mains_cfg = cfg
    results["mains_only"] = _sweep(
        mains_cfg,
        MAINS_CELLS,
        [(k, 1) for k in MAINS_GRIDS],
        (MAINS_TRUTH, 1),
        args.n_datasets,
        args.seed,
        fit_probe=probe,
    )
    (out / "results.json").write_text(json.dumps(results, indent=1))
    _write_table(out, results, args.n_datasets)


def _merge(out: pathlib.Path, n_datasets: int, config: str | None = None) -> None:
    """Merge chunk files; duplicate (panel, n, p, grid) rows (dataset-range
    chunks of one cell) merge by elementwise max of the deviation maxima."""
    merged: dict[tuple, dict] = {}
    for f in sorted(out.glob("chunk_*.json")):
        chunk = json.loads(f.read_text())
        for row in chunk["rows"]:
            key = (chunk["panel"], row["n"], row["p"], tuple(row["grid"]))
            if key in merged:
                merged[key]["max_dlogev"] = max(merged[key]["max_dlogev"], row["max_dlogev"])
                merged[key]["max_dpip"] = max(merged[key]["max_dpip"], row["max_dpip"])
            else:
                merged[key] = dict(row)
    results: dict = {"config_is_default": config is None, "n_datasets": n_datasets,
                     "config": config, "truth": TRUTH,
                     "interactions": [], "mains_only": []}
    for (panel, *_), row in sorted(merged.items(), key=lambda kv: kv[0]):
        results[panel].append(row)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    _write_table(out, results, n_datasets)


def _write_table(out: pathlib.Path, results: dict, n_datasets: int) -> None:
    lines = [
        "# Quadrature grid convergence",
        "",
        f"Max |dev| vs the finest grid (interactions {INT_TRUTH}, mains c-{MAINS_TRUTH}) "
        f"over {n_datasets} datasets/cell.",
        "",
        "| panel | n | p | grid (c x rho) | models | max dlogev | max dPIP |",
        "|---|---|---|---|---|---|---|",
    ]
    for panel in ("interactions", "mains_only"):
        for row in results.get(panel, []):
            lines.append(
                f"| {panel} | {row['n']} | {row['p']} | {row['grid'][0]}x{row['grid'][1]} "
                f"| {row['models']} | {row['max_dlogev']:.3e} | {row['max_dpip']:.3e} |"
            )
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
