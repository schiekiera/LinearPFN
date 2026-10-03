"""Stage 42 — where the pretraining mass lives on the (density x signal) plane.

Draws datasets from the training stream's own generator under a prior config
and bins them on the two axes the benchmark sweeps: the realized fraction
of active main effects (f = k_main / p) and the realized signal share
(R2 = var(Z beta) / var(y), the "true R2" of preflight). Ten equal bins per
axis, [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]; the top bin is closed so
f = 1 (every main active) is counted. Cell value = percent of drawn
datasets, so the grid sums to 100.

The exhibit answers "how much pretraining does a benchmark cell stand on":
the benchmark's 25 cells (F = R2 = .1/.3/.5/.7/.9) are marked at their
coordinates. Null datasets (no active effect, R2 exactly 0) are part of the
prior and are counted in the bottom-left cell; their share is reported
separately in the JSON and under the figure.

Writes reports/paper/prior_grid.json (cell
percentages, counts, marginals, conditional shares, per-cell mean p / n /
interaction count) and paper/figures/prior_grid.pdf (+png).
A seeded draw, no model fit.

Run:  python scripts/42_fig_prior_grid.py [--config <config>] [--batches 7500]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import CFG, CONFIG, FIG, NUM, P, store  # noqa: E402
from pipeline import figures  # noqa: E402
from pipeline.figures import GRID, INK, INK2, SURFACE  # noqa: E402

HUE = "#0072B2"          # house blue: the sequential ramp's dark end
EDGES = np.round(np.arange(0, 1.01, 0.1), 2)
BENCH = (0.1, 0.3, 0.5, 0.7, 0.9)   # the benchmark's F and R2 levels


def draw(cfg, batches: int, seed: int) -> dict[str, np.ndarray]:
    """Realized (f, R2) plus p, n and the interaction count per dataset."""
    from linearpfn.generator import sample_batch
    from linearpfn.prior import build_design

    f, r2, ps, ns, kint, kmain = [], [], [], [], [], []
    for i in range(batches):
        b = sample_batch(cfg, 8, seed, i)
        X, y, beta, g = b.X.numpy(), b.y.numpy(), b.beta.numpy(), b.gamma.numpy()
        n, p = X.shape[1], X.shape[2]
        for k in range(X.shape[0]):
            signal = build_design(X[k], cfg) @ beta[k]
            km = int(g[k, 1:1 + p].sum())
            f.append(km / p)
            r2.append(float(signal.var() / y[k].var()))
            ps.append(p), ns.append(n), kmain.append(km)
            kint.append(int(g[k, 1 + p:].sum()))
    return {"f": np.array(f), "r2": np.array(r2), "p": np.array(ps, float),
            "n": np.array(ns, float), "k_main": np.array(kmain, float),
            "k_int": np.array(kint, float)}


def bin_index(v: np.ndarray) -> np.ndarray:
    """Bin of each value: [0,.1) -> 0 ... [.9,1] -> 9 (top bin closed)."""
    return np.clip((v * 10).astype(int), 0, 9)


def tabulate(d: dict[str, np.ndarray]) -> dict:
    fi, ri = bin_index(d["f"]), bin_index(d["r2"])
    flat = fi * 10 + ri
    counts = np.bincount(flat, minlength=100).reshape(10, 10).astype(float)
    n_ds = counts.sum()
    pct = counts / n_ds * 100
    means = {}
    for key in ("p", "n", "k_int", "k_main"):
        s = np.bincount(flat, weights=d[key], minlength=100).reshape(10, 10)
        with np.errstate(invalid="ignore", divide="ignore"):
            means[key] = np.where(counts > 0, s / np.where(counts > 0, counts, 1), np.nan)
    null = float((d["k_main"] + d["k_int"] == 0).mean() * 100)
    sparse, sparse15 = d["f"] <= 0.2, d["f"] <= 0.15
    cells = {}
    for i in range(10):
        for j in range(10):
            cells[f"f={EDGES[i]:g}-{EDGES[i + 1]:g},r2={EDGES[j]:g}-{EDGES[j + 1]:g}"] = {
                "pct": round(float(pct[i, j]), 4), "n": int(counts[i, j]),
                "mean_p": None if np.isnan(means["p"][i, j]) else round(float(means["p"][i, j]), 2),
                "mean_n": None if np.isnan(means["n"][i, j]) else round(float(means["n"][i, j]), 1),
                "mean_k_int": (None if np.isnan(means["k_int"][i, j])
                               else round(float(means["k_int"][i, j]), 3))}
    bench = {f"f={f:g},r2={r:g}":
             round(float(pct[bin_index(np.array([f]))[0], bin_index(np.array([r]))[0]]), 4)
             for f in BENCH for r in BENCH}
    return {
        "n_datasets": int(n_ds), "pct": pct, "counts": counts, "means": means, "cells": cells,
        "marginal_f": [round(float(v), 3) for v in pct.sum(1)],
        "marginal_r2": [round(float(v), 3) for v in pct.sum(0)],
        "bench_cells_pct": bench,
        "summary": {
            "null_pct": round(null, 3),
            "no_active_main_pct": round(float((d["k_main"] == 0).mean() * 100), 3),
            "r2_le_zero_five_pct": round(float((d["r2"] <= 0.5).mean() * 100), 3),
            "r2_ge_zero_seven_pct": round(float((d["r2"] >= 0.7).mean() * 100), 3),
            "sparse_pct": round(float(sparse.mean() * 100), 3),
            "r2_ge_zero_seven_given_sparse_pct":
                round(float((d["r2"][sparse] >= 0.7).mean() * 100), 3),
            "r2_ge_zero_nine_given_sparse_pct":
                round(float((d["r2"][sparse] >= 0.9).mean() * 100), 3),
            "sparse_def": "k/p <= 0.2",
            "r2_ge_zero_seven_given_f_le_zero_one_five_pct":
                round(float((d["r2"][sparse15] >= 0.7).mean() * 100), 3),
            "r2_ge_zero_nine_given_f_le_zero_one_five_pct":
                round(float((d["r2"][sparse15] >= 0.9).mean() * 100), 3),
            "bench_cells_total_pct": round(float(sum(bench.values())), 3),
            "f_median": round(float(np.median(d["f"])), 3),
            "r2_median": round(float(np.median(d["r2"])), 3),
        }}


def _fmt(v: float) -> str:
    if v >= 1:
        return f"{v:.1f}"
    if v >= 0.1:
        return f"{v:.2f}"
    if v >= 0.005:
        return f"{v:.2f}"
    return "" if v == 0 else "·"


def _strip(ax, vals: np.ndarray, cmap, horizontal: bool, label: str) -> None:
    """Marginal totals as a one-cell-wide neutral strip with printed values. The
    bottom strip carries the x tick labels (the main axes would collide with it)."""
    m = vals.reshape(1, -1) if horizontal else vals.reshape(-1, 1)
    ax.imshow(np.zeros_like(m), cmap=cmap, vmin=0, vmax=1, aspect="auto",
              origin="lower", extent=(0, 1, 0, 1))
    for k, v in enumerate(vals):
        x, y = ((k + 0.5) / len(vals), 0.5) if horizontal else (0.5, (k + 0.5) / len(vals))
        ax.text(x, y, f"{v:.1f}", ha="center", va="center", fontsize=7, color=INK2)
    for e in EDGES:
        (ax.axvline if horizontal else ax.axhline)(e, color=SURFACE, linewidth=1.6)
    if horizontal:
        ax.set_xticks(EDGES)
        ax.set_xticklabels([f"{e:g}" for e in EDGES], fontsize=10, color=INK2)
        ax.set_yticks([])
        ax.set_xlabel(label, color=INK2, fontsize=12)
    else:
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(label, fontsize=10, color=INK2, pad=4)
    for sp in ax.spines.values():
        sp.set_color(GRID)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(P["final_config"]))
    ap.add_argument("--batches", type=int, default=7500, help="batches of 8 datasets")
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--out", default=str(FIG / "prior_grid.pdf"))
    ap.add_argument("--json", default=str(NUM / "prior_grid.json"))
    args = ap.parse_args()

    from linearpfn.train import load_config, make_prior_config

    cfg = make_prior_config(load_config(args.config)["data"])
    t = tabulate(draw(cfg, args.batches, args.seed))
    pct, s = t["pct"], t["summary"]

    figures.setup(CFG)
    cmap = LinearSegmentedColormap.from_list("mass", ["#eaf2fa", HUE])
    cmap.set_bad(SURFACE)
    neutral = LinearSegmentedColormap.from_list("flat", [SURFACE, SURFACE])
    fig = plt.figure(figsize=(7.2, 6.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=(10, 0.85), height_ratios=(10, 0.85),
                          wspace=0.04, hspace=0.04, left=0.10, right=0.86, top=0.97, bottom=0.12)
    ax = fig.add_subplot(gs[0, 0])
    lo = max(pct[pct > 0].min(), 1e-3)
    im = ax.imshow(np.ma.masked_where(pct == 0, pct).T, origin="lower", extent=(0, 1, 0, 1),
                   cmap=cmap, norm=LogNorm(vmin=lo, vmax=pct.max()), aspect="auto")
    for i in range(10):
        for j in range(10):
            v = pct[i, j]
            txt = _fmt(v)
            if not txt:
                continue
            dark = v > 0 and (np.log(max(v, lo)) - np.log(lo)) / (
                np.log(pct.max()) - np.log(lo)) > 0.80
            ax.text((i + 0.5) / 10, (j + 0.5) / 10, txt, ha="center", va="center",
                    fontsize=7, color="#ffffff" if dark else INK)
    for x in EDGES:                      # 2px surface gaps between the fills
        ax.axvline(x, color=SURFACE, linewidth=1.6)
        ax.axhline(x, color=SURFACE, linewidth=1.6)
    for f in BENCH:                      # the benchmark's 25 cells
        for r in BENCH:
            ax.plot(f, r, marker="o", markersize=6, markerfacecolor="none",
                    markeredgecolor=INK2, markeredgewidth=1.1, zorder=5)
    ax.set_xticks(EDGES), ax.set_yticks(EDGES)
    ax.set_xticklabels([])
    ax.set_yticklabels([f"{e:g}" for e in EDGES], fontsize=10, color=INK2)
    ax.set_ylabel("$R^2$", color=INK2, fontsize=12)
    for sp in ax.spines.values():
        sp.set_color(GRID)

    # right strip: one total per R2 row; bottom strip: one total per density column
    _strip(fig.add_subplot(gs[0, 1]), np.array(t["marginal_r2"]), neutral, False, "row %")
    _strip(fig.add_subplot(gs[1, 0]), np.array(t["marginal_f"]), neutral, True,
           "density $k_{\\mathrm{main}}/p$")
    cax = fig.add_axes((0.895, 0.30, 0.02, 0.40))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("% of prior draws (log scale)", color=INK2, fontsize=11)
    cb.ax.tick_params(colors=INK2, labelsize=10)
    cb.outline.set_edgecolor(GRID)

    written = figures.save(fig, Path(args.out))
    payload = {k: t[k] for k in ("n_datasets", "cells", "marginal_f", "marginal_r2",
                                 "bench_cells_pct", "summary")}
    payload["bin_edges"] = [float(e) for e in EDGES]
    payload["seed"], payload["batches"] = args.seed, args.batches
    payload["outputs"] = [str(w) for w in written]
    store.write_json(Path(args.json), payload, [CONFIG, Path(args.config)],
                     stage="42_fig_prior_grid")
    print(f"-> {args.json}\n-> " + "\n-> ".join(map(str, written)))
    print(f"{t['n_datasets']} datasets; null {s['null_pct']}%, "
          f"R2>=.7 | sparse {s['r2_ge_zero_seven_given_sparse_pct']}%, "
          f"R2>=.9 | sparse {s['r2_ge_zero_nine_given_sparse_pct']}%")


if __name__ == "__main__":
    main()
