"""Stage 78 — the MAIN benchmark's marginal sweeps on the real designs: ONE
figure with three meta-subfigures side by side, "Selection" (F1/ds), "Ranking"
(AUC/ds) and "Estimation" (coefficient RMSE), each a 2 x 2 grid by default.

Grid of each block: top left by p bin, top right by n bin, bottom left by
density cell (k_main/p), bottom right by R2 cell (BLOCK_SHAPE = (4, 1) stacks
the four in one column instead). Every point is the per-dataset macro average
over the datasets of that slice; every curve is one method. House Okabe-Ito
mapping, shared legend, off-scale rule for RMSE (a method with any point above
the cap is dropped from that panel and named in it), no figure title. The
coefficients of stability are its refit. The exact posterior is NOT drawn:
it exists on the p <= 5 datasets only, so its curve would not be scored on the
same datasets as the others in any slice but the first p bin; its numbers are in
the tables.

Reads reports/paper/ip_tables.json (stage 70) — never the npz rows.

Every label, font size and spacing is a setting in the FIGURE SETTINGS block
below the imports; edit there and rerun.

Run:  python scripts/78_fig_ip_sweeps_three.py [--tables reports/paper/ip_tables.json]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import sys  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import CFG, CONFIG, FIG, NUM, PROV, store  # noqa: E402
from pipeline import figures, sweeps  # noqa: E402
from pipeline.figures import INK2, SURFACE  # noqa: E402

# ============================================================================
# FIGURE SETTINGS: every label, font size and spacing of the figure lives here.
# Sizes are in points at the figure's own size; the paper sets the figure at
# \\linewidth (5.5 in), so at a width of 19.5 in everything prints at about
# 0.28 of these values.
# ============================================================================

# -- figure ------------------------------------------------------------------
FIG_SIZE = (19.5, 6.8)          # inches (width, height)
SUBFIG_WSPACE = 0.04            # gap between the three blocks
BLOCK_SHAPE = (2, 2)            # (rows, columns) of the four slices inside one block
# the grid inside each block; the block's title is centred over it
GRID = {"left": 0.13, "right": 0.98, "top": 0.9, "bottom": 0.14,
        "wspace": 0.34, "hspace": 0.5}

# -- labels ------------------------------------------------------------------
TITLE_SELECTION = "Selection"
TITLE_RANKING = "Ranking"
TITLE_COEFFICIENTS = "Estimation"
YLABEL_F1 = "F1/ds"
YLABEL_AUC = "AUC/ds"
YLABEL_RMSE = "coefficient RMSE"
# x labels are set in math, the same way the caption writes them
XLABEL_P = "$p$"
XLABEL_N = "$n$"
XLABEL_DENSITY = "$k_{\\mathrm{main}}/p$"
XLABEL_SIGNAL = "$R^2$"
PANEL_TITLE_P = "by predictors"
PANEL_TITLE_N = "by rows"
PANEL_TITLE_DENSITY = "by density"
PANEL_TITLE_SIGNAL = "by signal"

# -- y ranges ------------------------------------------------------------------
F1_YLIM = (0, 1)
AUC_YLIM = (0.65, 1)            # the slice averages lie in 0.71-0.99
RMSE_CAP_FACTOR = 5.0           # points above this multiple of the median point go off-scale
OFFSCALE_DROP_LINE = True       # a method with any off-scale point is dropped from that panel
OFFSCALE_NOTE_PAD = 1.3         # axis top over the highest remaining point when a note is shown

# -- font sizes ----------------------------------------------------------------
FONTSIZE_SECTION_TITLE = 16     # "Selection", "Ranking", "Estimation"
FONTSIZE_PANEL_TITLE = 12       # "by predictors", "by rows", ...
FONTSIZE_AXIS_LABEL = 12        # x and y axis labels
FONTSIZE_TICK = 8               # tick labels
XTICK_ROTATION_N = 0            # degrees; 0 keeps the n-bin labels horizontal
FONTSIZE_OFFSCALE_NOTE = 7.5    # "<method> off-scale" in an RMSE panel
FONTSIZE_LEGEND = 14            # the shared legend below the panels

# -- curves --------------------------------------------------------------------
LINE_WIDTH = 2
MARKER_SIZE = 5
CURVE_ALPHA = 0.95

# -- legend --------------------------------------------------------------------
LEGEND_MAX_COLUMNS = 8
LEGEND_ANCHOR = (0.5, -0.02)    # (x, y) of its lower centre, in figure coordinates
LEGEND_HANDLE_LENGTH = 3.0      # length of a line sample, in font sizes
LEGEND_MARKER_SCALE = 1.6       # legend marker size relative to the curves' markers
LEGEND_COLUMN_SPACING = 2.2     # space between entries, in font sizes
LEGEND_LINE_WIDTH = 3.0         # line width of the samples

# ============================================================================

# measure -> (metric key, y label, block title, y range; None = off-scale rule)
MEASURES = {"f1": ("f1_macro", YLABEL_F1, TITLE_SELECTION, F1_YLIM),
            "auc": ("auc_macro", YLABEL_AUC, TITLE_RANKING, AUC_YLIM),
            "rmse": ("rmse_macro", YLABEL_RMSE, TITLE_COEFFICIENTS, None)}
SLICES = (("p_bin", XLABEL_P, PANEL_TITLE_P), ("n_bin", XLABEL_N, PANEL_TITLE_N),
          ("density", XLABEL_DENSITY, PANEL_TITLE_DENSITY),
          ("signal", XLABEL_SIGNAL, PANEL_TITLE_SIGNAL))


def draw_grid(sub, panel: dict | None, measure: str, methods, top: str) -> None:
    metric, mlabel, title, ylim = MEASURES[measure]
    axes = np.atleast_1d(sub.subplots(*BLOCK_SHAPE))
    for ax, (key, xlab, ptitle) in zip(axes.ravel(), SLICES, strict=True):
        figures.style_axes(ax, ylabel=mlabel, xlabel=xlab, title=ptitle,
                           tick_size=FONTSIZE_TICK, label_size=FONTSIZE_AXIS_LABEL,
                           title_size=FONTSIZE_PANEL_TITLE)
        if not panel:
            continue
        curves = []
        for name, label, hue, marker in methods:
            xs, ys = sweeps.series(panel, key, name, metric, CFG)
            if not xs or np.all(np.isnan(ys)):
                continue
            curves.append((label, ys))
            ax.plot(range(len(xs)), ys, color=hue, linewidth=LINE_WIDTH, marker=marker,
                    markersize=MARKER_SIZE, label=label, zorder=5 if name == top else 3,
                    alpha=CURVE_ALPHA)
            ax.set_xticks(range(len(xs)))
            if key == "n_bin" and XTICK_ROTATION_N:
                ax.set_xticklabels(xs, rotation=XTICK_ROTATION_N, ha="right",
                                   rotation_mode="anchor")
            else:
                ax.set_xticklabels(xs)
        if ylim is None:
            figures.rmse_cap(ax, curves, RMSE_CAP_FACTOR, note_size=FONTSIZE_OFFSCALE_NOTE,
                             drop_line=OFFSCALE_DROP_LINE, note_pad=OFFSCALE_NOTE_PAD)
        else:
            ax.set_ylim(*ylim)
    sub.suptitle(title, color=figures.INK, fontsize=FONTSIZE_SECTION_TITLE, fontweight="bold",
                 x=(GRID["left"] + GRID["right"]) / 2, ha="center")
    sub.subplots_adjust(**GRID)


def draw(tables: dict, out: Path) -> list[Path]:
    panel = sweeps.panel_of(tables, CFG)
    methods = sweeps.methods(tables, CFG)
    top = CFG["ip"]["model_row"]
    fig = plt.figure(figsize=FIG_SIZE, facecolor=SURFACE)
    subs = fig.subfigures(1, len(MEASURES), wspace=SUBFIG_WSPACE)
    for sub, measure in zip(subs, MEASURES, strict=True):
        sub.set_facecolor(SURFACE)
        draw_grid(sub, panel, measure, methods, top)
    handles, labels = {}, []
    for ax in fig.get_axes():
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in handles:
                handles[lab] = h
                labels.append(lab)
    leg = fig.legend([handles[lab] for lab in labels], labels, loc="lower center",
                     ncol=min(len(labels), LEGEND_MAX_COLUMNS), frameon=False,
                     fontsize=FONTSIZE_LEGEND, labelcolor=INK2, bbox_to_anchor=LEGEND_ANCHOR,
                     handlelength=LEGEND_HANDLE_LENGTH, markerscale=LEGEND_MARKER_SCALE,
                     columnspacing=LEGEND_COLUMN_SPACING)
    for line in leg.get_lines():
        line.set_linewidth(LEGEND_LINE_WIDTH)
    return figures.save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tables", default=str(NUM / "ip_tables.json"))
    ap.add_argument("--out-dir", default=str(FIG))
    args = ap.parse_args()
    tables = store.read_json(args.tables)
    if sweeps.panel_of(tables, CFG) is None:
        print("panel absent from ip_tables.json — pending pull; nothing written")
        return
    figures.setup(CFG)
    written = draw(tables, Path(args.out_dir) / "ip_sweeps_three.pdf")
    store.write_json(PROV / "fig_ip_sweeps_three.json",
                     {"outputs": [str(w) for w in written], "methods": tables["methods"],
                      "pending_methods": tables["pending_methods"]},
                     [CONFIG, Path(args.tables)], stage="78_fig_ip_sweeps_three")
    print("-> " + "\n-> ".join(map(str, written)))


if __name__ == "__main__":
    main()
