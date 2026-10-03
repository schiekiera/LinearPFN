"""Stage 75 — the sweeps of the benchmark with outcomes OUTSIDE the prior: the
layout of stage 78 (three blocks, "Selection" F1/ds, "Ranking" AUC/ds and
"Estimation" coefficient RMSE, each a 2 x 2 grid by p bin, n bin,
density cell and R2 cell) repeated in one row per variant: fixed-magnitude
coefficients, many large interactions, heavy-tailed noise.

Every point is the per-dataset macro average over the datasets of that slice
of one variant; every curve is one method. The density and R2 cells are those
of the main-benchmark dataset each variant is built on (a variant keeps the
predictor matrix and the active main effects, so its density is the control's;
its own R2 can differ from the cell). House Okabe-Ito mapping, shared legend,
off-scale rule for RMSE (a method with any point above the cap is dropped from
that panel and named in it), no figure title. The exact posterior is NOT drawn:
it exists on the p <= 5 datasets only, so its curve would not be scored on the
same datasets as the others in any slice but the first p bin; its numbers are in
the tables.

Reads reports/paper/knob_tables.json (stage 74) — never the npz rows.

Every label, font size and spacing is a setting in the FIGURE SETTINGS block
below the imports; edit there and rerun.

Run:  python scripts/75_fig_knob_sweeps.py [--tables reports/paper/knob_tables.json]
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
# \\textwidth (5.5 in), so at a width of 19.5 in everything prints at about
# 0.28 of these values, the scale of the main-text sweep figure.
# ============================================================================

# -- figure ------------------------------------------------------------------
FIG_SIZE = (19.5, 21.5)           # inches (width, height)
ROW_HSPACE = 0.02               # gap between the three variant rows
BLOCK_WSPACE = 0.04             # gap between the three blocks of a row
BOTTOM_MARGIN = 0.04            # share of the figure height kept free for the legend
# the 2 x 2 grid inside each block; the block's title is centred over it
GRID = {"left": 0.13, "right": 0.98, "top": 0.8, "bottom": 0.13,
        "wspace": 0.34, "hspace": 0.5}
ROW_TITLE_Y = 0.985             # height of a row's title inside the row
BLOCK_TITLE_Y = 0.9             # height of a block's title inside the block

# -- labels ------------------------------------------------------------------
# one row per variant, top to bottom, in the order of the paper
ROWS = (("coef", "Fixed-magnitude coefficients"),
        ("int", "Many large interactions"),
        ("noise", "Heavy-tailed noise"))
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
AUC_YLIM = (0.65, 1)            # the slice averages lie in 0.67-0.99
RMSE_CAP_FACTOR = 5.0           # points above this multiple of the median point go off-scale
OFFSCALE_DROP_LINE = True       # a method with any off-scale point is dropped from that panel
OFFSCALE_NOTE_PAD = 1.3         # axis top over the highest remaining point when a note is shown

# -- font sizes ----------------------------------------------------------------
FONTSIZE_ROW_TITLE = 20         # "Fixed-magnitude coefficients", ...
FONTSIZE_SECTION_TITLE = 16     # "Selection", "Ranking", "Estimation"
FONTSIZE_PANEL_TITLE = 12       # "by predictors", "by rows", ...
FONTSIZE_AXIS_LABEL = 12        # x and y axis labels
FONTSIZE_TICK = 8               # tick labels
XTICK_ROTATION_N = 0            # degrees; 0 keeps the n-bin labels horizontal
FONTSIZE_OFFSCALE_NOTE = 7.5    # "<method> off-scale" in an RMSE panel
FONTSIZE_LEGEND = 16            # the shared legend below the panels

# -- curves --------------------------------------------------------------------
LINE_WIDTH = 2
MARKER_SIZE = 5
CURVE_ALPHA = 0.95

# -- legend --------------------------------------------------------------------
LEGEND_MAX_COLUMNS = 8
LEGEND_ANCHOR = (0.5, 0.0)      # (x, y) of its lower centre, in figure coordinates
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


def draw_block(sub, panel: dict, measure: str, methods, top: str) -> None:
    metric, mlabel, title, ylim = MEASURES[measure]
    axes = sub.subplots(2, 2)
    for ax, (key, xlab, ptitle) in zip(axes.ravel(), SLICES, strict=True):
        figures.style_axes(ax, ylabel=mlabel, xlabel=xlab, title=ptitle,
                           tick_size=FONTSIZE_TICK, label_size=FONTSIZE_AXIS_LABEL,
                           title_size=FONTSIZE_PANEL_TITLE)
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
                 x=(GRID["left"] + GRID["right"]) / 2, y=BLOCK_TITLE_Y, ha="center")
    sub.subplots_adjust(**GRID)


def draw(tables: dict, out: Path) -> list[Path]:
    top = tables["model_row"]
    fig = plt.figure(figsize=FIG_SIZE, facecolor=SURFACE)
    body, _legend_strip = fig.subfigures(2, 1, height_ratios=[1 - BOTTOM_MARGIN, BOTTOM_MARGIN])
    body.set_facecolor(SURFACE)
    _legend_strip.set_facecolor(SURFACE)
    rows = body.subfigures(len(ROWS), 1, hspace=ROW_HSPACE)
    for row, (knob, row_title) in zip(rows, ROWS, strict=True):
        row.set_facecolor(SURFACE)
        panel = tables["panels"][knob]
        methods = sweeps.methods_from(panel["methods"], CFG)
        row.suptitle(row_title, color=figures.INK, fontsize=FONTSIZE_ROW_TITLE,
                     fontweight="bold", x=0.01, y=ROW_TITLE_Y, ha="left", va="top")
        blocks = row.subfigures(1, len(MEASURES), wspace=BLOCK_WSPACE)
        for sub, measure in zip(blocks, MEASURES, strict=True):
            sub.set_facecolor(SURFACE)
            draw_block(sub, panel, measure, methods, top)
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
    ap.add_argument("--tables", default=str(NUM / "knob_tables.json"))
    ap.add_argument("--out-dir", default=str(FIG))
    args = ap.parse_args()
    tables = store.read_json(args.tables)
    missing = [k for k, _ in ROWS if k not in tables.get("panels", {})]
    if missing:
        print(f"variant panels {missing} absent from knob_tables.json — pending pull; "
              "nothing written")
        return
    figures.setup(CFG)
    written = draw(tables, Path(args.out_dir) / "knob_sweeps.pdf")
    store.write_json(PROV / "fig_knob_sweeps.json",
                     {"outputs": [str(w) for w in written], "knobs": [k for k, _ in ROWS],
                      "methods": {k: tables["panels"][k]["methods"] for k, _ in ROWS},
                      "pending": tables["pending"]},
                     [CONFIG, Path(args.tables)], stage="75_fig_knob_sweeps")
    print("-> " + "\n-> ".join(map(str, written)))


if __name__ == "__main__":
    main()
