"""Deterministic matplotlib output and the house method palette.

`save()` strips every timestamp/producer field so a rerun on unchanged
numbers is byte-identical (PDF for the manuscript, PNG preview alongside).
Baselines take the Okabe-Ito palette (stability selection's black slot
deliberately outside the lightness band); the model's rows take blue shades.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dddcd8", "#ffffff"
PALETTE: dict[str, tuple[str, str, str]] = {  # name -> (label, hue, marker)
    "susie": ("SuSiE", "#E69F00", "s"),
    "glinternet": ("glinternet", "#009E73", "^"),
    "hierNet": ("hierNet", "#D55E00", "D"),
    "lasso": ("lasso", "#CC79A7", "v"),
    "stability": ("stability", "#000000", "P"),
    # stability selection with an OLS refit on its selected support (stage 19): the
    # same selections as "stability", so it is drawn and labelled the same way
    "stability_refit": ("stability", "#000000", "P"),
    "exact": ("exact posterior", "#7f7f7f", "*"),
    "mc3": ("MCMC sampler", "#a0a0a0", "x"),
    "linearpfn_strong": ("LinearPFN", "#0072B2", "o"),
    "linearpfn_strong_nonll": ("LinearPFN, no predictive NLL", "#8fbce6", "o"),
}
ORDER = ("susie", "glinternet", "hierNet", "lasso", "stability", "stability_refit",
         "exact", "linearpfn_strong")


def style_of(name: str) -> tuple[str, str, str]:
    return PALETTE.get(name, (name, "#999999", "."))


def ordered(names: list[str]) -> list[str]:
    """ORDER first, then any other row with a PALETTE entry; rows without one
    (methods outside the method set that a results cache may hold) are dropped."""
    known = [n for n in ORDER if n in names]
    return known + sorted(n for n in names if n not in ORDER and n in PALETTE)


def setup(cfg: dict[str, Any] | None = None) -> None:
    fcfg = (cfg or {}).get("figures", {})
    # the paper is set in Times, so the figures
    # use the same face: Times New Roman with STIX math, which is Times-shaped
    family = fcfg.get("family", "serif")
    face = fcfg.get("font", "Times New Roman")
    plt.rcParams.update({
        "font.family": family,
        f"font.{family}": [face, "Times", "STIXGeneral", "DejaVu Serif", "DejaVu Sans"],
        "mathtext.fontset": fcfg.get("mathtext", "stix"),
        "figure.dpi": fcfg.get("dpi", 200), "savefig.dpi": fcfg.get("dpi", 200),
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.hashsalt": "linearpfn",
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    })


def style_axes(ax, ylabel=None, xlabel=None, title=None, *, tick_size: float = 8,
               label_size: float = 9, title_size: float = 10) -> None:
    """House axis style; the font sizes default to the house values, and a figure
    script may pass its own (e.g. 78_fig_ip_sweeps_three.py keeps them as settings)."""
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=tick_size)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=label_size)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=label_size)
    if title:
        ax.set_title(title, color=INK, fontsize=title_size, loc="left")


def rmse_cap(ax, series: list[tuple[str, Any]], factor: float = 5.0, *,
             note_size: float = 7.5, drop_line: bool = False,
             note_pad: float = 1.1) -> list[str]:
    """Robust y-cap for RMSE curves. Points above `factor` x the median of
    every plotted point are off-scale (an OLS refit on near-singular real
    designs can produce per-slice means in the hundreds); the axis is capped
    at 1.1 x the largest on-scale point and every curve with an off-scale
    point is named in the panel with the count; the curve is broken at those
    points. With `drop_line` a curve with any off-scale point is removed from
    the panel altogether, the axis is set by the remaining curves, and the
    note names the curve without a count.
    `note_pad` is the axis top over the largest remaining point in a panel that
    carries a note, so the note has room above the curves. Returns the labels
    pushed off."""
    import numpy as np

    pts = np.concatenate([np.asarray(y, float) for _lab, y in series]) if series else np.array([])
    pts = pts[~np.isnan(pts)]
    if pts.size == 0:
        return []
    cap_at = factor * float(np.median(pts))
    off = [(lab, int(np.sum(np.asarray(y, float) > cap_at))) for lab, y in series]
    off = [(lab, k) for lab, k in off if k]
    if drop_line:
        gone = {lab for lab, _k in off}
        for line in list(ax.get_lines()):
            if line.get_label() in gone:
                line.remove()
        kept = [np.asarray(y, float) for lab, y in series if lab not in gone]
        on = np.concatenate(kept) if kept else np.array([])
        on = on[~np.isnan(on)]
    else:
        on = pts[pts <= cap_at]
        for line in ax.get_lines():  # break the curve at off-scale points instead of a spike
            y = np.asarray(line.get_ydata(), float)
            if np.any(y > cap_at):
                line.set_ydata(np.where(y > cap_at, np.nan, y))
    pad = note_pad if off else 1.1
    ax.set_ylim(0, pad * float(on.max()) if on.size else cap_at)
    for i, (lab, k) in enumerate(off):
        note = f"{lab} off-scale" if drop_line else f"{lab} off-scale ({k})"
        ax.text(0.03, 0.95 - 0.09 * i, note,
                transform=ax.transAxes, fontsize=note_size, color=INK2, va="top")
    return [lab for lab, _k in off]


def save(fig, pdf_path: Path | str, png: bool = True) -> list[Path]:
    """PDF (+ PNG preview) without timestamps or producer strings."""
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, facecolor=SURFACE,
                metadata={"CreationDate": None, "ModDate": None, "Producer": None,
                          "Creator": None, "Title": None})
    written = [pdf_path]
    if png:
        png_path = pdf_path.with_suffix(".png")
        fig.savefig(png_path, facecolor=SURFACE, metadata={"Software": None})
        written.append(png_path)
    plt.close(fig)
    return written
