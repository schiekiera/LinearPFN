"""Data helpers of the sweep figures (stages 75 and 78): which
methods are drawn and in which order, how the bins of one slice are ordered,
and the per-bin series of one method and metric from `ip_tables.json` or
`knob_tables.json` (the same slice layout per panel).

The figure scripts keep every label, size and spacing in their own settings
block; this module only reads the numbers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pipeline import figures


def methods(tables: dict, cfg: dict) -> list[tuple[str, str, str, str]]:
    """(name, label, hue, marker): the baselines in house order, then the model row."""
    return methods_from(tables["methods"], cfg)


def methods_from(present: list[str], cfg: dict) -> list[tuple[str, str, str, str]]:
    """`methods` for an explicit list of method names (a knob panel lists its own)."""
    c = cfg["ip"]
    names = [m for m in c["baselines"] if m in present]
    names += [m for m in (c["model_row"],) if m in present]
    return [(n, *figures.style_of(n)) for n in names]


def bin_order(key: str, labels: list[str], cfg: dict) -> list[str]:
    """p and n bins in the configured order, density and R2 cells by lower edge."""
    if key in ("p_bin", "n_bin"):
        return [b for b in cfg["benchmark"][f"{key}s"] if b in labels]
    return sorted(labels, key=float)


def series(panel: dict, key: str, method: str, metric: str,
           cfg: dict) -> tuple[list[str], np.ndarray]:
    """The bins of slice `key` and the method's value of `metric` in each (NaN if absent)."""
    order = bin_order(key, list(panel[key]), cfg)
    ys = np.array([panel[key][b].get(method, {}).get(metric, np.nan) for b in order],
                  dtype=float)
    return order, ys


def panel_of(tables: dict, cfg: dict) -> dict[str, Any] | None:
    """The main-benchmark panel the figures draw (config `ip.fig_panel`)."""
    c = cfg["ip"]
    return tables["panels"].get(c["panels"][c["fig_panel"]])
