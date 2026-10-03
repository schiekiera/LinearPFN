"""Prior-calibration numbers from finished diagnostics: the preflight JSON
(linearpfn/preflight.py, 10k prior draws) and the quadrature convergence
JSON (scripts/38_compute_quadrature.py). Nothing is redrawn here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Meta-analytic effect-size norms the prior was calibrated against
# (Gignac & Szodorai 2016: |r| quartiles .11 / .19 / .29 for individual
# differences research); kept here so the paper quotes them from one place.
GIGNAC_SZODORAI = {"p25": 0.11, "p50": 0.19, "p75": 0.29}


def preflight_numbers(path: Path | str) -> dict[str, Any]:
    d = json.loads(Path(path).read_text())
    checks = d.get("checks", [])
    ek = d.get("ek_by_p", {})
    zs = {p: abs(float(v["z"])) for p, v in ek.items() if "z" in v}
    worst_p = max(zs, key=zs.get) if zs else None
    r2 = d.get("r2", {})
    out = {
        "n_datasets": d.get("n_datasets"), "seed": d.get("seed"),
        "effect_size": {
            "mains": d["effect_size"]["mains"], "interactions": d["effect_size"]["interactions"],
            "n_mains": d["effect_size"].get("n_mains"),
            "n_interactions": d["effect_size"].get("n_interactions"),
            "norms": GIGNAC_SZODORAI},
        "null_fraction": {"pooled": d["null"]["pooled"],
                          "by_p": {str(k): v for k, v in d["null"].get("by_p", {}).items()}},
        "y_escape": d.get("y_escape"),
        "r2": {"overall": r2.get("overall"), "nonnull": r2.get("nonnull"),
               "by_mode": r2.get("by_mode"), "by_k": r2.get("by_k")},
        "ek": {"max_abs_z": max(zs.values()) if zs else None, "worst_p": worst_p,
               "by_p": {p: v for p, v in ek.items()}},
        "corr": {k: d.get("corr", {}).get(k) for k in
                 ("pooled_abs", "share_ds_max_gt_090", "share_ds_max_gt_095", "tail_pos_frac",
                  "tail_neg_frac", "p5_tail")},
        "c_pcts": d.get("c_pcts"), "rho_pcts": d.get("rho_pcts"),
        "int_share": d.get("int_share"), "factor_share": d.get("factor_share"),
        "checks": {"passed": sum(1 for c in checks if c.get("ok")), "total": len(checks),
                   "failed": [c["name"] for c in checks if not c.get("ok")],
                   "all": checks},
    }
    return out


def quadrature_numbers(path: Path | str) -> dict[str, Any]:
    """Per panel: the worst cell at the largest tested grid (and at every
    grid), so the paper can state 'max |dPIP| <= X at the chosen nodes'."""
    d = json.loads(Path(path).read_text())
    out: dict[str, Any] = {"n_datasets": d.get("n_datasets"),
                           "config_is_default": d.get("config_is_default"), "panels": {}}
    for panel in ("interactions", "mains_only"):
        rows = d.get(panel, [])
        if not rows:
            continue
        by_grid: dict[str, dict[str, Any]] = {}
        for r in rows:
            g = "x".join(str(x) for x in r["grid"])
            e = by_grid.setdefault(g, {"max_dpip": 0.0, "max_dlogev": 0.0, "cells": 0,
                                       "worst_cell": None})
            e["cells"] += 1
            if r["max_dpip"] >= e["max_dpip"]:
                e["max_dpip"] = r["max_dpip"]
                e["worst_cell"] = {"n": r["n"], "p": r["p"], "models": r["models"]}
            e["max_dlogev"] = max(e["max_dlogev"], r["max_dlogev"])
        finest = max(by_grid, key=lambda g: tuple(int(x) for x in g.split("x")))
        # the c axis alone: max over the rho node counts of each c count (with rho
        # fixed the rho axis is inert and every rho count gives the same row)
        by_c: dict[str, dict[str, Any]] = {}
        for r in rows:
            e = by_c.setdefault(str(r["grid"][0]), {"max_dpip": 0.0, "max_dlogev": 0.0,
                                                    "worst_cell": None})
            if r["max_dpip"] >= e["max_dpip"]:
                e["max_dpip"] = r["max_dpip"]
                e["worst_cell"] = {"n": r["n"], "p": r["p"], "models": r["models"]}
            e["max_dlogev"] = max(e["max_dlogev"], r["max_dlogev"])
        out["panels"][panel] = {"grids": by_grid, "largest_tested": finest,
                                **by_grid[finest],
                                "by_c": dict(sorted(by_c.items(), key=lambda kv: int(kv[0]))),
                                "cells_n": sorted({r["n"] for r in rows}),
                                "cells_p": sorted({r["p"] for r in rows}),
                                "truth": (d.get("truth") or {}).get(panel),
                                "fit_seconds_default_grid": max(
                                    r.get("fit_seconds_default_grid", 0.0) for r in rows)}
    return out
