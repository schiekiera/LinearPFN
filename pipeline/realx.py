"""Counts of the real_designs sampling pipeline (frame, exclusion log,
selection, cells) from its own finished CSVs — the reproducibility claim
of the design list, quoted without re-running R.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _deliver_min_minority_rows(rx: Path) -> int | None:
    """`CFG$min_minority_rows <- 10L` in 10_deliver_eligible.R: the pool delivery's
    near-constant-column threshold (the stratified draw keeps 00_config.R's 0)."""
    f = rx / "10_deliver_eligible.R"
    if not f.is_file():
        return None
    m = re.search(r"CFG\$min_minority_rows\s*<-\s*(\d+)L?", f.read_text())
    return int(m.group(1)) if m else None


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _pool_block(pool: list[dict[str, str]]) -> dict[str, Any]:
    n = [int(r["n_delivered"]) for r in pool]
    p = [int(r["p_delivered"]) for r in pool]
    maxr = np.asarray([float(r["max_abs_r"]) for r in pool])
    return {"datasets": len(pool), "packages": len({r["package"] for r in pool}),
            "domain": dict(Counter(r["domain"] for r in pool).most_common()),
            "domain_packages": {d: sorted({r["package"] for r in pool if r["domain"] == d})
                                for d in {r["domain"] for r in pool}},
            "domains": len({r["domain"] for r in pool}),
            "package_counts": dict(Counter(r["package"] for r in pool).most_common()),
            "row_unit": dict(Counter(r["row_unit"] for r in pool)),
            "row_subsampled": sum(1 for r in pool if r["row_subsampled"].upper() == "TRUE"),
            "col_subsampled": sum(1 for r in pool if r["col_subsampled"].upper() == "TRUE"),
            "p_le_5": sum(1 for x in p if x <= 5),
            "selected": sum(1 for r in pool if r.get("selected", "").upper() == "TRUE"),
            "n_range": [min(n), max(n)], "p_range": [min(p), max(p)],
            "n_median": float(np.median(n)), "p_median": float(np.median(p)),
            "max_abs_r": {"median": float(np.median(maxr)), "max": float(maxr.max()),
                          "min": float(maxr.min())},
            "cells": dict(Counter(r["cell"] for r in pool))}


def numbers(rx_dir: Path | str) -> dict[str, Any]:
    rx = Path(rx_dir)
    index = _rows(rx / "work" / "00_datasets.csv")
    frame = _rows(rx / "work" / "02_frame.csv")
    excl = _rows(rx / "exclusions.csv")
    sel = _rows(rx / "selected_datasets.csv")
    cells = _rows(rx / "work" / "07_cells.csv")
    manifest = _rows(rx / "MANIFEST.csv")
    included_pkgs = {r["Package"] for r in index if r.get("include", "") == "include"}
    maxr = np.asarray([float(r["max_abs_r"]) for r in sel])
    by_stage = Counter(r["stage"] for r in excl)
    by_reason = Counter(r["reason"] for r in excl)
    cfg_text = (rx / "00_config.R").read_text()
    seed = re.search(r"seed\s*=\s*(\d+)L?", cfg_text)
    max_cor = re.search(r"max_abs_cor\s*=\s*([\d.]+)", cfg_text)

    def cfg_num(key):
        m = re.search(key + r"\s*=\s*([\d.]+)L?", cfg_text)
        return float(m.group(1)) if m else None

    sess = (rx / "MANIFEST_sessionInfo.txt").read_text().splitlines()
    idx_meta = {r["key"]: r["value"] for r in manifest if r["section"] == "index"}
    pool = _rows(rx / "eligible_designs.csv") if (rx / "eligible_designs.csv").is_file() else []
    out = {
        # every eligible design delivered through the same cleaning + seeded subsampling
        # (10_deliver_eligible.R): the main benchmark's design pool
        "pool": _pool_block(pool) if pool else None,
        "index": {"datasets": len(index), "packages": len({r["Package"] for r in index}),
                  "included_packages": len(included_pkgs),
                  "datasets_in_included_packages": sum(1 for r in index
                                                       if r["Package"] in included_pkgs),
                  "meta": idx_meta},
        "frame": len(frame),
        "downloaded": sum(1 for r in manifest if r["section"] == "raw_csv"),
        "eligible": int(sum(int(c["n_eligible"]) for c in cells)),
        "selected": {"datasets": len(sel), "packages": len({r["package"] for r in sel}),
                     "row_unit": dict(Counter(r["row_unit"] for r in sel)),
                     "row_subsampled": sum(1 for r in sel if r["row_subsampled"].upper() == "TRUE"),
                     "col_subsampled": sum(1 for r in sel if r["col_subsampled"].upper() == "TRUE"),
                     "census_cells": sum(1 for c in cells if c["census"].upper() == "TRUE"),
                     "rejected": int(sum(int(c["n_rejected"]) for c in cells)),
                     "not_drawn": int(sum(int(c["n_not_drawn"]) for c in cells)),
                     "p_le_5": sum(1 for r in sel if int(r["p_delivered"]) <= 5),
                     "max_abs_r": {"median": float(np.median(maxr)), "max": float(maxr.max()),
                                   "min": float(maxr.min())},
                     "n_range": [min(int(r["n_delivered"]) for r in sel),
                                 max(int(r["n_delivered"]) for r in sel)],
                     "p_range": [min(int(r["p_delivered"]) for r in sel),
                                 max(int(r["p_delivered"]) for r in sel)],
                     "n_median": float(np.median([int(r["n_delivered"]) for r in sel])),
                     "p_median": float(np.median([int(r["p_delivered"]) for r in sel])),
                     # hand labels of the source packages (inputs/package_labels.csv)
                     "domain": dict(Counter(r["domain"] for r in sel).most_common()),
                     "domain_packages": {d: sorted({r["package"] for r in sel if r["domain"] == d})
                                         for d in {r["domain"] for r in sel}},
                     "domains": len({r["domain"] for r in sel})},
        "exclusions": {"total": len(excl), "by_stage": dict(by_stage),
                       "by_reason": dict(by_reason)},
        "cells": {c["cell"]: {"eligible": int(c["n_eligible"]), "selected": int(c["n_admitted"]),
                              "census": c["census"].upper() == "TRUE"} for c in cells},
        "seed": int(seed.group(1)) if seed else None,
        "max_abs_cor": float(max_cor.group(1)) if max_cor else None,
        "min_n": cfg_num("min_n"), "min_p": cfg_num("min_p"),
        "max_row_loss_frac": cfg_num("max_row_loss_frac"),
        "max_col_na_frac": cfg_num("max_col_na_frac"),
        # the pool's near-constant rule is set by the delivery script, not 00_config.R
        "pool_min_minority_rows": _deliver_min_minority_rows(rx),
        "r_version": sess[0].strip() if sess else None,
        "clean_files": sum(1 for r in manifest if r["section"] == "clean"),
    }
    return out
