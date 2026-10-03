"""Stage 70 — every metric table of the MAIN benchmark from the cached npz
rows (no refits).

    python scripts/70_ip_numbers.py

Reads <paths[ip.results_key]>/<method>/ip<panel>_*.npz (the main benchmark,
panel mx) for the config's method set (ip.baselines + ip.model_row + ip.extra_rows). A method is
scored on a panel when its rows cover at least ip.min_coverage of the model
row's datasets (the reference rows exact and mc3 are exempt: they cover the
p <= 5 subset by construction); its fitted / missing counts are recorded
(`coverage`) so the paper can state where a baseline's implementation
refused a dataset. Below the threshold a method is pending, never
substituted or drawn. With ip.common_support every scored method (and the
reference rows) is restricted to the datasets ALL scored methods fitted,
so the rows are equal and every comparison is fully paired; the excluded
datasets and designs are recorded (`common_support`). Per panel: pooled, by p bin, by n bin, by
realized-density cell and by realized-R2 cell (pipeline.scores.metrics via
pipeline.ip), plus per-dataset distributions (mean, quartiles, p90) of
AUC/ds, F1/ds, RMSE and fit time, and paired comparisons of every method
with the model row on the datasets both score (bootstrap CI, win share,
Wilcoxon). With ip.hard_zeroed_row the model row's own fits are also scored
with their coefficients zeroed outside the MPM selection (derived, no refit;
pipeline.ip.hard_zeroed), and that row is paired on RMSE against every
baseline (`paired_hard_zeroed`); it is recorded in `derived_rows` and kept out
of `methods`. Also recorded: how the panel was drawn (`construction`: stage 15's
bin width and per-cell acceptance and design counts) and the fixed settings of
the baselines the appendix states (`method_settings`). Output:
reports/paper/ip_tables.json.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import re

from _bootstrap import CFG, CONFIG, NUM, P, paths, store
from pipeline import ip

GEN_SCRIPT = paths.ROOT / "scripts" / "15_gen_main_panel.py"
SUSIE_R = paths.ROOT / "bench" / "rscripts" / "susie.R"


def construction(results) -> dict | None:
    """How the panel was drawn (stage 15): the bin width of its acceptance grid
    (the generator's own constant) and, per cell, the draws, the acceptance rate
    and the number of distinct designs it holds (truth/_grid/cell_*.json)."""
    cells = sorted((results / "truth" / "_grid").glob("cell_*.json"))
    if not cells:
        return None
    spec = importlib.util.spec_from_file_location("gen_main_panel", GEN_SCRIPT)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    rec = [json.loads(f.read_text()) for f in cells]
    rates = [r["rate"] for r in rec]
    used = [r["designs_used"] for r in rec]
    return {"bin_width": float(gen.BIN), "lower_edges": list(gen.BENCH), "cells": len(rec),
            "pool": rec[0].get("pool"), "draws_total": int(sum(r["draws"] for r in rec)),
            "accept_min": min(rates), "accept_max": max(rates),
            "designs_per_cell_min": min(used), "designs_per_cell_max": max(used)}


def method_settings() -> dict:
    """Fixed settings of the baselines that the appendix states (the method
    modules are the single source), and the bootstrap size of every paired CI."""
    from bench.methods import stability

    m = re.search(r"susie\([^)]*\bL\s*=\s*(\d+)", SUSIE_R.read_text())
    return {"stability_pairs": int(stability.B_PAIRS),
            "stability_threshold": float(stability.THRESHOLD),
            "susie_L": int(m.group(1)) if m else None,
            "bootstrap_reps": int(inspect.signature(ip.paired).parameters["boots"].default)}


def main() -> int:
    c = CFG["ip"]
    results = P[c.get("results_key", "results_inprior_strong")]
    wanted = [*c["baselines"], c["model_row"], *c["extra_rows"], *c["reference_rows"]]
    present = ip.methods_present(results)
    raw = ip.load_all(results, [m for m in wanted if m in present])
    # derived hard-zeroed read-out of the model row (same fits, no refit); added before the
    # coverage and common-support steps so it is restricted exactly like every scored row
    hz = c.get("hard_zeroed_row")
    if hz:
        for by_m in raw.values():
            if c["model_row"] in by_m:
                by_m[hz] = ip.hard_zeroed(by_m[c["model_row"]])
    counts = ip.counts(raw)
    incomplete: dict[str, list[str]] = {}
    rows: dict[str, dict[str, list[dict]]] = {}
    coverage: dict[str, dict[str, dict]] = {}
    min_cov = float(c.get("min_coverage", 1.0))
    for panel, by_m in raw.items():
        full = len(by_m.get(c["model_row"], []))
        keep = {}
        for m, r in by_m.items():
            cov = len(r) / full if full else 0.0
            coverage.setdefault(panel, {})[m] = {"fitted": len(r), "of": full,
                                                 "missing": full - len(r), "coverage": cov}
            # a baseline whose implementation refuses a few datasets (hierNet's R code
            # errors inside a CV fold on 113 of the 10k main-panel rows) is scored on the
            # datasets it fitted, with its coverage recorded; below min_coverage it is
            # pending (still running, or broken)
            if m in c["reference_rows"] or cov >= min_cov:
                keep[m] = r
            else:
                incomplete.setdefault(m, []).append(panel)
        rows[panel] = keep
    # COMMON SUPPORT: every method is scored on the datasets that ALL
    # scored methods fitted, so a baseline's refusals (hierNet: 113 rows on three
    # designs with a near-constant binary column) leave the comparison for everyone
    # and the rows stay equal; the exclusions are recorded. Reference rows are
    # restricted to the same support.
    support: dict[str, dict] = {}
    if c.get("common_support", False):
        for panel, keep in rows.items():
            scored = [m for m in keep if m not in c["reference_rows"]]
            key_sets = [{r["i"] for r in keep[m]} for m in scored]
            common = set.intersection(*key_sets) if key_sets else set()
            full = {r["i"] for r in keep.get(c["model_row"], [])}
            dropped = sorted(full - common)
            designs = sorted({k.split("_f")[0].split("_", 1)[1] for k in dropped})
            support[panel] = {"n": len(common), "excluded": len(dropped),
                              "excluded_designs": designs,
                              "excluded_by_method": {m: len(full - {r["i"] for r in keep[m]})
                                                     for m in scored}}
            rows[panel] = {m: [r for r in rs if r["i"] in common] for m, rs in keep.items()}
    panels = {}
    for fam, panel in c["panels"].items():
        if panel in rows:
            panels[panel] = {"family": fam, **ip.panel_table(
                rows[panel], CFG["benchmark"]["p_bins"], CFG["benchmark"]["n_bins"]),
                **ip.distributions(rows[panel], c["model_row"]),
                "interactions": ip.interaction_counts(rows[panel].get(c["model_row"], []))}
            if "mc3" in rows[panel]:
                panels[panel]["sampler"] = ip.sampler_summary(rows[panel]["mc3"], results)
            if hz and hz in rows[panel]:
                hz_scores = ip.dataset_scores(rows[panel][hz])
                panels[panel]["paired_hard_zeroed"] = {
                    m: {"rmse": ip.paired(hz_scores, ip.dataset_scores(rows[panel][m]), "rmse")}
                    for m in c["baselines"] if m in rows[panel]}
            # row unit of the design (cross-section / panel, from eligible_designs.csv): the
            # simulated y is independent across rows on every design, so this slice asks
            # whether a panel's repeated units (an X property) change any standing
            unit = ip.design_row_units(P["real_data_x"] / "eligible_designs.csv")
            if unit:
                panels[panel]["row_unit"] = ip.row_unit_slice(rows[panel], c["model_row"], unit)
    methods = [m for m in wanted if m in present and m not in incomplete
               and m not in c["reference_rows"]]
    derived = ({hz: {"from": c["model_row"], "rule": "dense coefficients zeroed outside the "
                     "MPM selection, intercept kept (pipeline.ip.hard_zeroed)"}} if hz else {})
    pending = sorted((set(wanted) - set(present) - set(c["reference_rows"])) | set(incomplete))
    payload = {"results_dir": paths.rel(results), "stamp": store.dir_stamp(results),
               "methods": methods, "pending_methods": pending, "incomplete": incomplete,
               "reference_rows": [m for m in c["reference_rows"] if m in present],
               "counts": counts, "coverage": coverage, "common_support": support,
               "derived_rows": derived, "panels": panels,
               "construction": construction(results), "method_settings": method_settings(),
               "model_row": c["model_row"], "reference_row": CFG["benchmark"]["reference_row"]}
    inputs = [CONFIG, GEN_SCRIPT, SUSIE_R,
              paths.ROOT / "bench" / "methods" / "stability.py",
              *sorted((results / "truth" / "_grid").glob("cell_*.json"))]
    out = store.write_json(NUM / "ip_tables.json", payload, inputs, stage="70_ip_numbers")
    print(f"panels {sorted(panels)}; methods {payload['methods']}; pending {pending}; "
          f"counts {payload['counts']}\n-> {paths.rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
