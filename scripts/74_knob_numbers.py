"""Stage 74 — the MISSPECIFICATION panel's metric tables from the cached npz
rows (no refits): the control panel and the three one-knob panels
(bench.knobs: coef / int / noise), every knob paired by dataset with its
control. The knob panels are the main benchmark's datasets with one knob
turned (stage 16, results_knob_main, xc / xi / xn) and the control is the
main benchmark's own rows (results_main_strong, mx).

    python scripts/74_knob_numbers.py

Reads <knobs.results_key>/<method>/ip{xc,xi,xn}_*.npz and the control panel
from <knobs.control_results_key> (default: the same directory) for the main
benchmark's method set (ip.baselines + ip.model_row, the exact posterior as
reference) with the same coverage / common-support rules as stage 70. Per
panel: pooled metrics with ECE, realized-density / realized-R2 / p-bin /
n-bin slices, per-dataset distributions and paired comparisons of every
method with the model row (pipeline.ip), the exact posterior's subset and
the null datasets; ECE only for knobs.ece_methods (posterior probabilities).
Per knob and method: the PAIRED difference to the control (knob minus
control on the same dataset index; pipeline.knobs).
Pending panels and methods are recorded, never substituted.
Output: reports/paper/knob_tables.json.
"""

from __future__ import annotations

from _bootstrap import CFG, CONFIG, NUM, P, paths, store
from pipeline import ip, knobs


def _scored(by_m: dict[str, list[dict]], model_row: str, reference: list[str],
            min_cov: float, common: bool) -> tuple[dict[str, list[dict]], dict, list[str]]:
    full = len(by_m.get(model_row, []))
    keep, cov, incomplete = {}, {}, []
    for m, r in by_m.items():
        c = len(r) / full if full else 0.0
        cov[m] = {"fitted": len(r), "of": full, "missing": full - len(r), "coverage": c}
        if m in reference or c >= min_cov:
            keep[m] = r
        else:
            incomplete.append(m)
    support = {}
    if common and keep:
        scored = [m for m in keep if m not in reference]
        sets = [{r["i"] for r in keep[m]} for m in scored]
        inter = set.intersection(*sets) if sets else set()
        allk = {r["i"] for r in keep.get(model_row, [])}
        support = {"n": len(inter), "excluded": len(allk - inter)}
        keep = knobs.restrict(keep, inter)
    return keep, {"coverage": cov, "common_support": support}, sorted(incomplete)


def main() -> int:
    c, k = CFG["ip"], CFG["knobs"]
    model = c["model_row"]
    reference = [r for r in c["reference_rows"] if r == "exact"]
    wanted = [*c["baselines"], model, *reference]
    p_bins, n_bins = CFG["benchmark"]["p_bins"], CFG["benchmark"]["n_bins"]
    rdir = P[k["results_key"]]
    cdir = P[k.get("control_results_key", k["results_key"])]
    present = sorted(set(ip.methods_present(rdir)) & set(ip.methods_present(cdir)))
    names = [k["control"], *k["knobs"]]
    methods = [m for m in wanted if m in present]
    raw = ip.load_all(rdir, methods, panels=tuple(k["panels"][n] for n in k["knobs"]))
    raw.update(ip.load_all(cdir, methods, panels=(k["panels"][k["control"]],)))

    panels, rows_of, pending = {}, {}, []
    for name in names:
        panel = k["panels"][name]
        by_m = raw.get(panel, {})
        if model not in by_m:
            pending.append(name)
            panels[name] = {"panel": panel, "pending": True, "present": sorted(by_m)}
            continue
        rows, cov, incomplete = _scored(by_m, model, reference, float(k.get("min_coverage", 1.0)),
                                        bool(k.get("common_support", False)))
        rows_of[name] = rows
        table = ip.panel_table(rows, p_bins, n_bins)
        table["pooled"] = {m: knobs.with_ece(r) for m, r in rows.items()}
        panels[name] = {
            "panel": panel, "pending": False,
            "methods": [m for m in wanted if m in rows and m not in reference],
            "reference_rows": [m for m in reference if m in rows],
            "pending_methods": sorted((set(wanted) - set(present) - set(reference))
                                      | set(incomplete)),
            "counts": {m: len(r) for m, r in by_m.items()}, **cov, **table,
            "ece": {m: knobs.ece(r) for m, r in rows.items()},
            **ip.distributions(rows, model),
            "exact_subset": knobs.exact_subset(rows),
            "null": {m: knobs.null_datasets(r) for m, r in rows.items()},
            "interactions": ip.interaction_counts(rows.get(model, [])),
        }
    # ECE only where the returned probability is a posterior inclusion probability
    ece_ok = set(k.get("ece_methods", []))
    for pd in panels.values():
        if pd.get("pending"):
            continue
        for m in list(pd["ece"]):
            if m not in ece_ok:
                pd["ece"][m] = None
                pd["pooled"][m]["ece"] = None
                if m in pd["exact_subset"]:
                    pd["exact_subset"][m]["ece"] = None
    ctrl = rows_of.get(k["control"], {})
    for name in k["knobs"]:
        if name in rows_of and ctrl:
            panels[name]["vs_control"] = {
                m: knobs.paired_control(rows_of[name][m], ctrl[m], with_ece=m in ece_ok)
                for m in rows_of[name] if m in ctrl}
    payload = {"model_row": model, "control": k["control"], "knobs": k["knobs"],
               "panels": panels, "pending": pending,
               "results_dir": paths.rel(rdir), "stamp": store.dir_stamp(rdir),
               "control_results_dir": paths.rel(cdir), "control_stamp": store.dir_stamp(cdir),
               "definitions": {kn: __import__("bench.knobs", fromlist=["KNOBS"]).KNOBS[kn]
                               for kn in k["knobs"]},
               "n_datasets": k["n_datasets"]}
    out = store.write_json(NUM / "knob_tables.json", payload, [CONFIG],
                           stage="74_knob_numbers")
    print(f"panels {[n for n in names if n not in pending]}; pending {pending}; "
          f"methods {sorted(present)}\n-> {paths.rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
