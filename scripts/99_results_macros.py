"""Stage 99 — results_macros.tex: every number the manuscript may quote.

    python scripts/99_results_macros.py

Reads only reports/paper/*.json (finished, provenance-stamped numbers) and
emits \\newcommand lines through pipeline.macros; rounding, signs, percent
signs and scientific notation are set here (pipeline.style), never in
LaTeX. Names are letters only: metric + method + panel tokens, e.g.
\\ipAucMeanOursRealX, \\gateAucGap, \\rxSelectedN, \\priorNullFrac.
Quantities whose inputs are still pending are simply absent (an undefined
control sequence, not a silent placeholder). `Ours` marks the model row.
Outputs: paper/results_macros.tex + results_macros.json.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from _bootstrap import CFG, CONFIG, NUM, PROV, P, paths, store
from pipeline import ip, style
from pipeline.macros import Registry, name, token
from pipeline.style import (
    fmt_auc,
    fmt_corr,
    fmt_ece,
    fmt_fixed,
    fmt_int,
    fmt_p,
    fmt_pct,
    fmt_rmse,
    fmt_sci,
    fmt_seconds,
    fmt_signed,
    yes_no,
)

ST = style.Style.from_config(CFG)
REG = Registry()
USED: list[Path] = [CONFIG]
MODEL = CFG["benchmark"]["model_row"]


def load(fname: str) -> dict[str, Any] | None:
    p = NUM / fname
    USED.append(p)  # recorded present or MISSING: a number file landing later = stale
    if p.is_file():
        return store.read_json(p)
    return None


def add(parts: tuple, value, fmt, source: str) -> None:
    """Register `fmt(value, style)` under name(*parts); None/NaN emit nothing
    (an undefined control sequence in LaTeX, never a silent placeholder)."""
    if value is None:
        return
    if isinstance(value, float) and value != value:  # NaN
        return
    REG.add(name(*parts), fmt(value, ST), value, source)


# every formatter here takes (value, style)
F0 = lambda v, s: fmt_fixed(v, 0, s)  # noqa: E731
F1 = lambda v, s: fmt_fixed(v, 1, s)  # noqa: E731
F2 = lambda v, s: fmt_fixed(v, 2, s)  # noqa: E731
F3 = lambda v, s: fmt_fixed(v, 3, s)  # noqa: E731
F4 = lambda v, s: fmt_fixed(v, 4, s)  # noqa: E731
S3 = lambda v, s: fmt_signed(v, 3, s)  # noqa: E731
S4 = lambda v, s: fmt_signed(v, 4, s)  # noqa: E731
SR = lambda v, s: fmt_signed(v, s.digits["rmse"], s)  # noqa: E731  (signed, style digits)
SCI = lambda v, s: fmt_sci(v)  # noqa: E731
GEN = lambda v, s: f"{v:g}"  # noqa: E731
METRICS = (("auc_macro", "Auc", fmt_auc), ("auc_pooled", "AucPooled", fmt_auc),
           ("pr_auc_pooled", "PrAuc", fmt_auc), ("f1_macro", "FOne", fmt_auc),
           ("f1_pooled", "FOnePooled", fmt_auc), ("precision_macro", "Prec", fmt_auc),
           ("recall_macro", "Rec", fmt_auc), ("rmse_macro", "Rmse", fmt_rmse), ("n", "N", fmt_int))


def method_tokens(m: str) -> list[str]:
    toks = [token(m)]
    if m == MODEL:
        toks.append("Ours")
    return toks


def emit_metrics(prefix: str, met: dict, suffix: tuple, src: str) -> None:
    for key, tok, fmt in METRICS:
        add((prefix, tok, *suffix), met.get(key), fmt, src)
    fs = met.get("fit_seconds")
    if fs:
        add((prefix, "FitSec", *suffix), fs["median"], fmt_seconds, src)
        add((prefix, "FitSecMean", *suffix), fs["mean"], fmt_seconds, src)


SLICE_TOKENS = {"p_bin": "P", "n_bin": "N", "density": "Density", "signal": "Signal"}


def bin_label(b: str) -> str:
    """A slice label for running text: '200-499' -> '200--499', '1000-1999' ->
    '1,000--1,999' (the text's integer style); density and R2 cells unchanged."""
    if "-" in b:
        lo, hi = b.split("-")
        return f"{fmt_int(int(lo), ST)}--{fmt_int(int(hi), ST)}"
    return b


IP_STATS = (("mean", "Mean"), ("q25", "QOne"), ("q50", "Med"), ("q75", "QThree"),
            ("p90", "PNinety"))
IP_METRICS = (("auc", "Auc", fmt_auc), ("f1", "FOne", fmt_auc), ("rmse", "Rmse", fmt_rmse))


def ip_tables(doc) -> None:
    r"""MAIN benchmark: \ip<Metric><Stat><Method><Fam>,
    \ipFit<Stat><Method><Fam>, paired \ipDiff / \ipDiffLo / \ipDiffHi / \ipWin /
    \ipP<Metric><Method><Fam>, \ipFitRatio<Method><Fam>; counts."""
    src = "ip_tables.json"
    fams = {v: ("RealX" if k == "real_x" else "PriorX") for k, v in CFG["ip"]["panels"].items()}
    model = doc["model_row"]
    for panel, t in doc["panels"].items():
        fam = fams.get(panel, token(panel))
        for m, met in t["pooled"].items():
            toks = [token(m)] + (["Ours"] if m == model else [])
            for mt in toks:
                emit_metrics("ip", met, (mt, fam), src)
                add(("ipDatasets", mt, fam), met["n"], fmt_int, src)
        for m, by_k in t["summary"].items():
            toks = [token(m)] + (["Ours"] if m == model else [])
            for key, ktok, fmt in IP_METRICS:
                sm = by_k.get(key)
                if not sm:
                    continue
                for mt in toks:
                    for sk, stok in IP_STATS:
                        add(("ip", ktok, stok, mt, fam), sm[sk], fmt, src)
                    add(("ip", ktok, "N", mt, fam), sm["n"], fmt_int, src)
        for m, sm in t["fit_time"].items():
            if not sm:
                continue
            toks = [token(m)] + (["Ours"] if m == model else [])
            for mt in toks:
                for sk, stok in IP_STATS:
                    add(("ipFit", stok, mt, fam), sm[sk], fmt_seconds, src)
                add(("ipFit", "N", mt, fam), sm["n"], fmt_int, src)
        for m, by_k in t["paired"].items():
            mt = token(m)
            for key, ktok, _fmt in IP_METRICS:
                pr = by_k.get(key)
                if not pr:
                    continue
                # every RMSE in the paper has the style's RMSE digits
                sd = SR if key == "rmse" else S3
                add(("ipDiff", ktok, mt, fam), pr["mean_diff"], sd, src)
                add(("ipDiffLo", ktok, mt, fam), pr["ci_lo"], sd, src)
                add(("ipDiffHi", ktok, mt, fam), pr["ci_hi"], sd, src)
                add(("ipWin", ktok, mt, fam), pr["win_share"], fmt_pct, src)
                add(("ipP", ktok, mt, fam), pr["wilcoxon_p"], fmt_p, src)
                add(("ipPairedN", ktok, mt, fam), pr["n"], fmt_int, src)
            ft = by_k.get("fit_time")
            if ft and ft.get("ratio_other_over_model") is not None:
                add(("ipFitRatio", mt, fam), ft["ratio_other_over_model"], F1, src)
        # hard-zeroed model row (stage 70 `paired_hard_zeroed`): its RMSE minus
        # each baseline's, \ipHz{Diff,DiffLo,DiffHi,Win,P,PairedN}Rmse<Method><Fam>
        for m, by_k in t.get("paired_hard_zeroed", {}).items():
            pr = by_k.get("rmse")
            if not pr:
                continue
            mt = token(m)
            add(("ipHzDiff", "Rmse", mt, fam), pr["mean_diff"], SR, src)
            add(("ipHzDiffLo", "Rmse", mt, fam), pr["ci_lo"], SR, src)
            add(("ipHzDiffHi", "Rmse", mt, fam), pr["ci_hi"], SR, src)
            add(("ipHzWin", "Rmse", mt, fam), pr["win_share"], fmt_pct, src)
            add(("ipHzP", "Rmse", mt, fam), pr["wilcoxon_p"], fmt_p, src)
            add(("ipHzPairedN", "Rmse", mt, fam), pr["n"], fmt_int, src)
        ic = t.get("interactions")
        if ic:
            add(("ipIntCountMedian", fam), ic["median"], GEN, src)
            add(("ipIntCountMean", fam), ic["mean"], F2, src)
            add(("ipIntCountMax", fam), ic["max"], fmt_int, src)
            add(("ipIntNoneShare", fam), ic["share_none"], fmt_pct, src)
        cs = doc.get("common_support", {}).get(panel)
        if cs:
            add(("ipSupportN", fam), cs["n"], fmt_int, src)
            add(("ipSupportExcluded", fam), cs["excluded"], fmt_int, src)
            add(("ipSupportFull", fam), cs["n"] + cs["excluded"], fmt_int, src)
            add(("ipSupportExcludedDesigns", fam), len(cs["excluded_designs"]), fmt_int, src)
            REG.add(name("ipSupportExcludedDesignList", fam),
                    ", ".join(style.tex_escape(d) for d in cs["excluded_designs"]),
                    cs["excluded_designs"], src)
        for m, cv in doc.get("coverage", {}).get(panel, {}).items():
            mt = token(m)
            add(("ipFitted", mt, fam), cv["fitted"], fmt_int, src)
            add(("ipMissing", mt, fam), cv["missing"], fmt_int, src)
            add(("ipCoveragePct", mt, fam), cv["coverage"], fmt_pct, src)
        dens, sig = sorted(t["density"], key=float), sorted(t["signal"], key=float)
        add(("ipLevels", fam), len(dens), fmt_int, src)
        add(("ipCells", fam), len(dens) * len(sig), fmt_int, src)
        add(("ipLevelLo", fam), float(dens[0]), GEN, src)
        add(("ipLevelHi", fam), float(dens[-1]), GEN, src)
        per_cell = {v[model]["n"] for v in t["density"].values() if model in v}
        if len(per_cell) == 1:
            n_level = per_cell.pop()
            add(("ipPerDensityLevel", fam), n_level, fmt_int, src)
            if n_level % len(sig) == 0:
                add(("ipPerCell", fam), n_level // len(sig), fmt_int, src)
        for u, blk in t.get("row_unit", {}).items():
            ut = token(u)
            add(("ipUnitN", ut, fam), blk["n"], fmt_int, src)
            add(("ipUnitDesigns", ut, fam), blk["designs"], fmt_int, src)
            for m, by_k in blk["mean"].items():
                toks = [token(m)] + (["Ours"] if m == model else [])
                for key, ktok, fmt in IP_METRICS:
                    if by_k.get(key):
                        for mt in toks:
                            add(("ipUnit", ktok, "Mean", mt, ut, fam), by_k[key]["mean"], fmt, src)
            for m, by_k in blk["diff"].items():
                for key, ktok, _fmt in IP_METRICS:
                    if by_k.get(key):
                        add(("ipUnitDiff", ktok, token(m), ut, fam), by_k[key]["mean_diff"],
                            S3, src)
                        add(("ipUnitWin", ktok, token(m), ut, fam), by_k[key]["win_share"],
                            fmt_pct, src)
        sm = t.get("sampler")
        if sm:
            add(("ipMcmcN", fam), sm["n"], fmt_int, src)
            add(("ipMcmcConvPct", fam), sm["converged_share"], fmt_pct, src)
            add(("ipMcmcRhatMed", fam), sm["rhat_max_median"], F3, src)
            add(("ipMcmcEssMed", fam), sm["ess_min_median"], F0, src)
            add(("ipMcmcStepsMed", fam), sm["steps_median"], fmt_int, src)
        # margin over the best baseline of each slice (the Results text on fig:ip-sweeps):
        # \ipMargin<FOne|Auc><P|N|Density|Signal><Lo|Hi|Max|MaxBin|Min|MinBin><Fam>, and
        # \ipMargin<FOne|Auc>Min<Fam> over every slice
        for metric, mtok in (("f1_macro", "FOne"), ("auc_macro", "Auc")):
            margins = ip.slice_margins(t, model, CFG["ip"]["baselines"], metric,
                                       CFG["benchmark"]["p_bins"], CFG["benchmark"]["n_bins"])
            every = [v for rows in margins.values() for _b, v in rows]
            if every:
                add(("ipMargin", mtok, "Min", fam), min(every), S3, src)
            for key, stok in SLICE_TOKENS.items():
                rows = margins.get(key)
                if not rows:
                    continue
                add(("ipMargin", mtok, stok, "Lo", fam), rows[0][1], S3, src)
                add(("ipMargin", mtok, stok, "Hi", fam), rows[-1][1], S3, src)
                for tag, (b, v) in (("Max", max(rows, key=lambda r: r[1])),
                                    ("Min", min(rows, key=lambda r: r[1]))):
                    add(("ipMargin", mtok, stok, tag, fam), v, S3, src)
                    REG.add(name("ipMargin", mtok, stok, tag + "Bin", fam), bin_label(b), b, src)
    add(("ipMethodsN",), len(doc["methods"]), fmt_int, src)
    cons = doc.get("construction")
    if cons:  # stage 15's acceptance grid (main panel = real X)
        add(("ipBinWidthRealX",), cons["bin_width"], GEN, src)
        add(("ipAcceptMinPctRealX",), cons["accept_min"], lambda v, s: fmt_pct(v, s, 3), src)
        add(("ipAcceptMaxPctRealX",), cons["accept_max"], lambda v, s: fmt_pct(v, s, 1), src)
        add(("ipDesignsPerCellMinRealX",), cons["designs_per_cell_min"], fmt_int, src)
        add(("ipDesignsPerCellMaxRealX",), cons["designs_per_cell_max"], fmt_int, src)
        add(("ipDrawsTotalRealX",), cons["draws_total"], fmt_int, src)
    ms = doc.get("method_settings")
    if ms:
        add(("stabPairs",), ms["stability_pairs"], fmt_int, src)
        add(("stabThreshold",), ms["stability_threshold"], GEN, src)
        add(("susieL",), ms["susie_L"], fmt_int, src)
        add(("bootReps",), ms["bootstrap_reps"], fmt_int, src)


def fit_time_gpu(doc) -> None:
    """\\fitSecGpu (median), quartiles / p90 / mean, per p bin and n bin, the GPU
    name, the CPU-over-GPU median ratio on the same datasets, and the MCMC-over-GPU
    ratio of medians on the sampler's datasets (\\fitMcmcOverGpu)."""
    if doc.get("pending"):
        return
    src = "fit_time_gpu.json"
    sm = doc["summary"]
    # seconds round like every other fit time in the paper (style.digits.seconds),
    # so a GPU number in a table and the same macro in a sentence cannot disagree
    add(("fitSecGpu",), sm["median"], fmt_seconds, src)
    for sk, stok in IP_STATS:
        if sk in sm:
            add(("fitSecGpu", stok), sm[sk], fmt_seconds, src)
    add(("fitSecGpuMax",), sm["max"], fmt_seconds, src)
    add(("fitSecGpuN",), doc["n_datasets"], fmt_int, src)
    add(("fitSecGpuLoad",), doc["load_seconds"], fmt_seconds, src)
    if doc["hardware"].get("gpu"):
        REG.add(name("fitGpuName"), style.tex_escape(doc["hardware"]["gpu"]),
                doc["hardware"]["gpu"], src)
    for kind, bins in (("Pbin", CFG["benchmark"]["p_bins"]), ("Nbin", CFG["benchmark"]["n_bins"])):
        table = doc["p_bin"] if kind == "Pbin" else doc["n_bin"]
        for i, b in enumerate(bins):
            if b in table:
                add(("fitSecGpu", f"{kind}{token(str(i + 1))}"), table[b]["median"],
                    fmt_seconds, src)
    if doc.get("cpu_over_gpu_median") is not None:
        add(("fitCpuOverGpu",), doc["cpu_over_gpu_median"], F0, src)
    mg = doc.get("mcmc_over_gpu")
    if mg:  # the sampler against the GPU forward pass on the sampler's datasets
        add(("fitMcmcOverGpu",), mg["ratio"], lambda v, s: fmt_int(round(v), s), src)
        add(("fitMcmcOverGpuN",), mg["n"], fmt_int, src)
        add(("fitMcmcOverGpuSecGpu",), mg["gpu_median"], fmt_seconds, src)
        add(("fitMcmcOverGpuSecMcmc",), mg["mcmc_median"], fmt_seconds, src)


def validation(doc, arms) -> None:
    def emit(v, suffix: tuple, src: str) -> None:
        if not v or v.get("pending"):
            return
        add(("finalStep", *suffix), v.get("step"), fmt_int, src)
        add(("gatesPassed", *suffix), v.get("gates_passed"), fmt_int, src)
        add(("gatesTotal", *suffix), v.get("gates_total"), fmt_int, src)
        for key, g in v["gates"].items():
            add(("gate", token(key), *suffix), g["value"], F4, src)
            REG.add(name("gate", token(key), "Pass", *suffix), yes_no(g["passed"]),
                    g["passed"], src)
        pp = v.get("pooled_p5")
        if pp:
            for k, val in pp.items():
                add(("finalPFive", token(k), *suffix), val, F4, src)
        cr = v.get("coef_recovery")
        if cr:
            add(("finalCoefR", *suffix), cr["r"], F4, src)
            add(("finalCoefRmse", *suffix), cr["rmse"], fmt_rmse, src)
        if v.get("coef_source"):  # which read-out the gate scored (head | probe)
            REG.add(name("finalCoefSource", *suffix), v["coef_source"], v["coef_source"], src)
        crp = v.get("coef_recovery_probe")  # the probe beside a head-scored gate
        if crp:
            add(("finalCoefRProbe", *suffix), crp["r"], F4, src)
            add(("finalCoefRmseProbe", *suffix), crp["rmse"], fmt_rmse, src)
        for lvl, val in (v.get("coverage") or {}).items():
            add(("finalCoverage", token(lvl), *suffix), val, F4, src)
        for pstr, row in (v.get("per_p") or {}).items():
            add(("finalMainsAucP", token(pstr), *suffix), row.get("mains AUC"), F3, src)
            add(("finalIntAucP", token(pstr), *suffix), row.get("int AUC"), F3, src)
        for subset, row in (v.get("selection_vs_exact") or {}).items():
            add(("finalSelR", token(subset), *suffix), row.get("r"), F4, src)
            add(("finalSelRmse", token(subset), *suffix), row.get("RMSE"), F4, src)
            add(("finalSelAucGap", token(subset), *suffix), row.get("AUC gap"), S4, src)
            add(("finalSelResRatio", token(subset), *suffix), row.get("res ratio"), F4, src)
        for subset, row in (v.get("selection_vs_truth") or {}).items():
            add(("finalSelAuc", token(subset), *suffix), row.get("AUC"), F4, src)
            add(("finalSelEce", token(subset), *suffix), row.get("ECE"), F4, src)
        h = v.get("heredity")
        if h:
            add(("finalHeredityHeadPct", *suffix), h["head_all_pct"], F3, src)
            add(("finalHeredityExactPct", *suffix), h["exact_pct"], F3, src)

    crit = (doc or {}).get("criteria")
    if crit:  # thresholds and panel of the criteria (linearpfn.validate, the model's config)
        src = "validation_final.json"
        th = crit["thresholds"]
        add(("critHeadroomMin",), th["headroom_min"], F2, src)
        add(("critCoefRMin",), th["coef_r_min"], F2, src)
        add(("critCoverageTolPp",), th["coverage_tol"] * 100, GEN, src)
        add(("critAucGap",), th["auc_gap"], F2, src)
        add(("critEceGap",), th["ece_gap"], F2, src)
        add(("critResPooled",), th["res_pooled"], F2, src)
        add(("critResSubset",), th["res_subset"], F2, src)
        add(("critHeadroomFlagNats",), crit["headroom_flag_nats"], F2, src)
        REG.add(name("critCoverageLevels"),
                style.fmt_list(crit["coverage_levels"], str) + r"\,\%",
                crit["coverage_levels"], src)
        pn = crit["panel"]
        REG.add(name("valNValues"), style.fmt_list(pn["n_values"], lambda x: fmt_int(x, ST)),
                pn["n_values"], src)
        REG.add(name("valPValues"), style.fmt_list(pn["p_values"], lambda x: fmt_int(x, ST)),
                pn["p_values"], src)
        add(("valDatasetsPerCell",), pn["datasets_per_cell"], fmt_int, src)
        add(("valNQuery",), pn["n_query"], fmt_int, src)
        pr = crit["probe"]
        add(("probePointsPerEffect",), pr["points_per_effect"], fmt_int, src)
        add(("probeJitterSd",), pr["jitter_sd"], GEN, src)
        add(("probeRidge",), pr["ridge_lambda"], SCI, src)
    if doc:
        emit(doc.get("final"), (), "validation_final.json")
        emit(doc.get("final_exact_panel"), ("ExactPanel",), "validation_final.json")
    if arms:
        for arm, v in arms["arms"].items():
            emit(v, ("Arm", token(arm)), "validation_arms.json")


def prior(doc) -> None:
    src = "prior.json"
    d, prefix = doc.get("mixture"), "prior"  # the paper prior's preflight
    if d:
        es = d["effect_size"]
        for q in ("p5", "p25", "p50", "p75", "p95"):
            add((prefix, "Mains", token(q)), es["mains"][q], F3, src)
            add((prefix, "Int", token(q)), es["interactions"][q], F3, src)
        for q, v in es["norms"].items():
            add((prefix, "Norm", token(q)), v, F2, src)
        add((prefix, "NullFrac"), d["null_fraction"]["pooled"], F3, src)
        add((prefix, "NullFracPTwo"), d["null_fraction"]["by_p"].get("2"), F3, src)
        ye = d.get("y_escape") or {}
        add((prefix, "YEscape"), ye.get("pooled_row_frac"), SCI, src)
        add((prefix, "YEscapeShareDs"), ye.get("share_ds_any"), fmt_pct, src)
        add((prefix, "ChecksPassed"), d["checks"]["passed"], fmt_int, src)
        add((prefix, "ChecksTotal"), d["checks"]["total"], fmt_int, src)
        add((prefix, "EkMaxZ"), d["ek"]["max_abs_z"], F2, src)
        add((prefix, "NDraws"), d["n_datasets"], fmt_int, src)
        r2 = d["r2"]
        for blk, tok in (("overall", "RTwo"), ("nonnull", "RTwoNonNull")):
            for q in ("p25", "p50", "p75", "p95"):
                if r2.get(blk):
                    add((prefix, tok, token(q)), r2[blk].get(q), F2, src)
        for mode, qs in (r2.get("by_mode") or {}).items():
            for q in ("p50", "p95"):
                add((prefix, "RTwoMode", token(mode), token(q)), qs.get(q), F2, src)
        corr = d.get("corr") or {}
        add((prefix, "ShareMaxRGtNinety"), corr.get("share_ds_max_gt_090"), fmt_pct, src)
        add((prefix, "ShareMaxRGtNinetyFive"), corr.get("share_ds_max_gt_095"), fmt_pct, src)
        # realized share of factor columns (p_fac / p) per dataset: the design law's
        # split between the factor and the generic block, as drawn
        fs = d.get("factor_share") or {}
        add((prefix, "FactorShareZero"), fs.get("share_zero"), fmt_pct, src)
        for q, v in (fs.get("pcts") or {}).items():
            add((prefix, "FactorShare", token(q)), v, F2, src)
    q = doc["quadrature"]
    add(("quadNodesC",), doc["quad_nodes"]["c"], fmt_int, src)
    add(("quadNodesRho",), doc["quad_nodes"]["rho"], fmt_int, src)
    add(("quadNDatasets",), q["n_datasets"], fmt_int, src)
    for panel, v in q["panels"].items():
        pt = "Int" if panel == "interactions" else "Mains"
        add(("quadMaxDPip", pt), v["max_dpip"], SCI, src)
        add(("quadMaxDLogEv", pt), v["max_dlogev"], SCI, src)
        REG.add(name("quadGrid", pt), v["largest_tested"].replace("x", r"\times"),
                v["largest_tested"], src)
        add(("quadWorstN", pt), v["worst_cell"]["n"], fmt_int, src)
        add(("quadWorstP", pt), v["worst_cell"]["p"], fmt_int, src)
        # the grid the reference uses (c axis; rho is fixed in the paper's prior)
        chosen = (v.get("by_c") or {}).get(str(doc["quad_nodes"]["c"]))
        if chosen:
            add(("quadChosenMaxDPip", pt), chosen["max_dpip"], SCI, src)
            add(("quadChosenMaxDLogEv", pt), chosen["max_dlogev"], SCI, src)
            add(("quadChosenWorstN", pt), chosen["worst_cell"]["n"], fmt_int, src)
            add(("quadChosenWorstP", pt), chosen["worst_cell"]["p"], fmt_int, src)
        if v.get("by_c"):  # the tested c grids, "12, 15 and 20"
            REG.add(name("quadCGrid", pt), style.fmt_list(list(v["by_c"]), str),
                    list(v["by_c"]), src)
        for c_nodes, bc in (v.get("by_c") or {}).items():  # \\quadByCMaxDPipIntOneTwo, ...
            add(("quadByCMaxDPip", pt, token(c_nodes)), bc["max_dpip"], SCI, src)
            add(("quadByCMaxDLogEv", pt, token(c_nodes)), bc["max_dlogev"], SCI, src)
        if v.get("truth"):
            add(("quadFinestC", pt), v["truth"][0], fmt_int, src)
        if v.get("cells_n"):
            REG.add(name("quadCellsN", pt), style.fmt_list(v["cells_n"], lambda x: fmt_int(x, ST)),
                    v["cells_n"], src)
            REG.add(name("quadCellsP", pt), style.fmt_list(v["cells_p"], lambda x: fmt_int(x, ST)),
                    v["cells_p"], src)
    ex = doc.get("exact")
    if ex:  # the enumeration reach of the exact reference under the paper's prior
        add(("exactMaxP",), ex["p_max"], fmt_int, src)
        add(("exactNModelsMaxP",), ex["n_models_p_max"], fmt_int, src)
        add(("quadPruneTol",), ex["comp_prune_tol"], SCI, src)


def prior_grid(doc) -> None:
    """Where the pretraining mass sits on the (k/p, R2) plane (stage 42)."""
    src = "prior_grid.json"
    s = doc["summary"]
    add(("priorGridDatasets",), doc["n_datasets"], fmt_int, src)
    for key, part, fmt in (
            ("null_pct", "NullPct", F2),
            ("no_active_main_pct", "NoActiveMainPct", F2),
            ("r2_le_zero_five_pct", "RTwoLeZeroFivePct", F1),
            ("r2_ge_zero_seven_pct", "RTwoGeZeroSevenPct", F1),
            ("sparse_pct", "SparsePct", F1),
            ("r2_ge_zero_seven_given_sparse_pct", "RTwoGeZeroSevenGivenSparsePct", F2),
            ("r2_ge_zero_nine_given_sparse_pct", "RTwoGeZeroNineGivenSparsePct", F2),
            ("r2_ge_zero_seven_given_f_le_zero_one_five_pct",
             "RTwoGeZeroSevenGivenFLeZeroOneFivePct", F2),
            ("r2_ge_zero_nine_given_f_le_zero_one_five_pct",
             "RTwoGeZeroNineGivenFLeZeroOneFivePct", F2),
            ("bench_cells_total_pct", "BenchCellsTotalPct", F1),
            ("f_median", "MedianF", F2),
            ("r2_median", "MedianRTwo", F2)):
        add(("priorGrid", part), s.get(key), fmt, src)
    for cell, v in doc["bench_cells_pct"].items():
        add(("priorGrid", "Cell" + token(cell)), v, F2, src)


def real_x(doc) -> None:
    src = "real_x.json"
    ix, sel, ex = doc["index"], doc["selected"], doc["exclusions"]
    add(("rxIndexN",), ix["datasets"], fmt_int, src)
    add(("rxIndexPackages",), ix["packages"], fmt_int, src)
    add(("rxIncludedPackages",), ix["included_packages"], fmt_int, src)
    add(("rxIncludedDatasets",), ix["datasets_in_included_packages"], fmt_int, src)
    add(("rxFrameN",), doc["frame"], fmt_int, src)
    add(("rxDownloadedN",), doc["downloaded"], fmt_int, src)
    add(("rxEligibleN",), doc["eligible"], fmt_int, src)
    add(("rxSelectedN",), sel["datasets"], fmt_int, src)
    add(("rxSelectedPackages",), sel["packages"], fmt_int, src)
    add(("rxSelectedPLeFive",), sel["p_le_5"], fmt_int, src)
    add(("rxRowSubsampled",), sel["row_subsampled"], fmt_int, src)
    add(("rxColSubsampled",), sel["col_subsampled"], fmt_int, src)
    add(("rxCensusCells",), sel["census_cells"], fmt_int, src)
    add(("rxNotDrawn",), sel["not_drawn"], fmt_int, src)
    for unit, c in sel["row_unit"].items():
        add(("rxRowUnit", token(unit)), c, fmt_int, src)
    for dom, c in sel.get("domain", {}).items():
        add(("rxDomain", token(dom)), c, fmt_int, src)
        add(("rxDomainPackages", token(dom)), len(sel["domain_packages"][dom]), fmt_int, src)
    add(("rxDomainsN",), sel.get("domains"), fmt_int, src)
    add(("rxNMedian",), sel.get("n_median"), F0, src)
    add(("rxPMedian",), sel.get("p_median"), F0, src)
    add(("rxMaxAbsRMedian",), sel["max_abs_r"]["median"], F2, src)
    add(("rxMaxAbsRMax",), sel["max_abs_r"]["max"], F3, src)
    add(("rxNMin",), sel["n_range"][0], fmt_int, src)
    add(("rxNMax",), sel["n_range"][1], fmt_int, src)
    add(("rxPMin",), sel["p_range"][0], fmt_int, src)
    add(("rxPMax",), sel["p_range"][1], fmt_int, src)
    add(("rxExclTotal",), ex["total"], fmt_int, src)
    for reason, c in ex["by_reason"].items():
        add(("rxExcl", token(reason)), c, fmt_int, src)
    for stage, c in ex["by_stage"].items():
        add(("rxExclStage", token(stage)), c, fmt_int, src)
    for cell, v in doc["cells"].items():
        add(("rxCell", token(cell), "Selected"), v["selected"], fmt_int, src)
        add(("rxCell", token(cell), "Eligible"), v["eligible"], fmt_int, src)
    pool = doc.get("pool")
    if pool:  # \rxPool*: the 383-design pool of the main benchmark
        add(("rxPoolN",), pool["datasets"], fmt_int, src)
        add(("rxPoolPackages",), pool["packages"], fmt_int, src)
        add(("rxPoolDomainsN",), pool["domains"], fmt_int, src)
        for dom, c in pool["domain"].items():
            add(("rxPoolDomain", token(dom)), c, fmt_int, src)
            add(("rxPoolDomainPackages", token(dom)), len(pool["domain_packages"][dom]),
                fmt_int, src)
        top = sorted(pool["package_counts"].items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        for pkg, c in top:  # the JSON store sorts keys, so rank here
            add(("rxPoolPkg", token(pkg)), c, fmt_int, src)
        for unit, c in pool["row_unit"].items():
            add(("rxPoolRowUnit", token(unit)), c, fmt_int, src)
        add(("rxPoolRowSubsampled",), pool["row_subsampled"], fmt_int, src)
        add(("rxPoolColSubsampled",), pool["col_subsampled"], fmt_int, src)
        add(("rxPoolPLeFive",), pool["p_le_5"], fmt_int, src)
        add(("rxPoolNMin",), pool["n_range"][0], fmt_int, src)
        add(("rxPoolNMax",), pool["n_range"][1], fmt_int, src)
        add(("rxPoolPMin",), pool["p_range"][0], fmt_int, src)
        add(("rxPoolPMax",), pool["p_range"][1], fmt_int, src)
        add(("rxPoolNMedian",), pool["n_median"], F0, src)
        add(("rxPoolPMedian",), pool["p_median"], F0, src)
        add(("rxPoolMaxAbsRMedian",), pool["max_abs_r"]["median"], F2, src)
        add(("rxPoolMaxAbsRMax",), pool["max_abs_r"]["max"], F3, src)
    add(("rxGridCells",), len(doc["cells"]), fmt_int, src)
    add(("rxNPBins",), len(CFG["benchmark"]["p_bins"]), fmt_int, src)
    add(("rxNNBins",), len(CFG["benchmark"]["n_bins"]), fmt_int, src)
    quotas = {v["selected"] for v in doc["cells"].values()}
    if len(quotas) == 1:
        add(("rxQuotaPerCell",), quotas.pop(), fmt_int, src)
    add(("rxSeed",), doc["seed"], fmt_int, src)
    add(("rxMaxAbsCor",), doc.get("max_abs_cor"), F2, src)
    add(("rxMinN",), doc.get("min_n"), fmt_int, src)
    add(("rxMinP",), doc.get("min_p"), fmt_int, src)
    add(("rxMaxRowLoss",), doc.get("max_row_loss_frac"), fmt_pct, src)
    add(("rxMaxColMissing",), doc.get("max_col_na_frac"), fmt_pct, src)
    add(("rxMinMinorityRows",), doc.get("pool_min_minority_rows"), fmt_int, src)
    if doc.get("r_version"):
        REG.add(name("rxRVersion"), style.tex_escape(doc["r_version"]), doc["r_version"], src)
    add(("rxCleanFiles",), doc["clean_files"], fmt_int, src)


# \hp* macro -> PriorConfig field (the reported config's data block, defaults included)
HP_KEYS = (("hpFZeroProb", "f_zero_prob"), ("hpFBetaA", "f_beta_a"), ("hpFBetaB", "f_beta_b"),
           ("hpKMainZero", "k_main_zero"), ("hpKMainMean", "k_main_mean"),
           ("hpKIntZero", "k_int_zero"), ("hpKIntMean", "k_int_mean"),
           ("hpPiMainLo", "pi_main_low"), ("hpPiMainHi", "pi_main_high"),
           ("hpPiIntLo", "pi_int_low"), ("hpPiIntHi", "pi_int_high"),
           ("hpCountShare", "law_count_share"), ("hpSC", "c_logsd"), ("hpCMin", "c_min"),
           ("hpCMax", "c_max"), ("hpAZero", "a0"), ("hpBZero", "b0"),
           ("hpRho", "rho_int_fixed"), ("hpTauZeroSq", "tau2_intercept"),
           # the predictor prior's fixed values (the prior table of the appendix)
           ("hpHTwoMeanLo", "h2_mean_low"), ("hpHTwoMeanHi", "h2_mean_high"),
           ("hpHTwoConcLo", "h2_concentration_low"), ("hpHTwoConcHi", "h2_concentration_high"),
           ("hpHTwoCap", "h2_cap"), ("hpHTwoHighProb", "h2_high_prob"),
           ("hpHTwoHighMeanLo", "h2_high_mean_low"), ("hpHTwoHighCap", "h2_high_cap"),
           ("hpCrossLoadShare", "cross_loading_share"), ("hpNegLoadProb", "neg_loading_prob"),
           ("hpCrossBlockWMax", "cross_block_w_max"), ("hpBinaryRateLo", "binary_rate_low"),
           ("hpBinaryRateHi", "binary_rate_high"), ("hpLikertLevelsMin", "likert_levels_min"),
           ("hpLikertLevelsMax", "likert_levels_max"))


def model_card(doc) -> None:
    src = "model_card.json"
    a = doc["architecture"]
    m = a["model"]
    add(("modelLayers",), m.get("num_layers"), fmt_int, src)
    add(("modelHeads",), m.get("num_attention_heads"), fmt_int, src)
    add(("modelEmb",), m.get("embedding_size"), fmt_int, src)
    add(("modelMlp",), m.get("mlp_hidden_size"), fmt_int, src)
    add(("trainSteps",), a["steps"], fmt_int, src)
    add(("trainEffBatch",), a["effective_batch"], fmt_int, src)
    add(("trainPhysBatch",), a["physical_batch"], fmt_int, src)
    add(("trainAccum",), a["accum_steps"], fmt_int, src)
    add(("trainDatasetsSeen",), a["datasets_seen"], fmt_int, src)
    REG.add(name("trainDatasetsSeenBig"), style.fmt_big(a["datasets_seen"]),
            a["datasets_seen"], src)
    add(("trainLr",), a["lr"], GEN, src)
    if a.get("optimizer"):
        REG.add(name("trainOptimizer"), style.tex_escape(str(a["optimizer"])), a["optimizer"], src)
    if a["train"].get("muon_lr") is not None:
        add(("trainMuonLr",), a["train"]["muon_lr"], GEN, src)
    if a["train"].get("warmup_steps") is not None:
        add(("trainWarmup",), a["train"]["warmup_steps"], fmt_int, src)
    add(("trainNMin",), a["n_range"][0], fmt_int, src)
    add(("trainNMax",), a["n_range"][1], fmt_int, src)
    add(("trainPMin",), a["p_range"][0], fmt_int, src)
    add(("trainPMax",), a["p_range"][1], fmt_int, src)
    sb = doc["sbatch"]
    hw = doc.get("hardware") or {"gpu": sb.get("gpu"), "gpus": sb.get("gpus")}
    if hw.get("gpu"):
        REG.add(name("trainGpu"), style.tex_escape(hw["gpu"]), hw["gpu"], src)
    if hw.get("gpus"):
        add(("trainGpuCount",), hw["gpus"], fmt_int, src)
        REG.add(name("trainGpuCountWord"), style.num_word(hw["gpus"]), hw["gpus"], src)
    if sb.get("time"):
        REG.add(name("trainWalltimeLimit"), sb["time"].replace("-", " d "), sb["time"], src)
    if doc.get("params"):
        add(("modelParamsExact",), doc["params"]["params"], fmt_int, src)
        REG.add(name("modelParams"), style.fmt_big(doc["params"]["params"]),
                doc["params"]["params"], src)
    t = doc.get("train_log")
    if t:
        add(("trainStepsPerSec",), t["steps_per_s_overall"], F2, src)
        add(("trainWallHours",), t["wall_hours"], F1, src)
        add(("trainWallDays",), t["wall_hours"] / 24.0, F1, src)
        if t.get("gpu_hours") is not None:
            add(("trainGpuHours",), t["gpu_hours"], F1, src)
        add(("trainLossLast",), t["loss_last"], F3, src)
        add(("trainNllLast",), t["nll_last"], F3, src)
        add(("trainFinalStep",), t["final_step"], fmt_int, src)
        add(("trainRestarts",), t.get("restarts", 0), fmt_int, src)
    so = doc.get("slurm_out")
    if so:
        add(("trainPeakMemGiB",), so.get("peak_gpu_gib"), F1, src)
        add(("trainThroughputLogged",), so.get("throughput_steps_per_s"), F2, src)
        if so.get("resumed_from_step") is not None:
            add(("trainResumedStep",), so["resumed_from_step"], fmt_int, src)
        if so.get("gpu_total_gib") is not None:
            add(("trainGpuMemGiB",), so["gpu_total_gib"], F0, src)
    if t:
        REG.add(name("trainRestartsWord"), style.times_word(t.get("restarts", 0)),
                t.get("restarts", 0), src)
    if a["train"].get("checkpoint_every_minutes") is not None:
        add(("trainCkptMinutes",), a["train"]["checkpoint_every_minutes"], fmt_int, src)
    code = doc.get("code") or {}
    if code.get("bar"):  # the regression head's bar distribution (code defaults)
        add(("modelBins",), code["bar"]["n_inner_bins"], fmt_int, src)
        add(("modelBarLo",), code["bar"]["support"][0], GEN, src)
        add(("modelBarHi",), code["bar"]["support"][1], GEN, src)
    if code.get("query_fraction"):
        add(("trainQueryFracLo",), code["query_fraction"][0], GEN, src)
        add(("trainQueryFracHi",), code["query_fraction"][1], GEN, src)
    hp = doc.get("prior") or {}
    for tok, key in HP_KEYS:  # fixed values of the paper's prior (\hp*), from its config
        if hp.get(key) is not None:
            add((tok,), hp[key], GEN, src)
    for mtype, w in hp.get("marginal_weights") or []:  # \\hpMargGaussian, \\hpMargLikert, ...
        add(("hpMarg", token(mtype)), w, GEN, src)
    frac = hp.get("factor_count_frac_max")
    if frac:  # m ~ U{1..ceil(frac q)}: printed as q / k when frac = 1/k
        inv = 1.0 / frac
        if abs(inv - round(inv)) < 1e-9:
            add(("hpFactorFracInv",), round(inv), fmt_int, src)
    if hp.get("c_logmean") is not None:
        add(("hpCMedian",), math.exp(hp["c_logmean"]), F2, src)
        add(("hpMuC",), hp["c_logmean"], F2, src)
    for knob, inf in (doc.get("inference") or {}).items():
        if inf.get("pooled"):
            add(("inferSecMedian", token(knob)), inf["pooled"]["median"], fmt_seconds, src)
            add(("inferSecPNinety", token(knob)), inf["pooled"]["p90"], fmt_seconds, src)
        for b, fs in inf.get("by_p_bin", {}).items():
            add(("inferSecMedian", token(knob), "Pbin" + token(b)), fs["median"], fmt_seconds, src)


def real_y(doc) -> None:
    """The REAL-OUTCOME experiment: \\realyN, \\realyPipRAll, \\realyCoefRmseHead,
    \\realyNllOurs, \\realyPipRMainsAttitude, \\realyMpmSameAttitude, ..."""
    src = "real_y.json"
    add(("realyPendingN",), len(doc["pending"]), fmt_int, src)
    add(("realyNotConvergedN",), len(doc["not_converged"]), fmt_int, src)
    for ds_name in doc["order"]:
        d, t = doc["datasets"][ds_name], token(ds_name)
        add(("realyN", t), d["n"], fmt_int, src)
        add(("realyP", t), d["p"], fmt_int, src)
        add(("realyD", t), d["d"], fmt_int, src)
        for part in ("mains", "ints", "all"):
            add(("realyPipR", token(part), t), d["pip"][part]["r"], fmt_corr, src)
            add(("realyPipRmse", token(part), t), d["pip"][part]["rmse"], fmt_rmse, src)
            add(("realyPipMaxAbs", token(part), t), d["pip"][part]["max_abs"], fmt_rmse, src)
        add(("realyCoefRmseHead", t), d["coef"]["head"]["rmse"], fmt_rmse, src)
        add(("realyCoefRHead", t), d["coef"]["head"]["r"], fmt_corr, src)
        if d["coef"].get("probe"):
            add(("realyCoefRmseProbe", t), d["coef"]["probe"]["rmse"], fmt_rmse, src)
        add(("realyMpmSame", t), d["mpm"]["n_same"], fmt_int, src)
        add(("realyMpmSlots", t), d["mpm"]["n_slots"], fmt_int, src)
        add(("realyMpmJaccard", t), d["mpm"]["jaccard"], fmt_corr, src)
        add(("realyMpmSizeOurs", t), d["mpm"]["size_model"], fmt_int, src)
        add(("realyMpmSizeRef", t), d["mpm"]["size_reference"], fmt_int, src)
        for metric, mtok in (("nll", "Nll"), ("rmse", "Rmse")):
            for who, wtok in (("model", "Ours"), ("reference", "Ref"), ("ols_mains", "Ols"),
                              ("marginal", "Marginal")):
                add((f"realy{mtok}", wtok, t), d["held_out"][metric][who], F3, src)
        add(("realySecOurs", t), d["seconds"]["model"], fmt_seconds, src)
        add(("realySecRef", t), d["seconds"]["reference"], fmt_seconds, src)
        if d["mc3"]:
            add(("realyRhat", t), d["mc3"]["rhat_max"], F3, src)
            add(("realyEss", t), d["mc3"]["ess_min"], F0, src)
            add(("realySteps", t), d["mc3"]["steps"], fmt_int, src)
    pl = doc["pooled"]
    if not pl:
        return
    add(("realyN",), pl["n_datasets"], fmt_int, src)
    for part in ("mains", "ints", "all"):
        add(("realyPipR", token(part)), pl["pip"][part]["r"], fmt_corr, src)
        add(("realyPipRmse", token(part)), pl["pip"][part]["rmse"], fmt_rmse, src)
        add(("realyPipMaxAbs", token(part)), pl["pip"][part]["max_abs"], fmt_rmse, src)
        add(("realyPipSlots", token(part)), pl["pip"][part]["n"], fmt_int, src)
        add(("realyCoefRmseHead", token(part)), pl["coef_head"][part]["rmse"], fmt_rmse, src)
        add(("realyCoefRHead", token(part)), pl["coef_head"][part]["r"], fmt_corr, src)
        if pl.get("coef_probe"):
            add(("realyCoefRmseProbe", token(part)), pl["coef_probe"][part]["rmse"], fmt_rmse, src)
    medians = (("pip_r_mains", "PipRMains", fmt_corr), ("pip_rmse_all", "PipRmseAll", fmt_rmse),
               ("pip_max_abs", "PipMaxAbs", fmt_rmse), ("coef_rmse_head", "CoefRmseHead", fmt_rmse),
               ("mpm_jaccard", "MpmJaccard", fmt_corr), ("seconds_model", "SecOurs", fmt_seconds),
               ("seconds_reference", "SecRef", fmt_seconds))
    for key, tok, fmt in medians:
        add(("realyMedian", tok), pl["median"][key], fmt, src)
    # mean over TABLES (the main-text summary row), never over slots
    means = (("pip_r_all", "PipRAll", fmt_corr), ("pip_r_mains", "PipRMains", fmt_corr),
             ("pip_rmse_all", "PipRmseAll", fmt_rmse),
             ("coef_rmse_head", "CoefRmseHead", fmt_rmse), ("mpm_frac", "MpmFrac", fmt_pct))
    for key, tok, fmt in means:
        add(("realyMean", tok), (pl.get("mean") or {}).get(key), fmt, src)
    pr = pl["protocol"]
    for key, tok in (("n_min", "NMin"), ("n_max", "NMax"), ("p_min", "PMin"), ("p_max", "PMax"),
                     ("n_stress", "NStress"), ("steps_min", "StepsMin"), ("steps_max", "StepsMax")):
        add(("realy", tok), pr[key], fmt_int, src)
    add(("realyQuerySharePct",), pr["query_share_median"], fmt_pct, src)
    if len(pr["chains"]) == 1:
        add(("realyChains",), pr["chains"][0], fmt_int, src)
    add(("realyRhatMax",), pr["rhat_max"], F3, src)
    stress = [d["name"] for d in CFG["real_y"]["datasets"]
              if isinstance(d, dict) and d.get("stress")]
    if stress:
        REG.add(name("realyStressTables"), style.fmt_list(stress, style.tex_escape),
                stress, "config.yaml")
    add(("realyRhatThreshold",), CFG["mcmc"]["rhat_max"], F1, src)
    add(("realyEssThreshold",), CFG["mcmc"]["ess_min"], fmt_int, src)
    add(("realyEssMin",), pr["ess_min"], F0, src)
    add(("realyMpmIdentical",), pl["mpm_identical"], fmt_int, src)
    add(("realyMpmSlotsSame",), pl["mpm_slots_same"], fmt_int, src)
    add(("realyMpmSlots",), pl["mpm_slots"], fmt_int, src)
    add(("realyMpmSlotsSamePct",), pl["mpm_slots_same"] / pl["mpm_slots"], fmt_pct, src)
    add(("realyMaxPipGap",), pl["max_pip_gap"]["value"], fmt_rmse, src)
    REG.add(name("realyMaxPipGapTable"), style.tex_escape(pl["max_pip_gap"]["dataset"]),
            pl["max_pip_gap"]["dataset"], src)
    for metric, mtok in (("nll", "Nll"), ("rmse", "Rmse")):
        for who, wtok in (("model", "Ours"), ("reference", "Ref"), ("ols_mains", "Ols"),
                          ("marginal", "Marginal")):
            add((f"realy{mtok}", wtok), pl["held_out"][metric][who], F3, src)
        for cmp_, ctok in (("model_minus_reference", "Ref"), ("model_minus_ols", "Ols")):
            pr = pl["held_out"][cmp_][metric]
            add((f"realy{mtok}DiffMean", ctok), pr["mean"], S3, src)
            add((f"realy{mtok}DiffMedian", ctok), pr["median"], S3, src)
            add((f"realy{mtok}Wins", ctok), pr["wins"], fmt_int, src)


KNOB_POOLED = (("auc_macro", "Auc", fmt_auc), ("f1_macro", "FOne", fmt_auc),
               ("rmse_macro", "Rmse", fmt_rmse), ("ece", "Ece", fmt_ece))


def knob_tables(doc) -> None:
    r"""MISSPECIFICATION panel: per panel (None = the control, Coef / Int /
    Noise) and method, \knob<Metric><Method><Panel> (pooled, + Ece / EceMains /
    EceInts), \knobDatasets<Method><Panel>, paired \knobDiff / \knobDiffLo /
    \knobDiffHi / \knobWin / \knobP<Metric><Method><Panel> (model minus method
    within the panel), \knobExact<Metric><Method><Panel> on the exact posterior's
    subset, \knobNullEmpty / \knobNullFalse<Method><Panel>; per knob the PAIRED
    knob-minus-control \knobCtrlDiff / \knobCtrlDiffLo / \knobCtrlDiffHi /
    \knobCtrlWin / \knobCtrlP<Metric><Method><Knob> and \knobCtrlDiffEce<Method><Knob>."""
    src = "knob_tables.json"
    model = doc["model_row"]

    def toks(m):
        return [token(m)] + (["Ours"] if m == model else [])

    for pname, kd in doc["panels"].items():
        kt = token(pname)
        if kd.get("pending"):
            continue
        ic = kd.get("interactions")
        if ic:
            add(("knobIntCountMedian", kt), ic["median"], GEN, src)
            add(("knobIntCountMean", kt), ic["mean"], F2, src)
            add(("knobIntCountMax", kt), ic["max"], fmt_int, src)
            add(("knobIntNoneShare", kt), ic["share_none"], fmt_pct, src)
        for m, met in kd["pooled"].items():
            for mt in toks(m):
                emit_metrics("knob", met, (mt, kt), src)
                add(("knobEce", mt, kt), met.get("ece"), fmt_ece, src)
                add(("knobDatasets", mt, kt), met["n"], fmt_int, src)
        for m, e in kd.get("ece", {}).items():
            if not e:
                continue
            for mt in toks(m):
                add(("knobEceMains", mt, kt), e["ece_mains"], fmt_ece, src)
                add(("knobEceInts", mt, kt), e["ece_ints"], fmt_ece, src)
        for m, by_k in kd.get("paired", {}).items():
            mt = token(m)
            for key, ktok, _fmt in IP_METRICS:
                pr = by_k.get(key)
                if not pr:
                    continue
                add(("knobDiff", ktok, mt, kt), pr["mean_diff"], S3, src)
                add(("knobDiffLo", ktok, mt, kt), pr["ci_lo"], S3, src)
                add(("knobDiffHi", ktok, mt, kt), pr["ci_hi"], S3, src)
                add(("knobWin", ktok, mt, kt), pr["win_share"], fmt_pct, src)
                add(("knobP", ktok, mt, kt), pr["wilcoxon_p"], fmt_p, src)
                add(("knobPairedN", ktok, mt, kt), pr["n"], fmt_int, src)
        for m, met in kd.get("exact_subset", {}).items():
            for mt in toks(m):
                for key, ktok, fmt in KNOB_POOLED:
                    add(("knobExact", ktok, mt, kt), met.get(key), fmt, src)
                add(("knobExactN", mt, kt), met["n"], fmt_int, src)
        for m, nl in kd.get("null", {}).items():
            if not nl:
                continue
            for mt in toks(m):
                add(("knobNullN", mt, kt), nl["n"], fmt_int, src)
                add(("knobNullEmpty", mt, kt), nl["empty_share"], fmt_pct, src)
                add(("knobNullFalse", mt, kt), nl["mean_false_selections"], F2, src)
        for m, vc in kd.get("vs_control", {}).items():
            for mt in toks(m):
                for key, ktok, _fmt in IP_METRICS:
                    pr = vc.get(key)
                    if not pr:
                        continue
                    dfmt = SR if key == "rmse" else S3
                    add(("knobCtrlDiff", ktok, mt, kt), pr["mean_diff"], dfmt, src)
                    add(("knobCtrlDiffLo", ktok, mt, kt), pr["ci_lo"], dfmt, src)
                    add(("knobCtrlDiffHi", ktok, mt, kt), pr["ci_hi"], dfmt, src)
                    add(("knobCtrlWin", ktok, mt, kt), pr["win_share"], fmt_pct, src)
                    add(("knobCtrlP", ktok, mt, kt), pr["wilcoxon_p"], fmt_p, src)
                    add(("knobCtrlPairedN", ktok, mt, kt), pr["n"], fmt_int, src)
                e = vc.get("ece")
                if e:
                    add(("knobCtrlDiffEce", mt, kt), e["diff"], S3, src)
                    add(("knobCtrlDiffEceMains", mt, kt), e["knob_mains"] - e["control_mains"],
                        S3, src)
                    add(("knobCtrlDiffEceInts", mt, kt), e["knob_ints"] - e["control_ints"],
                        S3, src)
    add(("knobNDatasets",), doc.get("n_datasets"), fmt_int, src)


def main() -> int:
    docs = {n: load(n) for n in ("validation_final.json",
                                 "validation_arms.json", "prior.json", "real_x.json",
                                 "model_card.json", "ip_tables.json",
                                 "fit_time_gpu.json", "knob_tables.json", "real_y.json",
                                 "prior_grid.json")}
    validation(docs["validation_final.json"], docs["validation_arms.json"])
    if docs["prior.json"]:
        prior(docs["prior.json"])
    if docs["prior_grid.json"]:
        prior_grid(docs["prior_grid.json"])
    if docs["real_x.json"]:
        real_x(docs["real_x.json"])
    if docs["model_card.json"]:
        model_card(docs["model_card.json"])
    if docs.get("ip_tables.json"):
        ip_tables(docs["ip_tables.json"])
    if docs.get("fit_time_gpu.json"):
        fit_time_gpu(docs["fit_time_gpu.json"])
    if docs.get("knob_tables.json"):
        knob_tables(docs["knob_tables.json"])
    if docs.get("real_y.json"):
        real_y(docs["real_y.json"])
    REG.add("macrosModelRow", style.tex_escape(MODEL), MODEL, "config.yaml")
    REG.write(P["macros_tex"], P["macros_json"])
    store.write_json(PROV / "macros.json", {"macros": len(REG),
                                            "sources": [paths.rel(u) for u in USED if u.is_file()]},
                     USED, stage="99_results_macros")
    print(f"{len(REG)} macros -> {paths.rel(P['macros_tex'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
