"""Stage 80 — LaTeX tables from the paper's numbers (reports/paper/*.json).

    python scripts/80_tables.py

Every table is written only when its numbers exist; cells are formatted
by pipeline.style (never in LaTeX). Outputs: paper/tables/*.tex
+ the provenance sidecar reports/paper/_prov/tables.json.
"""

from __future__ import annotations

import math
from pathlib import Path

from _bootstrap import CFG, CONFIG, NUM, PROV, P, paths, store
from pipeline import figures, style, tables
from pipeline.style import (
    fmt_auc,
    fmt_corr,
    fmt_fixed,
    fmt_int,
    fmt_nll,
    fmt_pct,
    fmt_rmse,
    fmt_seconds,
    fmt_signed,
    yes_no,
)

ST = style.Style.from_config(CFG)
TAB = P["tables"]
USED: list[Path] = [CONFIG]
WRITTEN: list[str] = []


def load(name: str):
    p = NUM / name
    USED.append(p)  # recorded present or MISSING: a number file landing later = stale
    if p.is_file():
        return store.read_json(p)
    return None


def out(name: str, text: str) -> None:
    tables.write(TAB / name, text)
    WRITTEN.append(name)


def label(m: str) -> str:
    return style.tex_escape(figures.style_of(m)[0])


def paper_methods(names) -> list[str]:
    """The paper's method set in display order (figures.ORDER): baselines, the
    exact posterior and the model row. Rows outside ORDER never enter the
    benchmark tables."""
    return [m for m in figures.ORDER if m in set(names)]


GATE_NAMES = {"headroom": "headroom closed (min stratum)", "coef_r": "coefficient recovery r",
              "coverage": "coverage deviation (max)", "auc_gap": "selection AUC gap vs exact",
              "ece_gap": "selection ECE gap vs exact", "res_ratio": "resolution ratio vs exact"}


def _gate_bars() -> dict[str, str]:
    """Threshold column of the criteria tables, from linearpfn.validate.GATE_THRESHOLDS
    (the numbers the criteria apply), never retyped."""
    from linearpfn.validate import GATE_THRESHOLDS as T

    return {"headroom": f"$\\geq$ {T['headroom_min']:.2f}",
            "coef_r": f"$\\geq$ {T['coef_r_min']:.2f}",
            "coverage": f"$\\leq$ {T['coverage_tol']:.2f}",
            "auc_gap": f"$\\geq -{T['auc_gap']:.2f}$",
            "ece_gap": f"$\\leq$ {T['ece_gap']:.2f}",
            "res_ratio": f"$\\geq$ {T['res_pooled']:.2f} / {T['res_subset']:.2f}"}


GATE_BARS = _gate_bars()


def _mq(sm: dict | None, fmt) -> str:
    return "--" if not sm else f"{fmt(sm['mean'], ST)} [{fmt(sm['q25'], ST)}, {fmt(sm['q75'], ST)}]"


def ip_tables(doc) -> None:
    """MAIN benchmark on the real designs: per method mean [Q1, Q3] over datasets
    of AUC/ds, F1/ds and coefficient RMSE (all slots); fit time median [Q1, Q3]
    with the exact posterior and MC3 as reference rows (p <= 5 subset)."""
    c = CFG["ip"]
    panel = doc["panels"].get(c["panels"][c["fig_panel"]])
    if not panel:
        return
    shown = [m for m in doc["methods"] if m in c["baselines"] or m == c["model_row"]]
    order = [m for m in figures.ORDER if m in shown] + [m for m in shown if m not in figures.ORDER]
    rows = []
    for m in order:
        sm = panel["summary"].get(m, {})
        rows.append([label(m), _mq(sm.get("auc"), fmt_auc), _mq(sm.get("f1"), fmt_auc),
                     _mq(sm.get("rmse"), fmt_rmse)])
    out("ip_main.tex", tables.latex_table(
        ["method", "AUC/ds", "F1/ds", "coef RMSE"], rows))
    rows = []
    for m in order + [r for r in doc.get("reference_rows", []) if r in panel["fit_time"]]:
        ft = panel["fit_time"].get(m)
        if not ft:
            continue
        rows.append([label(m), fmt_seconds(ft["q50"], ST), fmt_seconds(ft["q25"], ST),
                     fmt_seconds(ft["q75"], ST), fmt_seconds(ft["p90"], ST), fmt_int(ft["n"], ST)])
    # the appendix caption states the timing contract and the reference rows' subsets
    out("ip_fit.tex", tables.latex_table(["method", "median", "Q1", "Q3", "p90", "datasets"],
                                         rows))


def ip_paired(doc) -> None:
    """Main benchmark, appendix: LinearPFN minus each baseline on the datasets both
    score, mean difference with its bootstrap 95 % interval, for F1/ds, AUC/ds, the
    dense coefficient RMSE and the hard-zeroed read-out's RMSE; the Wilcoxon p values
    that are not below 0.001 are named in the note (read from the data)."""
    c = CFG["ip"]
    panel = doc["panels"].get(c["panels"][c["fig_panel"]])
    if not panel:
        return
    order = [m for m in c["main_order"] if m in doc["methods"] and m != doc["model_row"]]
    hzp = panel.get("paired_hard_zeroed") or {}
    cols = (("f1", "F1/ds", 3, panel["paired"]), ("auc", "AUC/ds", 3, panel["paired"]),
            ("rmse", "RMSE", 4, panel["paired"]), ("rmse", "RMSE, hard-zeroed", 4, hzp))
    rows, exceptions = [], []
    for m in order:
        cells = []
        for key, head, digits, block in cols:
            pr = (block.get(m) or {}).get(key)
            if not pr:
                cells.append("--")
                continue
            cells.append(f"{fmt_signed(pr['mean_diff'], digits, ST)} "
                         f"\\iqr{{{fmt_signed(pr['ci_lo'], digits, ST)}}}"
                         f"{{{fmt_signed(pr['ci_hi'], digits, ST)}}}")
            if not pr["wilcoxon_p"] < 0.001:
                exceptions.append(f"{head} against {label(m)} "
                                  f"($p {style.fmt_p(pr['wilcoxon_p'], ST)}$)")
        rows.append([label(m), *cells])
    note = ("Wilcoxon signed-rank $p < 0.001$ for every difference"
            + (" except " + style.fmt_list(exceptions) if exceptions else "") + ".")
    out("ip_paired.tex", tables.latex_table(["method", *[h for _k, h, _d, _b in cols]], rows,
                                            note=note, resize=True))


def ip_summary(doc) -> None:
    """Table 2 of the main text: one row per method of the paper's method set,
    in the config's `ip.main_order`, with selection, coefficient recovery and
    fit cost side by side, every method timed on one CPU thread.

    The best value of each column is bold. The GPU time of LinearPFN's forward
    pass is not a row: it is the same computation on other hardware, so the
    manuscript states it in the caption and this
    note repeats it. The exact posterior and the MCMC sampler are not rows
    either, because neither is a competing method and both cover a different
    subset."""
    c = CFG["ip"]
    panel = doc["panels"].get(c["panels"][c["fig_panel"]])
    if not panel:
        return
    order = [m for m in c["main_order"] if m in doc["methods"]]
    if not order:
        return

    def sm(m, key):
        return panel["summary"].get(m, {}).get(key)

    auc_head = [fmt_auc(sm(m, "auc")["mean"], ST) if sm(m, "auc") else "--" for m in order]
    f1_head = [fmt_auc(sm(m, "f1")["mean"], ST) if sm(m, "f1") else "--" for m in order]
    rm_head = [fmt_rmse(sm(m, "rmse")["mean"], ST) if sm(m, "rmse") else "--" for m in order]
    ft = [panel["fit_time"].get(m) for m in order]
    med_head = [fmt_seconds(f["q50"], ST) if f else "--" for f in ft]
    auc_head = tables.bold_best(auc_head, [sm(m, "auc")["mean"] if sm(m, "auc") else None
                                           for m in order], "max")
    f1_head = tables.bold_best(f1_head, [sm(m, "f1")["mean"] if sm(m, "f1") else None
                                         for m in order], "max")
    rm_head = tables.bold_best(rm_head, [sm(m, "rmse")["mean"] if sm(m, "rmse") else None
                                         for m in order], "min")
    # one CPU thread per method, like for like; the GPU time of the same forward pass
    # is stated in the caption, not as a row
    med_head = tables.bold_best(med_head, [f["q50"] if f else None for f in ft], "min")
    gpu = load("fit_time_gpu.json")
    g = gpu["summary"] if gpu and not gpu.get("pending") and gpu.get("summary") else None

    def iqr(head, block, fmt):
        # \iqr is defined in the manuscript's preamble: it sets the bracket a size
        # smaller than the central value it follows
        if not block:
            return head
        return f"{head} \\iqr{{{fmt(block['q25'], ST)}}}{{{fmt(block['q75'], ST)}}}"

    rows = []
    for i, m in enumerate(order):
        rows.append([label(m), iqr(auc_head[i], sm(m, "auc"), fmt_auc),
                     iqr(f1_head[i], sm(m, "f1"), fmt_auc),
                     iqr(rm_head[i], sm(m, "rmse"), fmt_rmse),
                     iqr(med_head[i], ft[i], fmt_seconds)])
    device = n_gpu = None
    if g:
        device, n_gpu = gpu.get("hardware", {}).get("gpu"), g.get("n")
    note = ("Over datasets, with the first and third quartile in brackets. "
            "The mean summarizes $F_1$ and coefficient RMSE, the estimand the "
            "paired comparisons in the text test; the median summarizes fit seconds, whose mean "
            "the largest designs dominate. AUC ranks the candidate effects by each method's own "
            "native score, the PIP for LinearPFN and SuSiE, the selection frequency for "
            "stability selection and the order of entry on the path for the rest; it reads the "
            "ordering only, so those scales need not be commensurate. $F_1$ at each method's "
            "own selection rule, over the datasets with at least "
            "one true active effect; RMSE over every candidate slot with the intercept excluded. "
            "Stability selection has no coefficient estimate of its own, so its coefficients are "
            "an OLS refit on its selected support and its seconds include that refit. Fit seconds "
            "are one fit on one cluster CPU thread")
    if device:
        # the CPU rows' own dataset count, not a metric's n (F1 skips datasets with
        # no true active effect), so "the same datasets" is only claimed when it holds
        same = n_gpu == (panel["fit_time"].get(c["model_row"]) or {}).get("n")
        over = "the same datasets" if same else f"{fmt_int(n_gpu, ST)} of the datasets"
        note += (f". The same forward pass of LinearPFN on one "
                 f"{style.tex_escape(str(device))} "
                 f"over {over} takes {fmt_seconds(g['median'], ST)} s at the median")
    note += ". Best value per column in bold."
    out("ip_summary.tex", tables.latex_table(
        ["method", "AUC/ds, mean", "F1/ds, mean", "coef RMSE, mean",
         "fit s, median"], rows, note=note, full_width=True))


KNOB_LABEL = {"coef": "fixed-magnitude coefficients", "int": "interaction-heavy, $\\rho = 1$",
              "noise": "Student-$t_3$ noise"}


def knob_tables(doc) -> None:
    """MISSPECIFICATION panel: per knob and method, the pooled AUC/ds, F1/ds,
    coefficient RMSE on the knob panel, each with the PAIRED difference to the
    same method on the control panel in brackets (knob minus control, same
    dataset index); the control's own row first. No ECE column: the text quotes
    the ECE through the \\knob...Ece macros."""
    c = CFG["ip"]
    shown = [m for m in c["baselines"]] + [c["model_row"]]
    order = [m for m in figures.ORDER if m in shown]
    ctrl = doc["panels"].get(doc["control"])
    if not ctrl or ctrl.get("pending"):
        return

    def level(met, key, fmt):
        v = met.get(key)
        return "--" if v is None or v != v else fmt(v, ST)

    def cell(met, key, pr, fmt, digits):
        base = level(met, key, fmt)
        if base == "--" or not pr:
            return base
        return f"{base} ({fmt_signed(pr['mean_diff'], digits, ST)})"

    rows = []
    first = True
    for m in order:
        met = ctrl["pooled"].get(m)
        if not met:
            continue
        rows.append(["prior (control)" if first else "", label(m), level(met, "auc_macro", fmt_auc),
                     level(met, "f1_macro", fmt_auc), level(met, "rmse_macro", fmt_rmse)])
        first = False
    for knob in doc["knobs"]:
        kd = doc["panels"].get(knob)
        if not kd or kd.get("pending"):
            continue
        rows.append(["\\midrule"])
        first = True
        for m in order:
            met = kd["pooled"].get(m)
            if not met:
                continue
            vc = kd.get("vs_control", {}).get(m, {})
            rows.append([KNOB_LABEL.get(knob, knob) if first else "", label(m),
                         cell(met, "auc_macro", vc.get("auc"), fmt_auc, 3),
                         cell(met, "f1_macro", vc.get("f1"), fmt_auc, 3),
                         cell(met, "rmse_macro", vc.get("rmse"), fmt_rmse, ST.digits["rmse"])])
            first = False
    if len(rows) < 2:
        return
    out("knobs.tex", tables.latex_table(
        ["data", "method", "AUC/ds", "F1/ds", "coef RMSE"], rows,
        note="Misspecification panel on the real designs: y from the prior (control) and from the "
             "prior with one stage of the outcome model replaced, the same design and draw per "
             "dataset index. Entries are means over datasets; brackets give the paired difference "
             "to the same method on the control (knob minus control). RMSE over every "
             "candidate slot."))


def knobs_summary(doc) -> None:
    """Table 2 of the main text: one row per panel, the model against the best
    baseline on that panel.

    "Best" is chosen per panel and per metric among the baselines, which is the
    hardest comparison available and changes with the panel, so the cell names
    the baseline it refers to. Brackets hold the paired knob-minus-control
    difference of that same method; the control has none, being the reference.
    The per-method rows and the ECE column are in the appendix table."""
    c = CFG["ip"]
    model = c["model_row"]
    baselines = [m for m in figures.ORDER if m in c["baselines"]]
    ctrl = doc["panels"].get(doc["control"])
    if not ctrl or ctrl.get("pending"):
        return

    def val(panel, m, key):
        v = (panel["pooled"].get(m) or {}).get(key)
        return None if v is None or v != v else float(v)

    def best_baseline(panel, key, direction):
        live = [(m, val(panel, m, key)) for m in baselines if val(panel, m, key) is not None]
        if not live:
            return None, None
        return (max if direction == "max" else min)(live, key=lambda mv: mv[1])

    def pair(panel, knob, key, fmt, digits, direction):
        """(model cell, baseline cell) with the winner's number bold."""
        bm, bv = best_baseline(panel, key, direction)
        mv = val(panel, model, key)
        heads = tables.bold_best([fmt(mv, ST) if mv is not None else "--",
                                  fmt(bv, ST) if bv is not None else "--"],
                                 [mv, bv], direction)
        cells = []
        for i, m in enumerate((model, bm)):
            head = heads[i]
            if m is None or head == "--":
                cells.append("--")
                continue
            prefix = "" if m == model else f"{label(m)} "
            vc = (panel.get("vs_control", {}).get(m) or {}).get(key_metric(key)) if knob else None
            tail = f" ({fmt_signed(vc['mean_diff'], digits, ST)})" if vc else ""
            cells.append(f"{prefix}{head}{tail}")
        return cells

    def key_metric(key):
        return {"f1_macro": "f1", "rmse_macro": "rmse", "auc_macro": "auc"}[key]

    rows = []
    for knob, panel in [(None, ctrl)] + [(k, doc["panels"].get(k)) for k in doc["knobs"]]:
        if not panel or panel.get("pending"):
            continue
        name = "prior (control)" if knob is None else KNOB_LABEL.get(knob, knob)
        f1 = pair(panel, knob, "f1_macro", fmt_auc, 3, "max")
        rm = pair(panel, knob, "rmse_macro", fmt_rmse, ST.digits["rmse"], "min")
        rows.append([name, *f1, *rm])
    if len(rows) < 2:
        return
    out("knobs_summary.tex", tables.latex_table(
        ["data", "F1/ds LinearPFN", "F1/ds best baseline",
         "coef RMSE LinearPFN", "coef RMSE best baseline"], rows,
        note="Outcomes from the prior (control) and from the prior with one stage of the outcome "
             "model replaced, the same design and the same draw per dataset index. Means over "
             "datasets. The baseline column holds the best of the five baselines on that panel "
             "and that metric, named in the cell. Brackets give the paired difference to the same "
             "method on the control. Better of the two in bold. Every method and the ECE are in "
             "Table~\\ref{tab:knobs}."))


def real_y_summary(doc) -> None:
    """Table 3 of the main text: one row, the mean over the published tables of
    the agreement with the sampler and of the held-out predictive density.

    Every entry averages TABLES, not slots, so the seven carry equal weight
    whatever their size. The per-table values are in the appendix table."""
    mean = (doc.get("pooled") or {}).get("mean")
    nll = ((doc.get("pooled") or {}).get("held_out") or {}).get("nll")
    if not mean or not nll:
        return
    heads = tables.bold_best([fmt_nll(nll[k], ST) for k in ("model", "reference", "ols_mains")],
                             [nll["model"], nll["reference"], nll["ols_mains"]], "min")
    rows = [[f"mean over the {fmt_int(doc['pooled']['n_datasets'], ST)} tables",
             fmt_corr(mean["pip_r_all"], ST), fmt_rmse(mean["pip_rmse_all"], ST),
             fmt_rmse(mean["coef_rmse_head"], ST), fmt_pct(mean["mpm_frac"], ST), *heads]]
    out("real_y_summary.tex", tables.latex_table(
        ["", "PIP r", "PIP RMSE", "coef RMSE", "MPM agree", "NLL LinearPFN", "NLL MCMC",
         "NLL OLS"], rows,
        note="Published tables with their real response, y standardized. Agreement with the "
             "MCMC-sampled posterior under the same prior over every candidate effect: correlation "
             "and RMSE of the inclusion probabilities, RMSE of the posterior-mean coefficients, "
             "and the share of candidate effects on which the two median probability models "
             "agree. NLL is the mean negative log predictive density on a seeded held-out split, "
             "in sd(y) units, for LinearPFN, the sampler and OLS on the main effects. Lowest NLL "
             "in bold. Per-table values in Table~\\ref{tab:real-y}."))


def knobs_panels(doc) -> None:
    """Appendix companion to Table 1: the SAME columns, for every panel of the
    misspecification experiment at once.

    One table rather than four, with each panel a labelled block in the row
    order of `ip.main_order`, so a method is read down the column across the
    control and the three variants. Best value per column WITHIN a block, since
    the comparison of interest is between methods on one panel; the paired
    change from the control and the ECE stay in Table~\\ref{tab:knobs}. No GPU
    row: it was timed on the main panel only."""
    c = CFG["ip"]
    order = [m for m in c["main_order"] if m in doc["panels"][doc["control"]]["summary"]]
    if not order:
        return
    labels = {doc["control"]: "outcomes from the prior (control)",
              **{k: KNOB_LABEL.get(k, k) for k in doc["knobs"]}}
    rows: list[list[str]] = []
    for i, pname in enumerate([doc["control"], *doc["knobs"]]):
        panel = doc["panels"].get(pname)
        if not panel or panel.get("pending"):
            continue
        if i:
            rows.append(["\\midrule"])
        rows.append([f"\\multicolumn{{5}}{{l}}{{\\emph{{{labels[pname]}}}}}"])

        def blk(m, key, panel=panel):   # bound per block, not late
            return panel["summary"].get(m, {}).get(key)

        heads, vals = {}, {}
        for key, fmt, direction in (("auc", fmt_auc, "max"), ("f1", fmt_auc, "max"),
                                    ("rmse", fmt_rmse, "min")):
            vals[key] = [blk(m, key)["mean"] if blk(m, key) else None for m in order]
            heads[key] = tables.bold_best(
                [fmt(v, ST) if v is not None else "--" for v in vals[key]], vals[key], direction)
        ft = [panel["fit_time"].get(m) for m in order]
        fit_head = tables.bold_best([fmt_seconds(f["q50"], ST) if f else "--" for f in ft],
                                    [f["q50"] if f else None for f in ft], "min")

        def cell(head, block, fmt):
            if not block:
                return head
            return f"{head} \\iqr{{{fmt(block['q25'], ST)}}}{{{fmt(block['q75'], ST)}}}"

        for j, m in enumerate(order):
            rows.append([label(m),
                         cell(heads["auc"][j], blk(m, "auc"), fmt_auc),
                         cell(heads["f1"][j], blk(m, "f1"), fmt_auc),
                         cell(heads["rmse"][j], blk(m, "rmse"), fmt_rmse),
                         cell(fit_head[j], ft[j], fmt_seconds)])
    if len(rows) < 3:
        return
    out("knobs_panels.tex", tables.latex_table(
        ["method", "AUC/ds, mean", "F1/ds, mean", "coef RMSE, mean", "fit s, median"], rows,
        full_width=True,
        note="Over datasets, with the first and third quartile in brackets, the same columns and "
             "the same conventions as Table~\\ref{tab:ip-summary}. Each block is one panel of "
             "the misspecification experiment: the untouched prior draws and the three variants, "
             "paired by dataset index. Best value per column within a block in bold. The paired "
             "change from the control and the ECE are in Table~\\ref{tab:knobs}. Fit seconds are "
             "one fit on one cluster CPU thread; the GPU row of Table~\\ref{tab:ip-summary} has "
             "no counterpart here, it was timed on the main panel."))


def gates_final(doc) -> None:
    # the reported model only
    cols = [(k, v) for k, v in (("final", doc.get("final")),)
            if v and not v.get("pending")]
    if not cols:
        return
    rows = []
    for key in GATE_NAMES:
        cells = []
        for _k, v in cols:
            g = v["gates"].get(key)
            cells += (["--", "--"] if g is None
                      else [fmt_fixed(g["value"], 4, ST), yes_no(g["passed"])])
        rows.append([GATE_NAMES[key], GATE_BARS[key], *cells])
    hdr = ["criterion", "bar", *[c for k, _v in cols for c in
                              ("value (paper model)", "pass")]]
    note = None
    ep = doc.get("final_exact_panel")
    # the coefficient criterion scores the coefficient head; when the full report
    # scored the probe, the reported model's cell is taken from the exact-posterior
    # report of the same datasets, which scores the head
    if ep and not ep.get("pending") and ep.get("coef_source") == "head" \
            and cols and cols[0][0] == "final" and ep["gates"].get("coef_r"):
        g = ep["gates"]["coef_r"]
        for row in rows:
            if row[0] == GATE_NAMES["coef_r"]:
                row[2], row[3] = fmt_fixed(g["value"], 4, ST), yes_no(g["passed"])
        crp = ep.get("coef_recovery_probe")
        note = ("Coefficient recovery of the paper model is the coefficient head on the "
                "exact-panel rerun of the same datasets"
                + (f"; the probe reaches r = {crp['r']:.4f} on that panel and is not gated."
                   if crp else "."))
    out("gates_final.tex", tables.latex_table(hdr, rows, note=note))
    gates_main(doc)


# the main-text criteria table: a centred full-width table in two blocks with short
# names, the left block the predictive distribution and the coefficients, the right
# block the inclusion probabilities
GATE_BLOCKS = (("predictive distribution and coefficients",
                (("headroom", "headroom"), ("coverage", "coverage error"),
                 ("coef_r", "coefficient correlation"))),
               ("inclusion probabilities",
                (("auc_gap", "AUC gap"), ("ece_gap", "ECE gap"),
                 ("res_ratio", "resolution ratio"))))


def gates_main(doc) -> None:
    """Compact criteria table for the main text: two blocks side by side, each with
    criterion, threshold and the paper model's value, no note (the caption carries
    the context). The coefficient cell is the coefficient head on the
    exact-posterior report, as in gates_final."""
    v = doc.get("final")
    if not v or v.get("pending"):
        return
    gates = dict(v["gates"])
    ep = doc.get("final_exact_panel")
    if ep and not ep.get("pending") and ep.get("coef_source") == "head" \
            and ep["gates"].get("coef_r"):
        gates["coef_r"] = ep["gates"]["coef_r"]

    def cells(key: str, label_: str) -> list[str]:
        g = gates.get(key)
        return [label_, GATE_BARS[key], "--" if g is None else fmt_fixed(g["value"], 4, ST)]

    (lname, left), (rname, right) = GATE_BLOCKS
    assert len(left) == len(right), "the two blocks must have the same number of rows"
    lines = [tables.HEADER,
             "\\begin{tabular*}{\\textwidth}{@{}l@{\\extracolsep{\\fill}}rrlrr@{}}",
             "\\toprule",
             f"\\multicolumn{{3}}{{@{{}}l}}{{\\emph{{{lname}}}}} & "
             f"\\multicolumn{{3}}{{l@{{}}}}{{\\emph{{{rname}}}}} \\\\",
             "\\cmidrule(r){1-3}\\cmidrule(l){4-6}",
             "criterion & threshold & value & criterion & threshold & value \\\\",
             "\\midrule"]
    for (lk, ll), (rk, rl) in zip(left, right, strict=True):
        lines.append(" & ".join(cells(lk, ll) + cells(rk, rl)) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", ""]
    out("gates_main.tex", "\n".join(lines))


def gates_arms(doc) -> None:
    rows = []
    for arm, v in doc["arms"].items():
        if v.get("pending"):
            continue
        g = v["gates"]
        cells = [fmt_fixed(g[k]["value"], 4, ST) if k in g else "--" for k in GATE_NAMES]
        rows.append([arm.replace("_", "\\_"), str(v.get("step", "")), *cells,
                     f"{v['gates_passed']}/{v['gates_total']}"])
    if rows:
        out("gates_arms.tex", tables.latex_table(
            ["arm", "step", "headroom", "coef r", "coverage", "AUC gap", "ECE gap", "res ratio",
             "passed"], rows))


def real_x(doc) -> None:
    pb, nb = CFG["benchmark"]["p_bins"], CFG["benchmark"]["n_bins"]
    rows = []
    for p in pb:
        cells = []
        for n in nb:
            c = doc["cells"].get(f"p{p}_n{n}")
            cells.append("--" if c is None else f"{c['selected']} / {c['eligible']}")
        rows.append([f"p {p}", *cells])
    out("real_x_cells.tex", tables.latex_table(["", *[f"n {n}" for n in nb]], rows,
                                                note="selected / eligible designs per cell"))
    ex = doc["exclusions"]["by_reason"]
    rows = [[r.replace("_", "\\_"), fmt_int(c, ST)] for r, c in
            sorted(ex.items(), key=lambda kv: -kv[1])]
    rows.append(["\\midrule"])
    rows.append(["total", fmt_int(doc["exclusions"]["total"], ST)])
    out("real_x_exclusions.tex", tables.latex_table(["reason", "datasets"], rows))


def real_y(doc) -> None:
    """REAL OUTCOMES: one row per table — agreement of the model with the
    MCMC-sampled posterior under the prior and held-out predictive NLL."""
    rows = []
    for name in doc["order"]:
        d = doc["datasets"][name]
        ho = d["held_out"]["nll"]
        rows.append([style.tex_escape(name.replace("_", " ")) + ("" if d["converged"] else "*"),
                     fmt_int(d["n"], ST), fmt_int(d["p"], ST),
                     fmt_corr(d["pip"]["mains"]["r"], ST), fmt_rmse(d["pip"]["all"]["rmse"], ST),
                     fmt_rmse(d["pip"]["all"]["max_abs"], ST),
                     fmt_rmse(d["coef"]["head"]["rmse"], ST),
                     f"{d['mpm']['n_same']}/{d['mpm']['n_slots']}",
                     fmt_fixed(ho["model"], 3, ST), fmt_fixed(ho["reference"], 3, ST),
                     fmt_fixed(ho["ols_mains"], 3, ST)])
    for name in doc["pending"]:
        rows.append([style.tex_escape(name.replace("_", " ")), *["--"] * 10])
    out("real_y.tex", tables.latex_table(
        ["table", "n", "p", "PIP r (mains)", "PIP RMSE", "max |dPIP|", "coef RMSE",
         "MPM same", "NLL ours", "NLL MCMC", "NLL OLS"], rows,
        note=_real_y_note(doc), resize=True))


def _real_y_note(doc) -> str:
    pr = doc["pooled"]["protocol"]
    chains = pr["chains"][0] if len(pr["chains"]) == 1 else "/".join(map(str, pr["chains"]))
    star = ("; * marks a table whose sampler did not converge"
            if any(not doc["datasets"][n]["converged"] for n in doc["order"]) else "")
    return (f"Actual outcome columns, standardized. Agreement over every candidate effect "
            f"(main effects and pairwise interactions) with the MCMC sampler under the prior "
            f"({style.num_word(chains) if isinstance(chains, int) else chains} chains{star}). "
            f"MPM same: candidate effects on which the two median probability models agree. "
            f"Held-out NLL on a seeded split holding out {fmt_pct(pr['query_share_median'], ST)} "
            f"of the rows, in units of the outcome's standard deviation; OLS on the main effects.")


def _gpu_row(doc) -> str:
    hw = doc.get("hardware") or {}
    name = hw.get("gpu_name") or doc["sbatch"].get("gpu") or "pending"
    n = hw.get("gpus") or doc["sbatch"].get("gpus")
    return f"{n} x {name}" if n else str(name)


def _optimizer_row(a: dict) -> str:
    t = a["train"]
    if a.get("optimizer") == "muon" and t.get("muon_lr") is not None:
        return f"Muon ({t['muon_lr']:g}), AdamW ({a['lr']:g})"
    return f"{a.get('optimizer')} ({a.get('lr')})"


def _prior_row(doc: dict) -> str:
    pr = doc.get("prior") or {}
    law = {"mixture": "count and rate prior", "count": "count prior",
           "rate": "rate prior"}.get(pr.get("sparsity_law"), str(pr.get("sparsity_law")))
    her = pr.get("heredity_weights") or []
    return law + (", strong heredity" if her and her[0] == 1.0 else "")


def prior_config(doc) -> None:
    """Every fixed value of the paper's prior (the appendix table behind the
    \\hp* macros), from model_card.json's `prior` block (stage 60)."""
    pr = doc.get("prior")
    if not pr:
        return

    def g(x):
        return f"{x:g}"

    def u(lo, hi):
        return f"U({g(lo)}, {g(hi)})"

    frac = pr["factor_count_frac_max"]
    inv = 1.0 / frac
    frac_s = f"1/{round(inv)}" if abs(inv - round(inv)) < 1e-9 else g(frac)
    names = {"gaussian": "Gaussian", "likert": "Likert"}
    marg = ", ".join(f"{names.get(k, k)} {g(v)}" for k, v in pr["marginal_weights"])
    her = pr["heredity_weights"]
    her_s = "strong only" if her[0] == 1.0 else ", ".join(g(w) for w in her)

    def head(text):
        return [f"\\multicolumn{{2}}{{l}}{{\\emph{{{text}}}}}"]

    rows = [head("predictor prior"),
            ["rows $n$, predictors $p$",
             f"U$\\{{{pr['n_min']}, \\dots, {pr['n_max']}\\}}$, "
             f"U$\\{{{pr['p_min']}, \\dots, {pr['p_max']}\\}}$"],
            ["factor share $f$", f"$0$ w.p. {g(pr['f_zero_prob'])}, else "
                                 f"Beta({g(pr['f_beta_a'])}, {g(pr['f_beta_b'])})"],
            ["factors $m$", f"U$\\{{1, \\dots, \\lceil {frac_s}\\,q \\rceil\\}}$, "
                            "$q$ = factor-block size"],
            ["communality mean, concentration",
             f"{u(pr['h2_mean_low'], pr['h2_mean_high'])}, "
             f"{u(pr['h2_concentration_low'], pr['h2_concentration_high'])}, "
             f"cap {g(pr['h2_cap'])}"],
            ["high band: probability, mean", f"{g(pr['h2_high_prob'])}, "
             f"{u(pr['h2_high_mean_low'], pr['h2_high_cap'])}, cap {g(pr['h2_high_cap'])}"],
            ["share of $h^2_j$ on other factors", g(pr["cross_loading_share"])],
            ["sign flip of a loading", g(pr["neg_loading_prob"])],
            ["generic-column loading $|w_j|$", u(0, pr["cross_block_w_max"])],
            ["marginal types", marg],
            ["binary base rate, Likert levels",
             f"{u(pr['binary_rate_low'], pr['binary_rate_high'])}, "
             f"{pr['likert_levels_min']}--{pr['likert_levels_max']}"],
            ["\\midrule"],
            head("active-effects prior"),
            ["heredity", her_s],
            ["count prior probability $\\alpha$", g(pr["law_count_share"])],
            ["count prior, $\\zeta_{\\mathrm{main}}$, $\\bar{k}_{\\mathrm{main}}$",
             f"{g(pr['k_main_zero'])}, {g(pr['k_main_mean'])}"],
            ["count prior, $\\zeta_{\\mathrm{int}}$, $\\bar{k}_{\\mathrm{int}}$",
             f"{g(pr['k_int_zero'])}, {g(pr['k_int_mean'])}"],
            ["rate prior, $\\omega_{\\mathrm{main}}$", u(pr["pi_main_low"], pr["pi_main_high"])],
            ["rate prior, $\\omega_{\\mathrm{int}}$", u(pr["pi_int_low"], pr["pi_int_high"])],
            ["\\midrule"],
            head("effect-size prior"),
            ["slab scale $c$: median $e^{\\mu_c}$, $s_c$",
             f"{fmt_fixed(math.exp(pr['c_logmean']), 2, ST)}, {g(pr['c_logsd'])}"],
            ["truncation $[c_{\\min}, c_{\\max}]$", f"[{g(pr['c_min'])}, {g(pr['c_max'])}]"],
            ["interaction ratio $\\rho$", g(pr["rho_int_fixed"]) + " (fixed)"],
            ["intercept variance $\\tau_0^2$", g(pr["tau2_intercept"])],
            ["noise variance $\\sigma^2$: $a_0$, $b_0$", f"{g(pr['a0'])}, {g(pr['b0'])}"]]
    out("prior_config.tex", tables.latex_table(["", "value"], rows, align="ll"))


def model_card(doc) -> None:
    a = doc["architecture"]
    m = a["model"]
    arch = f"{m['num_layers']} / {m['num_attention_heads']} / {m['embedding_size']}"
    rows = [["layers / heads / embedding", arch],
            ["MLP hidden", str(m["mlp_hidden_size"])],
            ["optimizer steps", fmt_int(a["steps"], ST)],
            ["effective batch (physical x accumulation)",
             f"{a['effective_batch']} ({a['physical_batch']} x {a['accum_steps']})"],
            ["datasets seen", fmt_int(a["datasets_seen"], ST)],
            ["optimizer (learning rate)", _optimizer_row(a)],
            ["active-effects prior", _prior_row(doc)],
            ["n, p envelope",
             f"{a['n_range'][0]}--{a['n_range'][1]}, {a['p_range'][0]}--{a['p_range'][1]}"],
            ["GPUs", _gpu_row(doc)]]
    if doc.get("params"):
        rows.append(["parameters", f"{style.fmt_big(doc['params']['params'])} "
                                   f"({fmt_int(doc['params']['params'], ST)})"])
    if doc.get("train_log"):
        t = doc["train_log"]
        rows += [["wall time", style.fmt_hours(t["wall_hours"])]]
        if t.get("gpu_hours") is not None:
            rows.append(["GPU-hours", fmt_fixed(t["gpu_hours"], 1, ST)])
        rows.append(["throughput (steps/s)", fmt_fixed(t["steps_per_s_overall"], 2, ST)])
    if doc.get("slurm_out") and doc["slurm_out"].get("peak_gpu_gib"):
        rows.append(["peak GPU memory (GiB)", fmt_fixed(doc["slurm_out"]["peak_gpu_gib"], 1, ST)])
    out("model_card.tex", tables.latex_table(["", ""], rows, align="ll"))


def prior(doc) -> None:
    mx = doc["mixture"]
    es = mx["effect_size"]
    Q = ("p25", "p50", "p75")
    rows = [["main effects (prior)", *[fmt_fixed(es["mains"][q], 3, ST) for q in Q]],
            ["interactions (prior)", *[fmt_fixed(es["interactions"][q], 3, ST) for q in Q]],
            ["meta-analytic norms", *[fmt_fixed(es["norms"][q], 2, ST) for q in Q]]]
    # the norms' source is cited in the caption (\citet{gignac2016effect}), so no note
    out("prior_effect_sizes.tex", tables.latex_table(
        ["standardized effect", "Q1", "median", "Q3"], rows))
    q = doc["quadrature"]["panels"].get("interactions")
    if q and q.get("by_c"):
        rows = []
        for c_nodes, v in q["by_c"].items():
            wc = v["worst_cell"]
            rows.append([c_nodes, style.fmt_sci(v["max_dpip"]), style.fmt_sci(v["max_dlogev"]),
                         f"$n = {fmt_int(wc['n'], ST)}$, $p = {wc['p']}$"])
        out("quadrature.tex", tables.latex_table(
            ["nodes on c", "max PIP deviation", "max log-evidence deviation", "worst cell"],
            rows, align="rrrl"))


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    for name, fn in (("validation_final.json", gates_final),
                     ("validation_arms.json", gates_arms), ("real_x.json", real_x),
                     ("model_card.json", model_card), ("model_card.json", prior_config),
                     ("prior.json", prior),
                     ("ip_tables.json", ip_tables), ("ip_tables.json", ip_summary),
                     ("ip_tables.json", ip_paired),
                     ("knob_tables.json", knob_tables), ("knob_tables.json", knobs_summary),
                     ("knob_tables.json", knobs_panels),
                     ("real_y.json", real_y),
                     ("real_y.json", real_y_summary)):
        doc = load(name)
        if doc is not None:
            fn(doc)
    store.write_json(PROV / "tables.json", {"written": sorted(WRITTEN)}, USED, stage="80_tables")
    print(f"{len(WRITTEN)} tables -> {paths.rel(TAB)}: {sorted(WRITTEN)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
