"""Markdown + figure writer for the LinearPFN validation harness.

Private helper of `linearpfn.validate`.
Figures use matplotlib when importable; the markdown report is always
written and references only figures that were actually produced.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def _fmt(x, spec: str = ".4f") -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return format(x, spec)
    return str(x)


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _figures(results: dict, out: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    made: list[str] = []
    cells = results["cells"]
    records = results["records"]

    small = [c for c in cells if "gap_closed" in c]
    if small:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        labels = [f"n={c['n']}\np={c['p']}" for c in small]
        ax.bar(range(len(small)), [c["gap_closed"] for c in small], color="tab:blue")
        ax.axhline(0.95, color="tab:red", ls="--", lw=1, label="gate 0.95")
        ax.set_xticks(range(len(small)), labels, fontsize=8)
        ax.set_ylabel("headroom closed")
        ax.set_title("Fraction of marginal-to-exact headroom closed (p <= 5 strata)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "headroom_closed.png", dpi=120)
        plt.close(fig)
        made.append("headroom_closed.png")

    src = results["pooled"].get("coef_source", "probe")
    key = "coef_head" if src == "head" else "coef_probe"
    probed = [r for r in records if key in r and "coef_exact" in r]
    if probed:
        a = np.concatenate([r[key] for r in probed])
        # the head omits the intercept; the probe includes it
        b = np.concatenate([r["coef_exact"][1:] if src == "head" else r["coef_exact"]
                            for r in probed])
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.plot(b, a, ".", ms=3, alpha=0.4)
        lim = max(1e-3, float(np.abs(np.concatenate([a, b])).max())) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.8)
        ax.set_xlabel("exact posterior mean")
        ax.set_ylabel(f"model coefficient ({src})")
        ax.set_title(f"Coefficient recovery (r = {results['pooled']['coef_r']:.4f})")
        fig.tight_layout()
        fig.savefig(out / "coef_scatter.png", dpi=120)
        plt.close(fig)
        made.append("coef_scatter.png")

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    names = [50, 80, 90]
    emp = [results["pooled"][f"cover{k}"] for k in names]
    ax.bar(range(3), emp, color="tab:green", width=0.5)
    for i, k in enumerate(names):
        ax.plot([i - 0.35, i + 0.35], [k / 100] * 2, "k--", lw=1)
    ax.set_xticks(range(3), [f"{k}%" for k in names])
    ax.set_ylabel("empirical coverage")
    ax.set_title("Central predictive interval coverage (pooled)")
    fig.tight_layout()
    fig.savefig(out / "coverage.png", dpi=120)
    plt.close(fig)
    made.append("coverage.png")

    sel = results["pooled"]["sel"]
    if len(sel["hist_pred_p5"]):  # absent when the run has no p<=5 cells
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        bins = np.linspace(0, 1, 31)
        ax.hist(sel["hist_exact_p5"], bins=bins, alpha=0.55, density=True, label="exact PIPs")
        ax.hist(sel["hist_pred_p5"], bins=bins, alpha=0.55, density=True, label="head PIPs")
        ax.set_xlabel("inclusion probability")
        ax.set_ylabel("density")
        ax.set_title("PIP distributions, p <= 5 (degenerate = spike at base rate)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "pip_hist.png", dpi=120)
        plt.close(fig)
        made.append("pip_hist.png")

    her = sel["heredity"]
    if her["points_head"] is not None:
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), sharex=True, sharey=True)
        for ax, (x, y), name in zip(
            axes, [her["points_head"], her["points_exact"]], ["head", "exact posterior"],
            strict=True,
        ):
            ax.add_patch(plt.Rectangle((0, 0.5), 0.2, 0.5, color="tab:red", alpha=0.15))
            ax.plot([0, 1], [0, 1], "k--", lw=0.7)
            ax.plot(x, y, ".", ms=3, alpha=0.35)
            ax.set_xlabel("max parent PIP")
            ax.set_title(name)
        axes[0].set_ylabel("interaction PIP")
        fig.suptitle("Heredity consistency (red region = violation: int > 0.5, parents < 0.2)")
        fig.tight_layout()
        fig.savefig(out / "heredity.png", dpi=120)
        plt.close(fig)
        made.append("heredity.png")

    rel = results["pooled"].get("reliability")
    if rel:
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        conf = [r[0] for r in rel]
        acc = [r[1] for r in rel]
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.plot(conf, acc, "o-", color="tab:purple")
        ax.set_xlabel("predicted inclusion probability")
        ax.set_ylabel("empirical inclusion frequency")
        ax.set_title(f"Selection reliability (ECE = {results['pooled']['ece']:.4f})")
        fig.tight_layout()
        fig.savefig(out / "reliability.png", dpi=120)
        plt.close(fig)
        made.append("reliability.png")

    r2s = [r["surrogate_r2"] for r in records if "surrogate_r2" in r]
    if r2s:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.hist(r2s, bins=40)
        ax.set_xlabel("surrogate R2 of probe ridge fit")
        ax.set_ylabel("datasets")
        ax.set_title("Surrogate-R2 distribution (misspecification diagnostic)")
        fig.tight_layout()
        fig.savefig(out / "surrogate_r2.png", dpi=120)
        plt.close(fig)
        made.append("surrogate_r2.png")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for p in sorted({c["p"] for c in cells}):
        sub = sorted([c for c in cells if c["p"] == p], key=lambda c: c["n"])
        ax.plot([c["n"] for c in sub], [c["nll_model"] for c in sub], "o-", label=f"model p={p}")
        if all("nll_exact" in c for c in sub):
            ax.plot([c["n"] for c in sub], [c["nll_exact"] for c in sub], "k:", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("context size n")
    ax.set_ylabel("mean query NLL (nats)")
    ax.set_title("Predictive NLL vs n (dotted: exact BMA where available)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "nll_vs_n.png", dpi=120)
    plt.close(fig)
    made.append("nll_vs_n.png")
    return made


def write_report(results: dict, config: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    figures = _figures(results, out)
    cells = results["cells"]
    pooled = results["pooled"]
    mode = "HARD" if results["hard_gates"] else "informational (smoke model)"
    lines = [
        "# LinearPFN validation report",
        "",
        f"- checkpoint: `{results['ckpt']}` (step {results['step']})",
        f"- generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- gate mode: **{mode}**",
        "",
        "## Gates",
        "",
        _table(
            ["gate", "value", "passed"],
            [[g["name"], g["value"], g["passed"]] for g in results["gates"]],
        ),
        "## Predictive headroom (reference-comparable strata)",
        "",
        "Relative gate metric: (marginal - model) / (marginal - reference). Strata",
        "with headroom < 0.02 nats are flagged (percentage unstable) and excluded",
        "from the hard gate. Reference 'mc3' rows use the SAMPLED reference —",
        "approximate, never part of the hard gate, with chain diagnostics shown.",
        "",
        _table(
            ["n", "p", "reference", "model NLL", "marginal NLL", "ref NLL",
             "headroom (nats)", "closed", "flagged", "max R-hat", "min ESS"],
            [[c["n"], c["p"],
              (c.get("reference", "") + (" UNCONVERGED" if c.get("unconverged") else "")),
              c["nll_model"], c["nll_marginal"],
              c.get("nll_exact"), c.get("headroom"),
              (c.get("gap_closed") if "gap_closed" in c
               else None if "gap_closed_info" not in c
               else f"{c['gap_closed_info']:.3f} (info)"),
              c.get("flagged"), c.get("mc3_rhat_max"),
              None if "mc3_ess_min" not in c else round(c["mc3_ess_min"])]
             for c in cells if "nll_exact" in c or c.get("unconverged")],
        ),
        "Flagged rows print their ratio as '(info)': with < 0.02 nats of",
        "headroom the percentage is noise, and values above 1 (the model",
        "'beating' the exact reference) are expected there — sampling noise,",
        "never superiority. Flagged cells sit outside the hard gate.",
        "",
        "Bias direction of an unconverged sampled reference: missed",
        "high-evidence models RAISE the reference NLL toward the marginal",
        "baseline, shrinking the headroom denominator — so headroom-closed is",
        "INFLATED and unconverged chains flatter the model. Cells failing the",
        "convergence criteria (max R-hat < 1.1 and min ESS > 200 across the",
        "cell's datasets) are marked UNCONVERGED with all reference metrics",
        "suppressed, here and in every downstream panel.",
        "",
        (
            "Pooled p<=5: no p<=5 strata in this run."
            if pooled["p5"] is None
            else f"Pooled p<=5: model {pooled['p5']['nll_model']:.4f} | marginal "
            f"{pooled['p5']['nll_marginal']:.4f} | exact {pooled['p5']['nll_exact']:.4f} | "
            f"headroom {pooled['p5']['headroom']:.4f} | "
            f"closed {pooled['p5']['gap_closed']:.3f}"
        ),
        "",
        "## prior strata (reported, not gated)",
        "",
        "Headroom closed per data-generating stratum: heredity mode, column-",
        "type composition, correlation regime (generic X / factor X / factor",
        "with a realized |r| > 0.9 near-duplicate pair), and drawn-c tercile",
        (
            "(fixed analytic edges "
            + (
                "{:.3f} / {:.3f}".format(*pooled["c_tercile_edges"])
                if pooled.get("c_tercile_edges")
                else "n/a — c fixed under this prior"
            )
            + "). Under the fixed-scale configuration every stratum is degenerate. A pooled number"
        ),
        "hides weak cells; the hard gates stay per-(n, p).",
        "",
        _table(
            ["stratum", "label", "datasets", "model NLL", "marginal NLL",
             "exact datasets", "headroom", "closed"],
            [[s["kind"], s["label"], s["count"], s["nll_model"], s["nll_marginal"],
              s.get("count_exact"), s.get("headroom"), s.get("gap_closed")]
             for s in pooled.get("prior_strata", [])],
        ),
        "## Coefficient recovery (p <= 5)",
        "",
        f"Pooled {'head' if pooled.get('coef_source') == 'head' else 'probed'}-vs-exact: "
        f"r = {pooled['coef_r']:.5f}, RMSE = {pooled['coef_rmse']:.4f}",
        *([f"Probe (reported, not gated): r = {pooled['coef_r_probe']:.5f}, "
           f"RMSE = {pooled['coef_rmse_probe']:.4f}"]
          if pooled.get("coef_source") == "head" else []),
        "",
        "## Calibration (pooled coverage of central intervals)",
        "",
        _table(
            ["nominal", "empirical"],
            [[f"{k}%", pooled[f"cover{k}"]] for k in (50, 80, 90)],
        ),
        "## Selection head (amortized spike-and-slab PIPs)",
        "",
        f"Metrics vs TRUE gamma, pooled over {pooled['ece_effects']} effects from "
        f"{pooled['ece_datasets']} datasets, split by effect type. ECE alone is",
        "necessary-not-sufficient (base-rate output has ECE ~ 0 with zero",
        "resolution); sharpness lives in the Murphy resolution term and in r",
        "vs the exact PIPs. Interactions lagging mains flags the pooled",
        "concat(h_j+h_k, h_j*h_k) combination as the architectural bottleneck.",
        "",
        _table(
            ["subset", "AUC", "ECE", "Brier", "reliability", "resolution", "uncertainty"],
            [[name, s["auc"], s["ece"], s["brier"], s["reliability"], s["resolution"],
              s["uncertainty"]] for name, s in pooled["sel"]["subsets"].items()],
        ),
        "Against the exact posterior on identical datasets (exact panel).",
        "Correlation is DESCRIPTIVE (no gate): the pooled r is inflated by",
        "the mass of near-zero PIPs, so the restricted view (effects with",
        "exact PIP > 0.1) is shown alongside. The gates are the paired AUC",
        "gap, paired ECE gap, and resolution ratio (see module docstring):",
        "",
        _table(
            ["subset", "r", "r (exact>0.1)", "n>0.1", "RMSE", "RMSE (>0.1)",
             "AUC gap", "ECE model", "ECE exact", "res ratio"],
            [[name, d["r"], d.get("r_high"), d.get("n_high"), d["rmse"],
              d.get("rmse_high"),
              (d.get("auc_model", float("nan")) - d.get("auc_exact", float("nan"))),
              d.get("ece_model"), d.get("ece_exact"),
              d["res_model"] / max(d["res_exact"], 1e-12)]
             for name, d in pooled["sel"]["p5"].items()],
        ),
        "Per-p breakdown (vs true gamma everywhere; vs exact only at p <= 5).",
        "If the interaction gap widens with p, per-column pooling is failing",
        "exactly in the regime that cannot be validated exactly:",
        "",
        _table(
            ["p", "mains AUC", "int AUC", "int resolution", "int r (exact)",
             "int res ratio (exact)"],
            [[p, d["auc_mains"], d["auc_int"], d["res_int"], d.get("r_int_exact"),
              d.get("ratio_int")] for p, d in pooled["sel"]["per_p"].items()],
        ),
        "Heredity consistency — interaction PIP > 0.5 while both parent PIPs "
        "< 0.2, PAIRED against the exact posterior on identical datasets. "
        "Under a strong-heredity prior the exact rate is structurally 0 "
        "(int-PIP <= min parent PIP); under the heredity mixture a nonzero "
        "exact rate is legitimate orphan-interaction mass, so read the MODEL "
        "vs EXACT gap, not the raw model rate: "
        f"head {100 * pooled['sel']['heredity']['head_rate_all']:.3f}% of "
        f"{pooled['sel']['heredity']['n_int_all']} interactions (all strata), "
        f"{100 * pooled['sel']['heredity']['head_rate_p5']:.3f}% on the exact "
        f"panel, vs exact {100 * pooled['sel']['heredity']['exact_rate_p5']:.3f}% "
        "on identical datasets.",
        "",
        "## Large-p sweeps (no exact reference; OLS needs n > d, oracle n > 2|active|)",
        "",
        _table(
            ["n", "p", "model NLL", "marginal NLL", "OLS NLL", "oracle NLL"],
            [[c["n"], c["p"], c["nll_model"], c["nll_marginal"], c.get("nll_ols"),
              c.get("nll_oracle")]
             for c in cells if c["p"] > 5 and "nll_exact" not in c],
        ),
        "## True-R2 terciles (pooled across all strata)",
        "",
        f"Tercile edges: {pooled['r2_edges'][0]:.3f}, {pooled['r2_edges'][1]:.3f}",
        "",
        _table(
            ["tercile", "datasets", "model NLL", "marginal NLL", "closed (p<=5)",
             "surrogate R2 5th pct"],
            [[t["tercile"], t["count"], t["nll_model"], t["nll_marginal"],
              t.get("gap_closed_p5"), t.get("surrogate_r2_p5")]
             for t in pooled["terciles"]],
        ),
        "## Surrogate-R2 diagnostic per stratum",
        "",
        "Low values mark datasets where the fitted predictive surface left the",
        "linear-plus-interaction class — the misspecification detector. There",
        "is NO absolute cutoff: the in-prior level is stratum-dependent, so",
        "real-data values must be reported as quantiles against the matched",
        "(n, p) cell of the calibration table persisted alongside this report",
        "(`surrogate_baseline.json`; see `linearpfn.validate.surrogate_quantile`).",
        "The table is capacity-dependent and must be regenerated whenever the",
        "model is retrained.",
        "",
        _table(
            ["n", "p", "5th pct", "min"],
            [[c["n"], c["p"], c.get("surrogate_r2_p5"), c.get("surrogate_r2_min")]
             for c in cells],
        ),
    ]
    lines += [
        "## Post-training checklist",
        "",
        "- [ ] regenerate the surrogate-R2 baseline (`surrogate_baseline.json`)",
        "      with the trained checkpoint — the table is capacity-dependent",
        "- [ ] rerun validation with `validate.hard_gates: true`",
        "- [ ] run moderate-p panels (p in {8, 10, 15}) against the sampled",
        "      MCMC reference, clearly labelled approximate",
        "",
    ]
    if figures:
        lines += ["## Figures", ""]
        lines += [f"![{f}]({f})" for f in figures]
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
