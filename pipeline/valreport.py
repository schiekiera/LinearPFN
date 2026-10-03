"""Parser for the validation harness's report.md (linearpfn/_valreport.py).

The harness writes results.json beside report.md, and `load()` prefers it;
without it, the criteria table and every per-p / per-stratum table are parsed
from the markdown. Tables are located by their section heading and header signature, so a
new column appended by the harness does not break older columns.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

GATE_KEYS = ("headroom", "coef_r", "coverage", "auc_gap", "ece_gap", "res_ratio")
_GATE_WORDS = {"headroom": "headroom", "coefficient recovery": "coef_r", "coverage": "coverage",
               "AUC within": "auc_gap", "ECE within": "ece_gap", "resolution ratio": "res_ratio"}


def _value(s: str) -> Any:
    s = s.strip()
    if s in ("-", ""):
        return None
    if s in ("yes", "no"):
        return s == "yes"
    try:
        return int(s) if re.fullmatch(r"-?\d+", s) else float(s)
    except ValueError:
        return s


def parse_tables(text: str) -> list[dict[str, Any]]:
    """Every markdown table with the `## heading` it sits under."""
    out: list[dict[str, Any]] = []
    section = ""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("## "):
            section = ln[3:].strip()
        is_table = (ln.startswith("|") and i + 1 < len(lines)
                    and re.fullmatch(r"\|(?:-+\|)+", lines[i + 1].strip()))
        if is_table:
            header = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows = []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(dict(zip(header, [_value(c) for c in cells], strict=False)))
                i += 1
            out.append({"section": section, "header": header, "rows": rows})
            continue
        i += 1
    return out


def _find(tables: list[dict[str, Any]], section_prefix: str,
          first_col: str) -> dict[str, Any] | None:
    for t in tables:
        if t["section"].startswith(section_prefix) and t["header"] and t["header"][0] == first_col:
            return t
    return None


def parse_report(path: Path | str) -> dict[str, Any]:
    text = Path(path).read_text()
    tables = parse_tables(text)
    out: dict[str, Any] = {"path": str(path)}
    m = re.search(r"- checkpoint: `([^`]+)` \(step (\d+)\)", text)
    if m:
        out["checkpoint"], out["step"] = m.group(1), int(m.group(2))
    m = re.search(r"- generated: (.+)", text)
    if m:
        out["generated"] = m.group(1).strip()
    m = re.search(r"- gate mode: \*\*(\w+)\*\*", text)
    if m:
        out["gate_mode"] = m.group(1)
    gates = _find(tables, "Gates", "gate")
    out["gates"] = {}
    if gates:
        for r in gates["rows"]:
            key = next((k for w, k in _GATE_WORDS.items() if w in r["gate"]), None)
            if key:
                out["gates"][key] = {"name": r["gate"], "value": r["value"], "passed": r["passed"]}
    out["gates_passed"] = sum(1 for g in out["gates"].values() if g["passed"])
    out["gates_total"] = len(out["gates"])
    hr = _find(tables, "Predictive headroom", "n")
    out["headroom"] = hr["rows"] if hr else []
    m = re.search(r"Pooled p<=5: model ([\d.]+) \| marginal ([\d.]+) \| exact ([\d.]+) \| "
                  r"headroom ([\d.]+) \| closed ([\d.]+)", text)
    if m:
        keys = ("model_nll", "marginal_nll", "exact_nll", "headroom", "closed")
        out["pooled_p5"] = dict(zip(keys, map(float, m.groups()), strict=True))
    m = re.search(r"Pooled (probed|head)-vs-exact: r = ([\d.]+), RMSE = ([\d.]+)", text)
    if m:
        out["coef_recovery"] = {"r": float(m.group(2)), "rmse": float(m.group(3))}
        out["coef_source"] = "head" if m.group(1) == "head" else "probe"
    m = re.search(r"Probe \(reported, not gated\): r = ([\d.]+), RMSE = ([\d.]+)", text)
    if m:
        out["coef_recovery_probe"] = {"r": float(m.group(1)), "rmse": float(m.group(2))}
    cal = _find(tables, "Calibration", "nominal")
    if cal:
        out["coverage"] = {str(r["nominal"]).rstrip("%"): r["empirical"] for r in cal["rows"]}
    sel = [t for t in tables if t["section"].startswith("Selection head")]
    for t in sel:
        if t["header"][:2] == ["subset", "AUC"]:
            out["selection_vs_truth"] = {r["subset"]: r for r in t["rows"]}
        elif t["header"][:2] == ["subset", "r"]:
            out["selection_vs_exact"] = {r["subset"]: r for r in t["rows"]}
        elif t["header"][0] == "p":
            out["per_p"] = {str(r["p"]): r for r in t["rows"]}
    m = re.search(r"head ([\d.]+)% of (\d+) interactions \(all strata\), ([\d.]+)% on the "
                  r"exact panel, vs exact ([\d.]+)%", text)
    if m:
        out["heredity"] = {"head_all_pct": float(m.group(1)), "n_interactions": int(m.group(2)),
                           "head_exact_panel_pct": float(m.group(3)),
                           "exact_pct": float(m.group(4))}
    lp = _find(tables, "Large-p", "n")
    out["large_p"] = lp["rows"] if lp else []
    tr = _find(tables, "True-R2", "tercile")
    out["r2_terciles"] = tr["rows"] if tr else []
    strata = _find(tables, "prior strata", "stratum")
    out["strata"] = strata["rows"] if strata else []
    return out


def load(report_dir: Path | str) -> dict[str, Any]:
    """results.json when the harness wrote one, else the parsed
    report.md; both carry `gates` keyed by GATE_KEYS."""
    d = Path(report_dir)
    parsed = parse_report(d / "report.md")
    js = d / "results.json"
    if js.is_file():
        raw = json.loads(js.read_text())
        parsed["results_json"] = True
        parsed["pooled_raw"] = raw.get("pooled")
        parsed["step"] = raw.get("step", parsed.get("step"))
    return parsed


def gate_row(parsed: dict[str, Any]) -> dict[str, Any]:
    """Flat gate values for tables: {headroom: 0.97, headroom_pass: True, ...}."""
    row: dict[str, Any] = {"step": parsed.get("step"), "passed": parsed.get("gates_passed"),
                           "total": parsed.get("gates_total")}
    for k in GATE_KEYS:
        g = parsed["gates"].get(k)
        row[k] = g["value"] if g else None
        row[k + "_pass"] = g["passed"] if g else None
    return row
