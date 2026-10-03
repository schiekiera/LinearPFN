"""Stage 30 — validation numbers of the reported model (its full report and
the exact-posterior cells with the coefficient criterion scored on the
coefficient head) and of every ablation in paths.report_arms.

    python scripts/30_validation_numbers.py

Parses each report dir (results.json when present, else report.md), and
records the criteria themselves (thresholds, validation panel, probe settings)
from linearpfn.validate and the reported model's config.
Missing reports are recorded as pending. Outputs:
reports/paper/validation_final.json, reports/paper/validation_arms.json.
"""

from __future__ import annotations

from _bootstrap import CONFIG, NUM, P, paths, store
from pipeline import valreport


def _load(d):
    if not (d / "report.md").is_file():
        return {"pending": True, "path": paths.rel(d)}
    parsed = valreport.load(d)
    parsed["gate_row"] = valreport.gate_row(parsed)
    parsed["pending"] = False
    parsed["path"] = paths.rel(d)
    return parsed


def criteria(pin) -> dict:
    """What the criteria test and on which panel: the thresholds and the
    headroom flag (linearpfn.validate), the validation panel of the reported
    model's config (its validate block), the exact cells (p within the enumeration reach) and
    the probe settings the validation ran with."""
    from linearpfn import probe, validate
    from linearpfn.reference import max_enum_p
    from linearpfn.train import load_config, make_prior_config

    cfg = load_config(pin)
    v = cfg["validate"]
    reach = max_enum_p(make_prior_config(cfg["data"]))
    return {"thresholds": dict(validate.GATE_THRESHOLDS),
            "headroom_flag_nats": validate.HEADROOM_FLAG_NATS,
            "coverage_levels": [lvl for _lo, _hi, lvl in validate.COVERAGE_LEVELS],
            "panel": {"n_values": list(v["n_values"]), "p_values": list(v["p_values"]),
                      "exact_p_values": [p for p in v["p_values"] if p <= reach],
                      "datasets_per_cell": int(v["datasets_per_cell"]),
                      "n_query": int(v["n_query"]), "seed": v.get("seed")},
            "probe": {"points_per_effect": probe.POINTS_PER_EFFECT,
                      "jitter_sd": float(v.get("jitter_sd", probe.JITTER_SD)),
                      "ridge_lambda": float(v.get("ridge_lambda", probe.RIDGE_LAMBDA))}}


def main() -> int:
    final = {"final": _load(P["report_final"]),
             # the exact-posterior cells, coefficient criterion scored on the head
             "final_exact_panel": _load(P["report_final_exact_panel"]),
             "criteria": criteria(P["final_config"])}
    # every candidate input is recorded, present or MISSING: a report that lands
    # later must turn this stage stale (store.is_current)
    inputs = [CONFIG, P["final_config"],
              *[paths.ROOT / "linearpfn" / f for f in ("validate.py", "probe.py")]]
    inputs += [d / f for d in (P["report_final"], P["report_final_exact_panel"])
               for f in ("report.md", "results.json")]
    out1 = store.write_json(NUM / "validation_final.json", final, inputs,
                            stage="30_validation_numbers")
    arms = {}
    arm_inputs = [CONFIG] + [d / "report.md" for d in P["report_arms"]]
    for d in P["report_arms"]:
        arms[d.name] = _load(d)
    out2 = store.write_json(NUM / "validation_arms.json", {"arms": arms}, arm_inputs,
                            stage="30_validation_numbers")
    present = [k for k, v in arms.items() if not v["pending"]]
    print(f"final: {'pending' if final['final']['pending'] else 'parsed'}; "
          f"exact panel: {'pending' if final['final_exact_panel']['pending'] else 'parsed'}; "
          f"arms parsed: {present}")
    print(f"-> {paths.rel(out1)}\n-> {paths.rel(out2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
