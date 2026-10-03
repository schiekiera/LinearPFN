"""Stage 40 — prior calibration and quadrature numbers.

    python scripts/40_prior_numbers.py

The preflight of the reported model's prior (key "mixture": its
active-effects prior is the count/rate mixture), the quadrature convergence
table, and the exact reference's reach under the reported prior (largest p,
number of models, pruning tolerance); node counts and reach come from
linearpfn.reference.
Output: reports/paper/prior.json.
"""

from __future__ import annotations

from _bootstrap import CONFIG, NUM, P, paths, store
from linearpfn import reference
from pipeline import priornums


def exact_reach(pin) -> dict:
    """The enumeration reach of the exact reference under the reported prior: the
    largest p it enumerates, the number of models with prior mass there, and the
    component-pruning tolerance of `fit_exact` (its keyword default)."""
    import inspect

    from linearpfn.train import load_config, make_prior_config

    cfg = make_prior_config(load_config(pin)["data"])
    p_max = reference.max_enum_p(cfg)
    models, _log_prior = reference.enumerate_models(p_max, cfg)
    tol = inspect.signature(reference.fit_exact).parameters["comp_prune_tol"].default
    return {"p_max": int(p_max), "n_models_p_max": int(models.shape[0]),
            "comp_prune_tol": float(tol)}


def main() -> int:
    pre = P["preflight"] / "preflight.json"
    quad = P["quadrature"] / "results.json"
    payload = {"mixture": priornums.preflight_numbers(pre),
               "quadrature": priornums.quadrature_numbers(quad),
               "quad_nodes": {"c": reference.QUAD_NODES_C, "rho": reference.QUAD_NODES_RHO},
               "exact": exact_reach(P["final_config"])}
    # absent inputs are recorded as MISSING
    inputs = [CONFIG, pre, quad, P["final_config"]]
    out = store.write_json(NUM / "prior.json", payload, inputs, stage="40_prior_numbers")
    m = payload["mixture"]
    print(f"mixture preflight: {m['checks']['passed']}/{m['checks']['total']} checks; "
          f"mains |r| median {m['effect_size']['mains']['p50']:.3f}\n-> {paths.rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
