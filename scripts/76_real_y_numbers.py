"""Stage 76 — the REAL-OUTCOME numbers: per table and pooled, the model's
agreement with the posterior under the prior, sampled by the MCMC sampler, on seven
published tables with their actual response, and the held-out predictive
scores (pipeline.real_y).

    python scripts/76_real_y_numbers.py

Reads results/real_outcomes (stage 18).
Output: reports/paper/real_y.json (pending datasets listed, never substituted).
"""

from __future__ import annotations

from _bootstrap import CFG, CONFIG, NUM, P, paths, store
from pipeline import real_y


def main() -> int:
    payload = real_y.summarize(CFG["real_y"], P["results_real_y"])
    out = store.write_json(NUM / "real_y.json", payload, [CONFIG, P["results_real_y"]],
                           stage="76_real_y_numbers")
    if payload["pooled"] is None:
        print(f"no finished real-y datasets yet; pending {payload['pending']}")
    else:
        pl = payload["pooled"]
        print(f"{pl['n_datasets']} datasets (MC3 reference); "
              f"pending {payload['pending']}; not converged {payload['not_converged']}")
        ho = pl["held_out"]["nll"]
        print(f"PIP r all {pl['pip']['all']['r']:.3f}, rmse {pl['pip']['all']['rmse']:.3f}; "
              f"coef head rmse {pl['coef_head']['all']['rmse']:.4f}; "
              f"MPM identical {pl['mpm_identical']}/{pl['n_datasets']}; "
              f"held-out NLL model {ho['model']:.3f} ref {ho['reference']:.3f} "
              f"ols {ho['ols_mains']:.3f}")
    print(f"-> {paths.rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
