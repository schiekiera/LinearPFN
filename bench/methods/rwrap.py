"""Generic Rscript bridge for R benchmark methods.

Protocol (one tmpdir per call, CSVs — deliberately boring):
  in:  X.csv       raw standardized mains (n x p), no header
       Z.csv       expanded design without intercept (n x (d-1)), no header
       y.csv       response (n,)
       pairs.csv   canonical interaction order, 1-BASED parent indices (i, j)
  out: result.csv  one row per canonical effect (d-1 rows, same order):
                   selected (0/1), score, prob (NA allowed), coef (NA allowed)
       intercept.txt   fitted intercept (or 0)
       timing.txt      fit-plus-tuning seconds measured INSIDE R around the
                       fit call (subprocess startup and I/O excluded)
       version.txt     R package version string (captured into meta)

Each R script decides which input it consumes: glinternet / hierNet take the
raw mains X and build interactions internally (their output is mapped back
to canonical order inside the R script using pairs.csv); susie consumes
the expanded design directly. The pinned single-thread env is inherited from
the harness process.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from bench.harness import PIN_ENV, MethodResult
from linearpfn.prior import Dataset, PriorConfig, build_design, interaction_pairs

RSCRIPTS = Path(__file__).resolve().parent.parent / "rscripts"


def r_method(name: str, timeout_s: float = 3600.0):
    """Registry entry for bench/rscripts/<name>.R."""

    def run(ds: Dataset, cfg: PriorConfig) -> MethodResult:
        import os

        script = RSCRIPTS / f"{name}.R"
        pairs = np.asarray(interaction_pairs(ds.X.shape[1], cfg), dtype=int)
        with tempfile.TemporaryDirectory(prefix=f"bench_{name}_") as tmp:
            tdir = Path(tmp)
            np.savetxt(tdir / "X.csv", ds.X, delimiter=",")
            np.savetxt(tdir / "Z.csv", build_design(ds.X, cfg)[:, 1:], delimiter=",")
            np.savetxt(tdir / "y.csv", ds.y, delimiter=",")
            np.savetxt(tdir / "pairs.csv", pairs.reshape(-1, 2) + 1,
                       delimiter=",", fmt="%d")
            proc = subprocess.run(
                ["Rscript", "--vanilla", str(script), str(tdir)],
                capture_output=True, text=True, timeout=timeout_s,
                env={**os.environ, **PIN_ENV},
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"{name}.R failed:\nSTDOUT:\n{proc.stdout[-2000:]}\n"
                    f"STDERR:\n{proc.stderr[-2000:]}"
                )
            table = np.genfromtxt(tdir / "result.csv", delimiter=",",
                                  skip_header=1, ndmin=2)
            intercept = float((tdir / "intercept.txt").read_text().strip() or 0.0)
            fit_seconds = float((tdir / "timing.txt").read_text().strip())
            version = (tdir / "version.txt").read_text().strip()
        d_minus_1 = table.shape[0]
        selected = table[:, 0] > 0.5
        score = table[:, 1]
        prob = table[:, 2]
        coef_eff = table[:, 3]
        prob_out = None if np.isnan(prob).all() else np.nan_to_num(prob, nan=0.0)
        if np.isnan(coef_eff).all():
            coef_out = None
        else:
            coef_out = np.concatenate([[intercept], np.nan_to_num(coef_eff, nan=0.0)])
        return MethodResult(
            gamma_hat=selected,
            score=None if np.isnan(score).all() else np.nan_to_num(score, nan=0.0),
            prob=prob_out,
            coef=coef_out,
            fit_seconds=fit_seconds,
            meta={"r_version": version, "d_minus_1": d_minus_1},
        )

    run.__name__ = f"r_{name}"
    return run
