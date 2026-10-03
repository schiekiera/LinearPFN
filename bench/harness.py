from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from linearpfn.prior import Dataset, PriorConfig, n_effects, sample_dataset

BENCH_SEED = 2026
DEFAULT_N_VALUES = (32, 128, 512, 1024)
DEFAULT_P_VALUES = (3, 5, 8, 10, 15, 20, 30)
PIN_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BENCH_PINNED": "1",
}


@dataclass
class MethodResult:
    """What every method must return, in canonical effect order (gamma[1:]).

    score: higher = more likely active (threshold-free ranking for AUC/PR);
    prob: calibrated-ish inclusion probabilities (Tier 2 methods only);
    coef: length-d coefficient vector INCLUDING the intercept slot, or None;
    fit_seconds: fit + tuning, measured inside the method's process.
    """

    gamma_hat: np.ndarray
    score: np.ndarray | None
    prob: np.ndarray | None
    coef: np.ndarray | None
    fit_seconds: float
    meta: dict = field(default_factory=dict)

    def validate(self, d: int) -> None:
        assert self.gamma_hat.shape == (d - 1,), self.gamma_hat.shape
        for arr, name in ((self.score, "score"), (self.prob, "prob")):
            if arr is not None:
                assert arr.shape == (d - 1,), (name, arr.shape)
        if self.coef is not None:
            assert self.coef.shape == (d,), self.coef.shape
        assert np.isfinite(self.fit_seconds)


Method = Callable[[Dataset, PriorConfig], MethodResult]


def draw_cell_datasets(
    n: int, p: int, count: int, cfg: PriorConfig, seed: int = BENCH_SEED
) -> list[Dataset]:
    """Deterministic prefix stream for one cell (identical for every method)."""
    rng = np.random.default_rng([seed, n, p])
    return [sample_dataset(rng, cfg, n=n, p=p) for _ in range(count)]


def _result_path(out_dir: Path, method: str, n: int, p: int, i: int) -> Path:
    return out_dir / method / f"n{n}_p{p}_i{i}.npz"


def save_result(path: Path, ds: Dataset, res: MethodResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        gamma_hat=res.gamma_hat,
        score=np.asarray([]) if res.score is None else res.score,
        prob=np.asarray([]) if res.prob is None else res.prob,
        coef=np.asarray([]) if res.coef is None else res.coef,
        fit_seconds=res.fit_seconds,
        gamma_true=ds.gamma[1:],
        beta_true=ds.beta,
        max_absr=_max_absr(ds.X),
        meta=json.dumps(res.meta),
    )


def _max_absr(X: np.ndarray) -> float:
    p = X.shape[1]
    if p < 2:
        return 0.0
    R = np.corrcoef(X, rowvar=False)
    return float(np.abs(R[np.triu_indices(p, k=1)]).max())


def run_methods(
    methods: dict[str, Method],
    cfg: PriorConfig,
    n_values: tuple[int, ...] = DEFAULT_N_VALUES,
    p_values: tuple[int, ...] = DEFAULT_P_VALUES,
    datasets_per_cell: int = 25,
    out_dir: Path = Path("results/bench"),
    p_caps: dict[str, int] | None = None,
    seed: int = BENCH_SEED,
) -> list[str]:
    """Run every method over the strata grid; returns log lines.

    p_caps[m] limits method m to p <= cap (slow methods); every capped cell
    is logged — a cap is never silent.
    """
    p_caps = p_caps or {}
    log: list[str] = []
    for p in p_values:
        for n in n_values:
            datasets = draw_cell_datasets(n, p, datasets_per_cell, cfg, seed)
            for name, fn in methods.items():
                if p > p_caps.get(name, 10**9):
                    msg = f"SKIP {name} at p={p} (cap {p_caps[name]})"
                    if msg not in log:
                        log.append(msg)
                    continue
                for i, ds in enumerate(datasets):
                    path = _result_path(out_dir, name, n, p, i)
                    if path.exists():
                        continue
                    t0 = time.perf_counter()
                    res = fn(ds, cfg)
                    res.validate(n_effects(p, cfg))
                    save_result(path, ds, res)
                    log.append(
                        f"{name} n={n} p={p} i={i}: fit {res.fit_seconds:.2f}s "
                        f"(wall {time.perf_counter() - t0:.2f}s)"
                    )
    return log


def load_results(out_dir: Path, method: str) -> list[dict]:
    """All persisted per-dataset results for one method."""
    rows = []
    for path in sorted((out_dir / method).glob("n*_p*_i*.npz")):
        z = np.load(path, allow_pickle=False)
        stem = path.stem  # n{n}_p{p}_i{i}
        n, p, i = (int(x[1:]) for x in stem.split("_"))
        rows.append({
            "n": n, "p": p, "i": i,
            "gamma_hat": z["gamma_hat"].astype(bool),
            "score": z["score"] if z["score"].size else None,
            "prob": z["prob"] if z["prob"].size else None,
            "coef": z["coef"] if z["coef"].size else None,
            "fit_seconds": float(z["fit_seconds"]),
            "gamma_true": z["gamma_true"].astype(bool),
            "beta_true": z["beta_true"],
            "max_absr": float(z["max_absr"]),
            "meta": json.loads(str(z["meta"])),
        })
    return rows
