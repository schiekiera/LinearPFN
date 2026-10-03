from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRIOR_CONFIG = ROOT / "configs" / "paper_model.yaml"  # default prior the methods assume


def _fit(entries, methods: dict, cfg, out_dir: Path, shard: tuple[int, int]) -> None:
    from bench.shared import _save

    i, nshards = shard
    todo = [(idx, e) for idx, e in enumerate(entries) if idx % nshards == i]
    print(f"shard {i}/{nshards}: {len(todo)} datasets x {len(methods)} methods")
    for m in methods:  # once up front: parallel shards create the same directories
        for attempt in range(5):  # and may race on a shared filesystem, so retry
            try:
                (out_dir / m).mkdir(parents=True, exist_ok=True)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.5 * (attempt + 1))
    for _, (key, make, meta) in todo:
        paths = {m: out_dir / m / f"{key}.npz" for m in methods}
        if all(p.exists() for p in paths.values()):
            continue
        ds, place_meta = make()
        for name, fn in methods.items():
            if paths[name].exists():
                continue
            t0 = time.perf_counter()
            try:
                res = fn(ds, cfg)
                res.validate(ds.gamma.size)
            except Exception as exc:  # a degenerate dataset must cost one
                # cell, never the shard (e.g. a Cholesky failure on exactly
                # duplicated columns). No npz is written; the per-cell (n)
                # counts expose the gap.
                print(f"FAIL {name} {key}: {type(exc).__name__}: {exc}",
                      flush=True)
                continue
            _save(paths[name], ds, res, {**meta, **place_meta})
            print(f"{name} {key}: fit {res.fit_seconds:.2f}s "
                  f"(wall {time.perf_counter() - t0:.2f}s)", flush=True)


def _save(path: Path, ds, res, meta: dict) -> None:
    from bench.harness import _max_absr

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
        n=ds.X.shape[0],
        p=ds.X.shape[1],
        max_absr=_max_absr(ds.X),
        meta=json.dumps({**meta, **res.meta}),
    )


def _load_all(out_dir: Path, method: str, prefix: str = "ss_") -> list[dict]:
    rows = []
    for path in sorted((out_dir / method).glob(f"{prefix}*.npz")):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        rows.append({
            "n": int(z["n"]), "p": int(z["p"]), "i": path.stem,
            "gamma_hat": z["gamma_hat"].astype(bool),
            "score": z["score"] if z["score"].size else None,
            "prob": z["prob"] if z["prob"].size else None,
            "coef": z["coef"] if z["coef"].size else None,
            "fit_seconds": float(z["fit_seconds"]),
            "gamma_true": z["gamma_true"].astype(bool),
            "beta_true": z["beta_true"],
            "max_absr": float(z["max_absr"]),
            "meta": meta,
        })
    return rows


def _macro_rmse(rows: list[dict]) -> float:
    """Per-dataset coefficient RMSE vs true beta (intercept excluded),
    averaged over datasets; NaN for methods without coefficients."""
    vals = []
    for r in rows:
        if r.get("coef") is None:
            return float("nan")
        err = np.asarray(r["coef"][1:], dtype=float) - np.asarray(r["beta_true"][1:], dtype=float)
        vals.append(float(np.sqrt(np.mean(err ** 2))))
    return float(np.mean(vals)) if vals else float("nan")
