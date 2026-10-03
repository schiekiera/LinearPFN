"""Stage 73 — the reported model's GPU fit time on the real designs, from the
finished stage-72 measurement (paths.fit_time_gpu_rows).

The measurement is read from the file the GPU job writes INSIDE the panel it
timed (results/main_benchmark/fit_time_gpu.json), so its datasets are the
same 10,000 the CPU rows beside it were timed on and the CPU/GPU ratio below
compares like with like.

    python scripts/73_fit_time_gpu_numbers.py

Summaries over datasets (median, quartiles, p90, mean), the same by p bin
and n bin, the hardware, the ratio of the benchmark's single-thread CPU
median (ip_tables.json, model row, real X) to the GPU median on the same
datasets, and the ratio of the MCMC sampler's median to the GPU median on the
datasets the sampler ran on (its cached rows from stage 20).
Pending (never guessed) until the measurement exists.
Output: reports/paper/fit_time_gpu.json.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import CFG, CONFIG, NUM, P, paths, store
from pipeline import ip


def _summary(v):
    v = np.asarray(v, float)
    return {"n": int(v.size), "median": float(np.median(v)), "mean": float(v.mean()),
            "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
            "p90": float(np.percentile(v, 90)), "max": float(v.max())}


def _mcmc_over_gpu(gpu_rows: list[dict]) -> dict | None:
    """Ratio of medians (sampler / GPU forward pass) on the datasets both timed,
    the way ip.distributions pairs CPU fit times; None while the rows are absent."""
    c = CFG["ip"]
    results = P[c["results_key"]]
    if "mc3" not in c["reference_rows"] or not (results / "mc3").is_dir():
        return None
    mc = {k: v["fit_seconds"] for k, v in
          ip.dataset_scores(ip.load_panel(results, "mc3", c["panels"]["real_x"])).items()}
    gpu = {r["key"]: r["seconds"] for r in gpu_rows}
    keys = sorted(k for k in mc if k in gpu)
    if not keys:
        return None
    a = float(np.median([mc[k] for k in keys]))
    b = float(np.median([gpu[k] for k in keys]))
    return {"n": len(keys), "mcmc_median": a, "gpu_median": b, "ratio": a / b}


def main() -> int:
    src = P["fit_time_gpu_rows"]
    ipt = NUM / "ip_tables.json"
    inputs = [CONFIG, src, ipt]
    if not src.is_file():
        out = store.write_json(NUM / "fit_time_gpu.json", {"pending": True, "path": paths.rel(src)},
                               inputs, stage="73_fit_time_gpu_numbers")
        print(f"pending: {paths.rel(src)}\n-> {paths.rel(out)}")
        return 0
    doc = store.read_json(src)
    rows = doc["rows"]
    pb, nb = ip.parse_bins(CFG["benchmark"]["p_bins"]), ip.parse_bins(CFG["benchmark"]["n_bins"])
    by_p, by_n = {}, {}
    for r in rows:
        by_p.setdefault(ip.bin_of(r["p"], pb), []).append(r["seconds"])
        by_n.setdefault(ip.bin_of(r["n"], nb), []).append(r["seconds"])
    payload = {"pending": False, "source": paths.rel(src), "device": doc["device"],
               "hardware": doc["hardware"], "protocol": doc["protocol"], "ckpt": doc["ckpt"],
               "load_seconds": doc["load_seconds"], "n_datasets": doc["n_datasets"],
               "summary": _summary([r["seconds"] for r in rows]),
               "p_bin": {k: _summary(v) for k, v in by_p.items() if k},
               "n_bin": {k: _summary(v) for k, v in by_n.items() if k}}
    if ipt.is_file():
        t = store.read_json(ipt)
        c = CFG["ip"]
        panel = t["panels"].get(c["panels"]["real_x"])
        cpu = panel and panel["fit_time"].get(c["model_row"])
        if cpu:
            payload["cpu_single_thread"] = cpu
            payload["cpu_over_gpu_median"] = cpu["q50"] / payload["summary"]["median"]
    mcmc = _mcmc_over_gpu(rows)
    if mcmc:
        payload["mcmc_over_gpu"] = mcmc
    out = store.write_json(NUM / "fit_time_gpu.json", payload, inputs,
                           stage="73_fit_time_gpu_numbers")
    print(f"{payload['n_datasets']} datasets on {payload['hardware'].get('gpu')}: median "
          f"{payload['summary']['median']:.4f} s\n-> {paths.rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
