"""Stage 72 (compute, GPU) — the reported model's fit time per real-design
dataset on a GPU: the SAME computation the benchmark timed on one CPU thread.

    python scripts/72_fit_time_gpu.py --ckpt checkpoints/paper_model/ckpt_final.pt \
        --cfg configs/paper_model.yaml --truth-dir results/main_benchmark/truth \
        --panel ipmx --out results/main_benchmark/fit_time_gpu.json

The benchmark's LinearPFN fit (bench/methods/linearpfn_slot.py) is one call
of forward_with_heads followed by the sigmoid of the selection logits and
the coefficient head's posterior mean; a checkpoint with a coefficient head
never probes. This script times exactly that per dataset, on cuda when
available (torch.cuda.synchronize before and after), after one warm-up
pass, and records one row per dataset: key, n, p, seconds, device. The
checkpoint load is timed once and reported separately. Reads the panel's
truth rows (X, y), never a fit row, so it runs anywhere the checkpoint
and the truth cache exist. Output is consumed by stage 73.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", required=True, help="the prior config the checkpoint was trained on")
    ap.add_argument("--truth-dir", default="results/diagnostic_panel/truth")
    ap.add_argument("--panel", default="ip2g")
    ap.add_argument("--out", default="reports/fit_time_gpu/real_x.json")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N datasets only")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    args = ap.parse_args()

    import torch

    from linearpfn.train import build_model, load_config, make_prior_config

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    t0 = time.perf_counter()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    prior_cfg = make_prior_config(load_config(args.cfg).get("data", {}))
    model, _bar = build_model(state["config"]["model"], prior_cfg.include_interactions)
    model.load_state_dict(state["model"])
    model.eval().to(device)
    load_seconds = time.perf_counter() - t0

    files = sorted(Path(args.truth_dir).glob(f"{args.panel}_*.npz"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no {args.panel}_*.npz under {args.truth_dir}")

    def fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        X_t = torch.from_numpy(X).float().unsqueeze(0).to(device)
        y_t = torch.from_numpy(y).float().unsqueeze(0).to(device)
        n = X.shape[0]
        sync()
        t = time.perf_counter()
        with torch.no_grad():
            _, sel, coef_out = model.forward_with_heads(X_t, y_t, n)
            prob = torch.sigmoid(sel.float())[0]
            if coef_out is not None:
                eff = model.coefficient_posterior_mean(sel[0], coef_out[0])
            else:
                eff = None
            sync()
        secs = time.perf_counter() - t
        return prob.cpu().numpy(), (eff.cpu().numpy() if eff is not None else None), secs

    z = np.load(files[0], allow_pickle=False)
    fit(z["X"], z["y"])  # warm-up (kernel load, allocator), not recorded
    rows = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        _prob, eff, secs = fit(z["X"], z["y"])
        rows.append({"key": f.stem, "n": int(z["n"]), "p": int(z["p"]), "seconds": secs,
                     "coef_source": "head" if eff is not None else "none"})
    secs = np.asarray([r["seconds"] for r in rows])
    out = {
        "ckpt": args.ckpt, "cfg": args.cfg, "panel": args.panel, "truth_dir": args.truth_dir,
        "device": device.type,
        "hardware": {"gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                     "cpu_threads": torch.get_num_threads(), "torch": torch.__version__,
                     "platform": platform.platform()},
        "protocol": "one forward_with_heads + sigmoid + coefficient head per dataset, "
                    "cuda-synchronized, after one warm-up; the benchmark's fit on a GPU",
        "load_seconds": load_seconds, "n_datasets": len(rows),
        "summary": {"median": float(np.median(secs)), "mean": float(secs.mean()),
                    "q25": float(np.percentile(secs, 25)), "q75": float(np.percentile(secs, 75)),
                    "p90": float(np.percentile(secs, 90)), "max": float(secs.max())},
        "rows": rows,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=1))
    print(f"{len(rows)} datasets on {device.type} ({out['hardware']['gpu']}): median "
          f"{out['summary']['median']:.4f} s, p90 {out['summary']['p90']:.4f} s, load "
          f"{load_seconds:.2f} s\n-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
