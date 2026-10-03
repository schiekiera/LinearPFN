"""Model-card facts of a trained checkpoint: architecture and recipe from
its config, parameter count from the checkpoint's own config (the model is
built on CPU, no forward pass), throughput / wall time / GPU-hours from
train_log.csv, hardware from the launcher's #SBATCH directives and the job's
stdout, and inference seconds per dataset from benchmark rows (fit_seconds
under the single-thread timing contract).

A requeued run (resumed from ckpt_latest.pt) restarts the `seconds` column
of train_log.csv at the resume, so the wall time is the SUM over segments,
and a data-parallel run multiplies it by the world size for GPU-hours;
both facts are read from the files, never assumed.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import yaml


def architecture(pin: Path | str) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(pin).read_text())
    m, t, d = cfg["model"], cfg["train"], cfg["data"]
    # physical batch = data.batch_size (the generator's), accumulation = train.accum_steps
    phys = int(d.get("batch_size", t.get("batch_size", 1)))
    eff = phys * int(t.get("accum_steps", 1))
    return {"model": m, "train": t, "physical_batch": phys,
            "accum_steps": int(t.get("accum_steps", 1)),
            "effective_batch": eff, "steps": int(t["steps"]),
            "datasets_seen": eff * int(t["steps"]), "optimizer": t.get("optimizer"),
            "lr": t.get("lr"), "n_range": [d.get("n_min"), d.get("n_max")],
            "p_range": [d.get("p_min"), d.get("p_max")], "sparsity_law": d.get("sparsity_law"),
            "seed": cfg.get("seed")}


def prior_config(pin: Path | str) -> dict[str, Any]:
    """Every field of the prior the config trains on (PriorConfig with the config's
    data block applied, so defaults the config leaves out are included): the source of the
    appendix's prior table and its \\hp* macros."""
    import dataclasses

    from linearpfn.train import load_config, make_prior_config

    cfg = make_prior_config(load_config(pin)["data"])
    return {k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(cfg).items()}


def code_constants() -> dict[str, Any]:
    """Recipe constants that live in code, not in the config: the regression head's
    bar distribution (the defaults `linearpfn.train` builds it with) and the
    query-row share of a training batch (`linearpfn.generator.QUERY_FRACTION`)."""
    import inspect

    from linearpfn._bardist import BarDistribution
    from linearpfn.generator import QUERY_FRACTION

    sig = inspect.signature(BarDistribution.__init__).parameters
    return {"bar": {"n_inner_bins": sig["n_inner_bins"].default,
                    "support": list(sig["support"].default),
                    "tail_scale": sig["tail_scale"].default},
            "query_fraction": list(QUERY_FRACTION)}


def param_count(ckpt: Path | str) -> dict[str, Any]:
    """Parameters of the checkpoint's model (built from the config stored
    inside it — the same call linearpfn.validate makes)."""
    import torch

    from linearpfn.train import build_model, make_prior_config

    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    prior = make_prior_config(state["config"].get("data", {}))
    model, _bar = build_model(state["config"]["model"], prior.include_interactions)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params": int(total), "trainable": int(trainable), "step": state.get("step"),
            "model_config": state["config"]["model"]}


def train_log_stats(csv_path: Path | str, tail_rows: int = 50) -> dict[str, Any]:
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows, skipped = [], 0
        for rec in reader:  # rows whose field count differs from the header: skip, never realign
            if len(rec) != len(header):
                skipped += 1
                continue
            rows.append({k: float(v) for k, v in zip(header, rec, strict=True) if v != ""})
    if not rows:
        return {"rows": 0, "rows_skipped": skipped}
    last = rows[-1]
    tail = rows[-tail_rows:]
    head = rows[:tail_rows]
    steps = last["step"]
    # `seconds` restarts at 0 when a requeued job resumes: one segment per restart,
    # wall time = sum of the segments' last readings (the unlogged steps between a
    # segment's last row and the resume point are not counted; < log_every steps)
    segments: list[dict[str, float]] = []
    seg_start, prev = rows[0], None
    for r in rows:
        sec = r.get("seconds")
        if sec is None:
            continue
        if prev is not None and sec < prev["seconds"]:
            segments.append({"first_step": seg_start["step"], "last_step": prev["step"],
                             "seconds": prev["seconds"]})
            seg_start = r
        prev = r
    if prev is not None:
        segments.append({"first_step": seg_start["step"], "last_step": prev["step"],
                         "seconds": prev["seconds"]})
    secs = sum(sg["seconds"] for sg in segments) if segments else float("nan")
    sps = [r["steps_per_s"] for r in rows if "steps_per_s" in r and r["steps_per_s"] > 0]
    return {"rows": len(rows), "rows_skipped": skipped, "final_step": int(steps),
            "wall_seconds": secs, "segments": segments, "restarts": max(len(segments) - 1, 0),
            "wall_hours": secs / 3600.0,
            "steps_per_s_mean": (sum(sps) / len(sps)) if sps else None,
            "steps_per_s_overall": (steps / secs) if secs else None,
            "loss_first": sum(r["loss"] for r in head) / len(head),
            "loss_last": sum(r["loss"] for r in tail) / len(tail),
            "nll_last": sum(r["nll"] for r in tail) / len(tail),
            "bce_mains_last": (sum(r["bce_mains"] for r in tail) / len(tail)
                               if "bce_mains" in tail[0] else None),
            "data_wait_frac_mean": (sum(r["data_wait_frac"] for r in rows) / len(rows)
                                    if "data_wait_frac" in rows[0] else None)}


_SBATCH_FLAGS = (("partition", r"(?:--partition=|-p )(\S+)"), ("gres", r"--gres=(\S+)"),
                 ("time", r"(?:--time=|-t )(\S+)"), ("cpus", r"(?:--cpus-per-task=|-c )(\d+)"),
                 ("mem", r"--mem=(\S+)"), ("job_name", r"(?:--job-name=|-J )(\S+)"),
                 ("nodes", r"(?:--nodes=|-N )(\d+)"))


def sbatch_facts(path: Path | str) -> dict[str, Any]:
    """#SBATCH directives of a launcher (long or short flags). `gpu` / `gpus`
    come from a --gres directive when the file has one; a launcher that takes
    gres on the command line leaves them None (see slurm_out_facts)."""
    text = Path(path).read_text()
    out: dict[str, Any] = {}
    for key, pat in _SBATCH_FLAGS:
        m = re.search(r"^#SBATCH\s+" + pat, text, re.M)
        out[key] = m.group(1) if m else None
    m = re.search(r"h200|h100|a100|l40", (out.get("gres") or "") + (out.get("partition") or ""),
                  re.I)
    out["gpu"] = m.group(0).upper() if m else None
    m = re.search(r"gpu:(?:[a-z0-9]+:)?(\d+)$", out.get("gres") or "")
    out["gpus"] = int(m.group(1)) if m else None
    out["requeue"] = bool(re.search(r"^#SBATCH\s+--requeue", text, re.M))
    return out


def slurm_out_facts(path: Path | str) -> dict[str, Any]:
    """Facts printed by train.py / the launcher into the job's stdout: final
    throughput and peak memory, the DDP world size, the GPUs nvidia-smi listed
    (count and model), the node, and the step a requeued job resumed from."""
    text = Path(path).read_text(errors="replace")
    out: dict[str, Any] = {}
    m = re.search(r"throughput: ([\d.]+) steps/s \| data-loader wait ([\d.]+)s of ([\d.]+)s loop "
                  r"\(([\d.]+)%\)", text)
    if m:
        out["throughput_steps_per_s"] = float(m.group(1))
        out["loop_seconds"] = float(m.group(3))
        out["data_wait_pct"] = float(m.group(4))
    m = re.search(r"peak GPU memory: ([\d.]+) GiB of ([\d.]+) GiB", text)
    if m:
        out["peak_gpu_gib"] = float(m.group(1))
        out["gpu_total_gib"] = float(m.group(2))
    m = re.search(r"^world=(\d+) .*?job=(\d+) node=(\S+)", text, re.M)
    if m:
        out["world_size"] = int(m.group(1))
        out["job_id"] = int(m.group(2))
        out["node"] = m.group(3)
    names = re.findall(r"^GPU \d+: (.+?) \(UUID", text, re.M)
    if names:
        out["gpu_names"] = names
        out["gpus"] = len(names)
        out["gpu_name"] = names[0] if len(set(names)) == 1 else " / ".join(sorted(set(names)))
    else:
        m = re.search(r"GPU: (.+)|device name: (.+)", text)
        if m:
            out["gpu_name"] = (m.group(1) or m.group(2)).strip()
    m = re.search(r"^resumed from \S+ at step (\d+)", text, re.M)
    if m:
        out["resumed_from_step"] = int(m.group(1))
    return out


def hardware(sbatch: dict[str, Any], slurm_out: dict[str, Any] | None) -> dict[str, Any]:
    """GPU model and count of the run, preferring what the job printed over
    the launcher's directive; short model tag (H200) beside the full name."""
    so = slurm_out or {}
    name = so.get("gpu_name") or sbatch.get("gpu")
    count = so.get("gpus") or so.get("world_size") or sbatch.get("gpus")
    m = re.search(r"h200|h100|a100|l40", name or "", re.I)
    return {"gpu_name": name, "gpu": m.group(0).upper() if m else name, "gpus": count,
            "node": so.get("node"), "job_id": so.get("job_id"),
            "source": "slurm_out" if so.get("gpu_name") else "sbatch"}


def inference_seconds(rx_tables: dict[str, Any], row: str,
                      panels: tuple[str, ...] = ("1g", "2g")) -> dict[str, Any]:
    """fit_seconds summaries of the model row per p-bin from a benchmark table
    file (panels -> methods -> fit-time slices)."""
    out: dict[str, Any] = {"row": row, "panels": list(panels), "by_p_bin": {}, "pooled": None}
    for panel in panels:
        t = rx_tables.get("panels", {}).get(panel)
        if not t or row not in t["pooled"]:
            continue
        out["pooled"] = t["pooled"][row]["fit_seconds"]
        for b, by_m in t["p_bin"].items():
            if row in by_m:
                out["by_p_bin"][b] = by_m[row]["fit_seconds"]
    return out
