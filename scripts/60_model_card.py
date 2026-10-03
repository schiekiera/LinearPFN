"""Stage 60 — model-card facts of the reported model.

    python scripts/60_model_card.py

Architecture, recipe and every prior field from its config (paths.final_config),
the bar-distribution and query-share constants from the code, launcher
directives from paths.train_sbatch, hardware (GPU model and count) and
peak memory from the training job's stdout (paths.train_slurm_out), parameter
count from the checkpoint's own config, wall time from train_log.csv (summed
over resumed segments) and GPU-hours = wall time x GPU count. Inference
seconds come from the main benchmark (ip_tables.json, fit_time_gpu.json), not
from here. Missing pieces are null, never guessed. Output: reports/paper/model_card.json.
"""

from __future__ import annotations

from _bootstrap import CONFIG, NUM, P, paths, store
from pipeline import modelcard


def main() -> int:
    pin, sbatch = P["final_config"], P["train_sbatch"]
    ck = P["checkpoints_final"]
    payload = {"pin": paths.rel(pin), "architecture": modelcard.architecture(pin),
               "prior": modelcard.prior_config(pin), "code": modelcard.code_constants(),
               "sbatch": modelcard.sbatch_facts(sbatch), "params": None, "train_log": None,
               "slurm_out": None, "hardware": None, "pending": []}
    ckpt = ck / "ckpt_final.pt"
    log = ck / "train_log.csv"
    slurm_out = P["train_slurm_out"]
    # every candidate is recorded, present or MISSING, so a file that appears later
    # flips the stage stale
    code = [paths.ROOT / "linearpfn" / f for f in ("_bardist.py", "generator.py", "prior.py")]
    inputs = [CONFIG, pin, sbatch, ckpt, log, slurm_out, *code]
    if ckpt.is_file():
        payload["params"] = modelcard.param_count(ckpt)
    else:
        payload["pending"].append(paths.rel(ckpt))
    if log.is_file():
        payload["train_log"] = modelcard.train_log_stats(log)
    else:
        payload["pending"].append(paths.rel(log))
    if slurm_out.is_file():
        payload["slurm_out"] = modelcard.slurm_out_facts(slurm_out)
    else:
        payload["pending"].append(paths.rel(slurm_out))
    hw = modelcard.hardware(payload["sbatch"], payload["slurm_out"])
    payload["hardware"] = hw
    if payload["train_log"] and hw["gpus"]:
        payload["train_log"]["gpu_hours"] = payload["train_log"]["wall_hours"] * hw["gpus"]
    out = store.write_json(NUM / "model_card.json", payload, inputs, stage="60_model_card")
    print(f"params: {payload['params'] and payload['params']['params']}; pending: "
          f"{payload['pending']}\n-> {paths.rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
