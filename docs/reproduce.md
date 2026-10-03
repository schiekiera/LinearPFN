# Reproducing the results

Every number, table and figure of the paper is produced by a numbered stage in
`scripts/`. The final stages read only finished outputs and write LaTeX macros, so no
number in the paper is typed by hand.

## Requirements

- **Python 3.12** (tested with 3.12.7): torch 2.10.0, numpy 1.26.4, scipy 1.13.1, PyYAML
  6.0.1, pandas 2.2.2, scikit-learn 1.5.1, matplotlib 3.10.0.

      pip install -r requirements.txt

- **R** (for the R baselines only): R 4.5.1 with susieR 0.14.2, glinternet 1.0.13 and
  hierNet 1.9.
- **Hardware:** the reported model was trained on two GPUs. Everything else runs on a CPU,
  and the benchmark timings use a single CPU thread.

## Quick check

    python smoke_test.py

This draws one dataset from the reported prior, runs one forward pass of a randomly
initialized network with the reported architecture, and computes the exact posterior of the
same dataset at p = 3. It takes about a minute on a CPU.

    python -m linearpfn.train --config configs/smoke.yaml

This trains a small network on a CPU in a few minutes.

## The trained model

The reported model's weights are on the Hugging Face Hub (`schiekiera/LinearPFN`). The
commands below expect them at `checkpoints/paper_model/ckpt_final.pt`:

    mkdir -p checkpoints/paper_model
    python -c "import shutil; from linearpfn import hub; \
        shutil.copy(hub.download(sha256=hub.SHA256), 'checkpoints/paper_model/ckpt_final.pt')"

The released file holds the weights and the configuration, without optimizer state. Every
stage that loads a checkpoint reads both.

## Commands

The commands below use the default paths of `configs/pipeline.yaml`. Datasets with a common
design are split into shards (`--shard i/N`, `--cell i`), so each command can run as a job
array.

    # the real predictor matrices: run the R pipeline from inside data_prep/real_designs/,
    # then point data/real_designs at its outputs
    (cd data_prep/real_designs && Rscript run_all.R && Rscript 10_deliver_eligible.R)
    mkdir -p data && ln -s ../data_prep/real_designs data/real_designs
    shasum -a 256 -c data_prep/SHA256SUMS      # compare with our copies

    # the seven real-outcome tables go to data/real_outcomes/<name>.csv (the predictor
    # columns, then a column named y); names and sources are in configs/pipeline.yaml
    # (real_y.datasets)

    # the model
    python -m linearpfn.preflight --config configs/paper_model.yaml --out reports/preflight
    python -m linearpfn.train --config configs/paper_model.yaml
    python -m linearpfn.validate --config configs/paper_model.yaml \
        --ckpt checkpoints/paper_model/ckpt_final.pt --out reports/validation
    python scripts/38_compute_quadrature.py --config configs/paper_model.yaml \
        --n-datasets 12 --out reports/quadrature

    # the main benchmark and its variants
    python scripts/15_gen_main_panel.py --config configs/paper_model.yaml \
        --out-dir results/main_benchmark/truth
    python scripts/16_gen_knob_main.py --config configs/paper_model.yaml
    python scripts/17_fit_benchmark.py --config configs/paper_model.yaml \
        --panels mx --truth-dir results/main_benchmark/truth --out-dir results/main_benchmark \
        --methods susie glinternet hierNet lasso stability
    python scripts/17_fit_benchmark.py --config configs/paper_model.yaml \
        --panels mx --truth-dir results/main_benchmark/truth --out-dir results/main_benchmark \
        --methods exact --p-cap 5
    python scripts/17_fit_benchmark.py --config configs/paper_model.yaml \
        --panels mx --truth-dir results/main_benchmark/truth --out-dir results/main_benchmark \
        --methods linearpfn_strong --slot-name linearpfn_strong \
        --slot-ckpt checkpoints/paper_model/ckpt_final.pt --slot-cfg configs/paper_model.yaml
    # the same three commands with --panels xc xi xn --truth-dir results/misspecification/truth
    # --out-dir results/misspecification fit the variants
    python scripts/19_stability_refit.py
    python scripts/20_mcmc_benchmark.py --dir results/main_benchmark
    python scripts/72_fit_time_gpu.py --ckpt checkpoints/paper_model/ckpt_final.pt \
        --cfg configs/paper_model.yaml --truth-dir results/main_benchmark/truth --panel ipmx \
        --out results/main_benchmark/fit_time_gpu.json

    # real outcomes
    python scripts/18_real_y_fit.py

    # numbers, tables, figures and macros
    for s in 30 40 42 50 60 70 73 74 75 76 78 80 99; do python scripts/${s}_*.py; done

Stages write their numbers to `reports/paper/`. They write tables, figures and macros to
`paper/`.

## The stages in `scripts/`

**Experiments**

| stage | what it does |
|---|---|
| 15 | builds the main benchmark: outcomes drawn from the prior on the real predictor matrices, balanced over realized density and R2 cells (App. F) |
| 16 | builds its three variants with outcomes outside the prior (App. G) |
| 17 | fits every method on the saved datasets |
| 18 | runs the real-outcome comparison against the MCMC sampler (real-outcome table) |
| 19 | adds least-squares coefficients to stability selection |
| 20 | runs the MCMC sampler on a random subset of the main benchmark |
| 38 | checks convergence of the quadrature node counts (App. D) |
| 72 | times the forward pass on a GPU |

**Numbers**

| stage | what it computes |
|---|---|
| 30 | validation (criteria table, App. E) |
| 40 | prior numbers (App. B) |
| 50 | the real predictor matrices (App. F) |
| 60 | model card (model-card and prior tables) |
| 70 | main benchmark (Sec. 4 summary table, App. G) |
| 73 | GPU fit time (Sec. 4) |
| 74 | outcomes outside the prior (App. G table) |
| 76 | real outcomes |

**Figures and output files**

| stage | what it draws or writes |
|---|---|
| 42 | pretraining mass figure |
| 75 | outcomes-outside-the-prior figure |
| 78 | main-benchmark figure |
| 80 | LaTeX tables |
| 99 | every number used in the text, written as a LaTeX macro |

## Identifiers in file and row names

| identifier | meaning |
|---|---|
| `mx` | the main benchmark (outcomes drawn from the prior, real predictor matrices) |
| `xc`, `xi`, `xn` | its variants: fixed-magnitude coefficients, interaction-heavy outcomes, heavy-tailed noise |
| `ip` | prefix of the main benchmark's numbers |
| `knob` | prefix of the variants' numbers |
| `real_y` | prefix of the real-outcome comparison |
| `linearpfn_strong` | the reported model's row; the model is trained under the strong-heredity prior |
| `linearpfn_strong_nonll` | the ablation trained without the predictive loss |
| `linearpfn_strong_hz` | the reported model's coefficients hard-zeroed outside its median probability model |
| `exact` | reference row: the exact posterior, for p <= 5 |
| `mc3` | reference row: the MCMC sampler |
| `stability_refit` | stability selection with least-squares coefficients on its selected effects |
