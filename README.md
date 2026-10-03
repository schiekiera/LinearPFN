# LinearPFN

**Amortized variable selection for linear models with interactions.**

Louis Schiekiera<sup>1,2,3</sup>, Max Zimmer<sup>3,4</sup>, Christophe Roux<sup>3,4</sup>,
Manuel Arnold<sup>1</sup>, Sebastian Pokutta<sup>3,4</sup> and Fritz Günther<sup>1</sup>

<sup>1</sup>Institute of Psychology, Humboldt-Universität zu Berlin, Berlin, Germany ·
<sup>2</sup>Department of Education and Psychology, Freie Universität Berlin, Berlin, Germany ·
<sup>3</sup>Department for AI in Society, Science, and Technology, Zuse Institute Berlin, Germany ·
<sup>4</sup>Institute of Mathematics, Technische Universität Berlin, Germany

Paper: arXiv (link follows) · Weights: [huggingface.co/schiekiera/LinearPFN](https://huggingface.co/schiekiera/LinearPFN) ·
Contact: Louis Schiekiera, louis.schiekiera@hu-berlin.de

LinearPFN is a transformer pretrained on synthetic datasets drawn from a conjugate
spike-and-slab prior over main effects and pairwise interactions. One forward pass over a new
dataset returns posterior inclusion probabilities (PIPs), posterior-mean coefficients and the
posterior predictive distribution, with no per-dataset fitting and no MCMC. Wherever the exact
posterior can still be computed (enumeration over models, quadrature over the slab scale), the
network's outputs are validated against it.

## Install

    git clone https://github.com/schiekiera/LinearPFN.git
    cd LinearPFN
    pip install -e .

Python 3.11 or newer with numpy, scipy, PyYAML and torch. The weights (126 MB) are downloaded
from the Hugging Face Hub on first use and cached under `~/.cache/linearpfn`. Installing
`huggingface_hub` (`pip install -e .[hub]`) is optional.

## Quickstart

```python
from linearpfn import LinearPFN

m = LinearPFN.from_pretrained()     # the released weights, verified against their SHA-256
r = m.fit(X, y)                     # X: (n, p) array or data frame, y: (n,)
print(r.summary())
```

On R's `attitude` data (y = overall rating, six predictors) the summary begins:

```
effect                    PIP      coef  MPM
--------------------------------------------
complaints              0.996    +0.634  yes
learning                0.561    +0.129  yes
advance                 0.301    -0.018
...
```

`examples/quickstart.py` runs the same steps on a simulated dataset.

| attribute | meaning |
|---|---|
| `r.effect_names` | the p main effects, then all C(p, 2) pairwise interactions `a:b` |
| `r.pip` | posterior inclusion probability of each effect |
| `r.selected` | the median probability model: effects with PIP > 0.5 |
| `r.coef` | posterior mean E[β \| data], averaged over models (every effect keeps its shrunken mean) |
| `r.coef_mpm` | the same vector, set to zero outside the median probability model |
| `r.coef_raw()` | intercept and coefficients in the units of X and y (see below) |

**Scale.** As in the paper, the predictors are z-scored (ddof = 0) and y is centred and scaled
to unit standard deviation before the forward pass, so `r.coef` is on that standardized scale.
`r.coef_raw()` converts to the units of the data, with each predictor centred at its mean, so a
main effect is the slope at the means of the other predictors. `r.coef_raw(centered=False)`
writes the same surface in the uncentred predictors. Once interactions are present, its main
effects then depend on where zero lies on each predictor's scale.

**Options.**
- `m.fit(X, y, holdout=0.2, seed=0)` holds out 20 % of the rows (at least five) and scores the
  posterior predictive on them (`r.heldout`: NLL and RMSE). PIPs and coefficients still use every
  row.
- `m.fit(X, y, coef_source="probe")` reads the coefficients off the predictive mean at jittered
  probe points instead of the coefficient head.
- `m.predict(X_new, kind="mean" | "quantiles" | "distribution")` returns the posterior
  predictive for new rows given the rows of the last fit, in the units of y.
- `LinearPFN.from_pretrained("path/to/weights.pt", device="cpu")` loads a local file.

## Scope

- The model approximates the posterior under the prior it was trained on: linear main effects
  and pairwise interactions, Gaussian noise, strong heredity (an interaction can only be active
  when both of its main effects are). Its PIPs and coefficients are statements under that
  prior.
- It was trained on p = 2 to 30 predictors and n = 20 to 2,000 rows. `fit` refuses p outside
  that range and fewer than 10 rows, warns below 20 rows, and uses a seeded random subsample of
  2,000 rows (`max_context`) for larger datasets.
- Missing values and categorical predictors are out of scope: impute or drop rows, and
  dummy-code categorical predictors first.
- On a CPU a fit takes well under a second. GPU results agree with CPU results to float
  precision.

## Repository layout

| path | what it does | paper |
|---|---|---|
| `linearpfn/api.py`, `_fitresult.py`, `hub.py` | the user-facing API above and the weight download | none |
| `linearpfn/prior.py`, `_xgen.py`, `_counts.py` | the prior: predictor matrices (factor blocks, Gaussian copula), active effects (count, rate and mixture priors, heredity), coefficients, noise | Sec. 3.1, App. B |
| `linearpfn/generator.py` | batched training datasets drawn from the prior | Sec. 3.2 |
| `linearpfn/model.py`, `_heads.py`, `_bardist.py`, `_pairstats.py` | the network: transformer backbone, predictive (bar-distribution), selection and coefficient heads | Sec. 3.2, App. C |
| `linearpfn/train.py`, `_optim.py`, `_trainutil.py`, `_ddp.py`, `_evalgate.py` | loss (predictive NLL + selection BCE + coefficient term) and training loop | Sec. 3.2, App. C |
| `linearpfn/reference.py`, `_quadrature.py` | exact posterior: enumeration over the model space, quadrature over the slab scale | App. D |
| `linearpfn/mc3.py`, `mc3_general.py` | MCMC sampler over the model space | App. D |
| `linearpfn/validate.py`, `_refcache.py`, `_valreport.py`, `probe.py` | validation against the exact posterior | Sec. 3.3, App. E |
| `linearpfn/preflight.py` | diagnostics of the prior (effect sizes, R2, null share) | App. B |
| `bench/` | benchmark harness, LinearPFN and the baselines (lasso, stability selection, SuSiE, glinternet, hierNet), outcomes outside the prior | App. F, App. G |
| `data_prep/real_designs/` | seeded selection and cleaning of the real predictor matrices from the Rdatasets collection (R) | App. F |
| `scripts/NN_*.py` | the experiments and every number, table and figure of the paper | see `docs/reproduce.md` |
| `pipeline/` | shared code of the number, table and figure stages | none |
| `configs/` | the reported model (`paper_model.yaml`), a small CPU configuration (`smoke.yaml`), the stage settings (`pipeline.yaml`) | App. B, App. C |
| `tests/`, `smoke_test.py` | tests of the API; an end-to-end check of the method | none |

## Reproducing the paper

`docs/reproduce.md` lists the commands, from the real predictor matrices to every number,
table and figure. A quick check that everything runs:

    pip install -r requirements.txt
    python smoke_test.py

## Data

The repository ships no data. The real predictor matrices are public tables from the
[Rdatasets](https://vincentarelbundock.github.io/Rdatasets/) collection.
`data_prep/real_designs/` reproduces their selection and cleaning.
`data_prep/real_designs/expected/` holds the outputs of our run (with each table's source URL
and license), and `data_prep/SHA256SUMS` holds the checksums of the matrices and of the seven
real-outcome tables, so a copy can be checked against ours:

    shasum -a 256 -c data_prep/SHA256SUMS

## Citation

```bibtex
@article{schiekiera2026linearpfn,
  title   = {{LinearPFN}: Amortized Variable Selection for Linear Models with Interactions},
  author  = {Schiekiera, Louis and Zimmer, Max and Roux, Christophe and Arnold, Manuel
             and Pokutta, Sebastian and G{\"u}nther, Fritz},
  journal = {arXiv preprint},
  year    = {2026}
}
```

`CITATION.cff` has the same entry.

## License

Apache License 2.0 (`LICENSE`). The transformer backbone in `linearpfn/model.py` is derived from
nanoTabPFN (Apache License 2.0), and the optimizer in `linearpfn/_optim.py` follows Muon (MIT
License). `NOTICE` and `LICENSES/` have the details.
