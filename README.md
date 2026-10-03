# LinearPFN

**Bayesian variable selection for linear models with interactions, in one forward pass.**

LinearPFN is a transformer pretrained on a spike-and-slab prior over main effects and
pairwise interactions. Given a dataset, it returns posterior inclusion probabilities and
posterior-mean coefficients, with no MCMC and no per-dataset fitting.

<br>

## Install

```bash
git clone https://github.com/schiekiera/LinearPFN.git
pip install -e LinearPFN
```

The weights (126 MB) are downloaded from
[Hugging Face](https://huggingface.co/schiekiera/LinearPFN) on first use.

<br>

## Use

```python
from linearpfn import LinearPFN

model = LinearPFN.from_pretrained()
result = model.fit(X, y)

print(result.summary())
```

```
effect                    PIP      coef  MPM
--------------------------------------------
complaints              0.996    +0.634  yes
learning                0.561    +0.129  yes
advance                 0.301    -0.018
...
```

<br>

| | |
|---|---|
| `result.pip` | posterior inclusion probability of each effect |
| `result.selected` | the median probability model (PIP > 0.5) |
| `result.coef` | posterior-mean coefficients (standardized scale) |
| `result.coef_raw()` | the same coefficients in the units of your data |
| `model.predict(X_new)` | the posterior predictive for new rows |

<br>

The model covers 2 to 30 predictors and up to 2,000 rows.
Missing values and categorical predictors must be handled first.

<br>

## Reproduce the paper

See [`docs/reproduce.md`](docs/reproduce.md).

<br>

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

<br>

## License

Apache 2.0. The transformer backbone is derived from nanoTabPFN, the optimizer from Muon
(see `NOTICE`).
