# LinearPFN: model card

LinearPFN is a prior-data fitted transformer network for Bayesian variable selection in linear
models with main effects and pairwise interactions. Code, documentation and the paper's
experiments: [github.com/schiekiera/LinearPFN](https://github.com/schiekiera/LinearPFN).

## Model

- **Input:** a dataset of n rows with p predictors and an outcome y (n = 20 to 2,000 and
  p = 2 to 30 during training).
- **Output, in one forward pass:** a posterior inclusion probability for each of the
  p + C(p, 2) candidate effects (selection head), their posterior means (coefficient head),
  and the posterior predictive distribution of y for query rows (a bar distribution).
- **Architecture:** a fork of nanoTabPFN. Per-cell embeddings with alternating attention
  between features and rows, 8 layers, embedding size 512, 16 attention heads, MLP width 1,024,
  31,541,766 parameters, float32.
- **Training data:** synthetic datasets only, drawn from a conjugate spike-and-slab prior.
  Predictor matrices come from factor blocks with mixed marginals (Gaussian copula). Active
  sets follow a mixture of count and rate priors under strong heredity. Coefficients come from
  a slab with a random scale, and the noise is Gaussian. No real data was used in training.
- **Training objective:** predictive NLL, plus binary cross-entropy of the selection head
  against the simulated active sets, plus a squared-error term of the coefficient head against
  the simulated coefficients. `config.yaml` lists every prior, architecture and training
  setting.

## Intended use

The model is meant for variable selection and effect estimation in linear regression with
up to 30 predictors, where interactions are plausible and a posterior over which effects are
active is wanted instead of a single selected subset. Its outputs are the posterior under the
prior it was trained on. Where the data depart from that prior (heredity, coefficient
distribution, noise), they are an approximation whose quality the paper measures for several
departures.

## Evaluation

Wherever the exact posterior can be computed (enumeration over models with quadrature over
the slab scale), the network's PIPs, coefficients and predictive distributions are validated
against it. The paper also reports a benchmark on real predictor matrices against five
classical baselines, and a comparison with an MCMC sampler on real outcomes.

## Limitations

- Main effects and pairwise interactions of numeric predictors only. There are no
  higher-order terms or non-linear effects, no categorical predictors (dummy-code them first)
  and no missing values.
- Outside the trained ranges of n and p the outputs are not validated. The `linearpfn` API
  refuses p > 30 and subsamples n > 2,000.
- Coefficients are on the standardized scale (z-scored predictors, unit-variance outcome).
  The API converts them to raw units.

## Files

| file | content |
|---|---|
| `linearpfn_strong.pt` | weights and configuration (`torch.load(path, weights_only=True)`), no optimizer state |
| `config.yaml` | prior, architecture and training settings |
| `SHA256SUMS` | checksums of the files above |

## Usage

The code is installed from GitHub. The weights are not part of that repository: on first
use, `from_pretrained()` downloads `linearpfn_strong.pt` from this Hugging Face repository,
checks its SHA-256 and caches it under `~/.cache/linearpfn`.

```bash
git clone https://github.com/schiekiera/LinearPFN.git
pip install -e LinearPFN
```

```python
from linearpfn import LinearPFN

m = LinearPFN.from_pretrained()   # downloads the weights from the Hugging Face Hub
r = m.fit(X, y)
print(r.summary())
```

## License and citation

Apache License 2.0. Please cite:

```bibtex
@article{schiekiera2026linearpfn,
  title   = {{LinearPFN}: Amortized Variable Selection for Linear Models with Interactions},
  author  = {Schiekiera, Louis and Zimmer, Max and Roux, Christophe and Arnold, Manuel
             and Pokutta, Sebastian and G{\"u}nther, Fritz},
  journal = {arXiv preprint},
  year    = {2026}
}
```
