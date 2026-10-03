# LinearPFN

**Bayesian variable selection for linear models with interactions, in one forward pass.**

A transformer pretrained on a spike-and-slab prior over main effects and pairwise
interactions. Given a dataset, it returns posterior inclusion probabilities and
posterior-mean coefficients, with no MCMC and no per-dataset fitting.

Code and documentation: [github.com/schiekiera/LinearPFN](https://github.com/schiekiera/LinearPFN)


## Use

```bash
git clone https://github.com/schiekiera/LinearPFN.git
pip install -e LinearPFN
```

```python
from linearpfn import LinearPFN

model = LinearPFN.from_pretrained()   # downloads this model's weights
result = model.fit(X, y)

print(result.summary())
```


## Model

| | |
|---|---|
| Input | n rows, p predictors, outcome y (trained on p = 2 to 30, n = 20 to 2,000) |
| Output | a PIP and a posterior mean for each of the p + p(p-1)/2 effects, and a predictive distribution |
| Architecture | nanoTabPFN-style transformer, 8 layers, 31.5M parameters, fp32 |
| Training data | synthetic only, drawn from the prior in `config.yaml` |

<br>

## Limitations

The outputs are the posterior under the training prior: linear main effects, pairwise
interactions, strong heredity, Gaussian noise. Missing values and categorical predictors
must be handled before fitting.


## Files

| | |
|---|---|
| `linearpfn_strong.pt` | weights and configuration (`torch.load(..., weights_only=True)`) |
| `config.yaml` | prior, architecture and training settings |
| `SHA256SUMS` | checksums of the two files above |

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

License: Apache 2.0.
