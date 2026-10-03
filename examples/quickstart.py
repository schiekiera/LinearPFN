"""Fit LinearPFN to one simulated dataset and compare with the truth.

    python examples/quickstart.py

Downloads the released weights on first use (about 126 MB, cached under
~/.cache/linearpfn). The dataset has two active main effects and their
interaction, so the median probability model should recover x1, x2 and x1:x2.
"""

from __future__ import annotations

import numpy as np

from linearpfn import LinearPFN


def main() -> int:
    rng = np.random.default_rng(0)
    n, p = 300, 6
    X = rng.normal(size=(n, p))
    y = 1.0 + 0.5 * X[:, 0] - 0.4 * X[:, 1] + 0.3 * X[:, 0] * X[:, 1] + rng.normal(size=n)

    m = LinearPFN.from_pretrained()
    r = m.fit(X, y, holdout=0.2, seed=0)
    print(r.summary())
    print(f"\nselected: {[e for e, s in zip(r.effect_names, r.selected, strict=True) if s]}")
    b0, b = r.coef_raw()
    print(f"raw-scale intercept {b0:.3f}, x1 {b[0]:.3f}, x2 {b[1]:.3f}, "
          f"x1:x2 {b[r.effect_names.index('x1:x2')]:.3f}")
    print(f"held-out NLL {r.heldout['nll']:.3f}, RMSE {r.heldout['rmse']:.3f} "
          "(standardized y)")
    print("predictive 90% intervals of the first three rows:")
    print(m.predict(X[:3], kind="quantiles", levels=(0.05, 0.95)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
