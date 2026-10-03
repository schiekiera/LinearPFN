"""Result types of the user-facing API (`linearpfn.api`): the fit and the predictive.

Private helper of `linearpfn.api`, which re-exports both classes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class FitResult:
    """Posterior summaries of one dataset, one entry per candidate effect.

    Effects are the p main effects, then the C(p, 2) pairwise interactions in
    lexicographic order (`effect_names`). All coefficients are on the
    STANDARDIZED scale the model works on: predictors z-scored (ddof=0) and y
    centred and scaled to unit sd (unless the fit used `scale_y=False`).
    `coef_raw` converts to the units of the data.

    pip        posterior inclusion probability of each effect
    selected   the median probability model: PIP > 0.5
    coef       posterior mean E[beta | data], averaged over models (dense: an
               effect outside the median probability model keeps its shrunken mean)
    coef_mpm   the same vector with every effect outside the median probability
               model set to zero
    intercept  the context mean of the scaled y (no head models the intercept; the
               probe read-out estimates it)
    heldout    with `fit(..., holdout=...)`: NLL and RMSE of the posterior predictive
               on the held-out rows (standardized y units; `*_raw` in y units)
    """

    effect_names: list[str]
    pip: np.ndarray
    coef: np.ndarray
    intercept: float
    x_mean: np.ndarray
    x_sd: np.ndarray
    y_mean: float
    y_sd: float
    n_context: int
    coef_source: str
    pairs: list[tuple[int, int]] = field(repr=False)
    heldout: dict | None = None

    @property
    def selected(self) -> np.ndarray:
        return self.pip > 0.5

    @property
    def coef_mpm(self) -> np.ndarray:
        return np.where(self.selected, self.coef, 0.0)

    def summary(self, sort: str = "pip", digits: int = 3) -> str:
        """Plain-text table of every effect: PIP, posterior mean, membership in the
        median probability model. sort: "pip" (descending) or "order"."""
        order = np.argsort(-self.pip, kind="stable") if sort == "pip" else range(len(self.pip))
        w = max(len("effect"), *(len(n) for n in self.effect_names))
        lines = [f"{'effect':<{w}}  {'PIP':>{digits + 3}}  {'coef':>{digits + 5}}  MPM",
                 "-" * (w + digits * 2 + 17)]
        for j in order:
            lines.append(f"{self.effect_names[j]:<{w}}  {self.pip[j]:>{digits + 3}.{digits}f}  "
                         f"{self.coef[j]:>+{digits + 5}.{digits}f}  "
                         f"{'yes' if self.selected[j] else ''}")
        lines.append(f"(n = {self.n_context} context rows, coefficients on the standardized "
                     f"scale, source: {self.coef_source})")
        return "\n".join(lines)

    def coef_raw(self, which: str = "dense", centered: bool = True) -> tuple[float, np.ndarray]:
        """(intercept, coefficients) in the units of X and y.

        centered=True: slopes per unit of each predictor, with every predictor
        centred at its context mean. Main effects are then the slopes at the
        means of the other predictors, and only the scaling changes.
        centered=False: the same surface written in the uncentred predictors.
        Once interactions are present, the main effects (and the intercept) then
        depend on where zero lies on each predictor's scale: beta_j picks up
        -sum_k beta_jk * mean_k. Report the centred form unless zero is meaningful.
        """
        b = {"dense": self.coef, "mpm": self.coef_mpm}[which]
        p = len(self.x_sd)
        a = np.empty_like(b, dtype=float)
        a[:p] = self.y_sd * b[:p] / self.x_sd
        for k, (i, j) in enumerate(self.pairs):
            a[p + k] = self.y_sd * b[p + k] / (self.x_sd[i] * self.x_sd[j])
        b0 = self.y_mean + self.y_sd * self.intercept
        if centered:
            return float(b0), a
        m = self.x_mean
        mains = a[:p].copy()
        b0 -= float(a[:p] @ m)
        for k, (i, j) in enumerate(self.pairs):
            mains[i] -= a[p + k] * m[j]
            mains[j] -= a[p + k] * m[i]
            b0 += float(a[p + k] * m[i] * m[j])
        return float(b0), np.concatenate([mains, a[p:]])


class Predictive:
    """Posterior predictive distribution of the query rows, in the units of y.

    Wraps the model's bar distribution (a histogram over [-8, 8] on the
    standardized scale with half-normal tails) and the scaling of y.
    """

    def __init__(self, bar, logits: torch.Tensor, y_mean: float, y_sd: float) -> None:
        self._bar, self._logits = bar, logits.float()
        self.y_mean, self.y_sd = float(y_mean), float(y_sd)

    def __len__(self) -> int:
        return self._logits.shape[0]

    def mean(self) -> np.ndarray:
        with torch.no_grad():
            m = self._bar.mean(self._logits).cpu().numpy().astype(float)
        return self.y_mean + self.y_sd * m

    def quantile(self, levels=(0.05, 0.5, 0.95)) -> np.ndarray:
        """(n_query, len(levels)) quantiles."""
        lv = torch.as_tensor(np.asarray(levels, dtype=np.float32), device=self._logits.device)
        with torch.no_grad():
            q = self._bar.quantile(self._logits, lv).cpu().numpy().astype(float)
        return self.y_mean + self.y_sd * q

    def _z(self, y) -> torch.Tensor:
        z = (np.asarray(y, dtype=float).reshape(-1) - self.y_mean) / self.y_sd
        if z.shape[0] != len(self):
            raise ValueError(f"y has {z.shape[0]} values for {len(self)} query rows")
        return torch.from_numpy(z.astype(np.float32)).to(self._logits.device)

    def cdf(self, y) -> np.ndarray:
        """P(Y <= y_i) for each query row i."""
        with torch.no_grad():
            c = self._bar.cdf(self._logits, self._z(y).unsqueeze(-1))
        return c[..., 0].cpu().numpy().astype(float)

    def nll(self, y) -> np.ndarray:
        """Negative log density of y_i for each query row (nats, units of y)."""
        with torch.no_grad():
            v = self._bar.nll(self._logits, self._z(y)).cpu().numpy().astype(float)
        return v + math.log(self.y_sd)
