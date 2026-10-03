"""User-facing API: Bayesian variable selection and coefficients in one forward pass.

    from linearpfn import LinearPFN

    m = LinearPFN.from_pretrained()          # the released weights (Hugging Face Hub)
    r = m.fit(X, y)                          # all rows are context
    print(r.summary())                       # PIPs, posterior means, median probability model
    r.pip, r.selected, r.coef, r.coef_mpm, r.effect_names
    m.predict(X_new, kind="quantiles")       # optional: the posterior predictive

The candidate effects are the p main effects and all C(p, 2) pairwise
interactions. The model approximates the posterior under the spike-and-slab
prior it was trained on (strong heredity: an interaction can only be active
when both of its main effects are). It was trained on p in [2, 30] and
n in [20, 2000] rows. Preprocessing follows the paper: predictors z-scored
(ddof=0), y centred and scaled to unit sd, coefficients reported on that
standardized scale (`FitResult.coef_raw` converts). Missing values and
categorical predictors are out of scope: impute or dummy-code first.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

from linearpfn import hub
from linearpfn._fitresult import FitResult, Predictive
from linearpfn.prior import interaction_pairs
from linearpfn.train import build_model, make_prior_config

__all__ = ["FitResult", "LinearPFN", "Predictive"]

P_MIN, P_MAX = 2, 30
N_MIN, N_WARN = 10, 20
QUERY_FRAC, MIN_QUERY = 0.2, 5
FORMAT_VERSION = 1


def zscore(a: np.ndarray, stats: tuple | None = None) -> tuple[np.ndarray, tuple]:
    """(a - mean) / sd with ddof=0; `stats` reuses another block's moments."""
    mu, sd = stats if stats is not None else (a.mean(0), a.std(0))
    return (a - mu) / sd, (mu, sd)


def split_rows(n: int, seed: int, frac: float = QUERY_FRAC) -> tuple[np.ndarray, np.ndarray]:
    """(context rows, query rows): max(5, round(frac n)) query rows, seeded."""
    nq = max(MIN_QUERY, int(round(frac * n)))
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[nq:]), np.sort(perm[:nq])


def _as_matrix(X, feature_names) -> tuple[np.ndarray, list[str]]:
    names = feature_names
    if names is None and hasattr(X, "columns"):
        names = [str(c) for c in X.columns]
    try:
        A = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("X must be numeric; dummy-code categorical predictors first") from exc
    if A.ndim != 2:
        raise ValueError(f"X must be 2-dimensional (n rows, p predictors), got shape {A.shape}")
    names = [f"x{j + 1}" for j in range(A.shape[1])] if names is None else list(names)
    if len(names) != A.shape[1] or len(set(names)) != len(names):
        raise ValueError(f"feature_names must be {A.shape[1]} unique names")
    if not np.isfinite(A).all():
        raise ValueError("X contains NaN or infinite values; missing data is out of scope "
                         "(impute or drop rows first)")
    return A, names


def _as_vector(y, n: int) -> np.ndarray:
    try:
        v = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("y must be numeric") from exc
    if v.ndim == 2 and v.shape[1] == 1:
        v = v[:, 0]
    if v.ndim != 1 or v.shape[0] != n:
        raise ValueError(f"y must be a vector of length {n}, got shape {v.shape}")
    if not np.isfinite(v).all():
        raise ValueError("y contains NaN or infinite values")
    return v


class LinearPFN:
    """A pretrained LinearPFN network with the paper's preprocessing around it."""

    def __init__(self, model, bar, prior_cfg, config: dict, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.bar = bar.to(self.device)
        self.prior_cfg = prior_cfg
        self.config = config
        self._context: tuple | None = None

    # ------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, path_or_repo: str | Path | None = None, *,
                        revision: str | None = None, filename: str | None = None,
                        device: str = "auto", token: str | None = None) -> LinearPFN:
        """Load weights from a local file, or download them from the Hugging Face Hub.

        path_or_repo: a local `.pt` file (the released weights or a training
        checkpoint), a Hub repository id, or None for the released model at its
        pinned revision (verified against its SHA-256).
        device: "auto" (CUDA if available, else CPU), "cpu", "cuda", ...
        """
        if path_or_repo is not None and Path(path_or_repo).expanduser().is_file():
            path = Path(path_or_repo).expanduser()
        else:
            repo = str(path_or_repo or hub.REPO_ID)
            name, rev = filename or hub.FILENAME, revision or hub.REVISION
            pinned = (repo, name, rev) == (hub.REPO_ID, hub.FILENAME, hub.REVISION)
            path = hub.download(repo, name, rev, sha256=hub.SHA256 if pinned else None,
                                token=token)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or "model" not in state or "config" not in state:
            raise ValueError(f"{path}: not a LinearPFN weights file")
        config = state["config"]
        prior_cfg = make_prior_config(config.get("data", {}))
        model, bar = build_model(config["model"], prior_cfg.include_interactions)
        model.load_state_dict(state["model"])
        return cls(model, bar, prior_cfg, config, device=device)

    def save(self, path: str | Path, config: dict | None = None) -> Path:
        """Write the weights-only file `from_pretrained` reads (plain types, loadable
        with torch.load(weights_only=True))."""
        state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        path = Path(path)
        torch.save({"model": state, "config": config or self.config,
                    "format_version": FORMAT_VERSION}, path)
        return path

    # ------------------------------------------------------------ fitting
    def _forward(self, X_ctx: np.ndarray, y_ctx: np.ndarray, X_q: np.ndarray | None = None):
        X_all = X_ctx if X_q is None else np.vstack([X_ctx, X_q])
        X_t = torch.from_numpy(X_all).float().unsqueeze(0).to(self.device)
        y_t = torch.from_numpy(y_ctx).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model.forward_with_heads(X_t, y_t, len(y_ctx))

    def _cap(self, rows: np.ndarray, max_context: int, seed: int) -> np.ndarray:
        if len(rows) <= max_context:
            return rows
        warnings.warn(f"{len(rows)} context rows exceed max_context = {max_context} (the "
                      "model was trained on n <= 2000); using a seeded random subsample",
                      stacklevel=3)
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(rows, size=max_context, replace=False))

    def _prepare(self, A: np.ndarray, v: np.ndarray, names: list[str], scale_y: bool):
        n = len(v)
        if n < N_MIN:
            raise ValueError(f"{n} context rows: at least {N_MIN} are needed")
        if n < N_WARN:
            warnings.warn(f"{n} context rows: the model was trained on n >= {N_WARN}",
                          stacklevel=3)
        const = [names[j] for j in np.flatnonzero(~(A.std(0) > 0))]
        if const:
            raise ValueError(f"constant predictor(s) in the context rows: {const}")
        X, xs = zscore(A)
        if scale_y:
            if not v.std() > 0:
                raise ValueError("y is constant")
            y, (mu, sd) = zscore(v)
        else:
            y, mu, sd = v, 0.0, 1.0
        return X, xs, y, float(mu), float(sd)

    def fit(self, X, y, feature_names: list[str] | None = None, *, holdout: float | None = None,
            seed: int = 0, max_context: int = 2000, coef_source: str = "head",
            scale_y: bool = True) -> FitResult:
        """Posterior inclusion probabilities and coefficients, all rows as context.

        holdout: a fraction (e.g. 0.2) of rows held out for a check of the
            posterior predictive on unseen rows (max(5, round(holdout n)) rows,
            seeded with `seed`). A second, separate pass on the remaining rows
            scores them; `pip` and `coef` still use every row.
        max_context: rows beyond it are subsampled (seeded); the model was trained
            on at most 2000.
        coef_source: "head" (the coefficient head, as in the paper) or "probe"
            (least squares on the predictive mean at jittered probe points).
        scale_y: False passes y as given; use only when y is already on unit scale.
        """
        A, names = _as_matrix(X, feature_names)
        v = _as_vector(y, A.shape[0])
        p = A.shape[1]
        if not P_MIN <= p <= P_MAX:
            raise ValueError(f"p = {p} predictors: the model covers {P_MIN} to {P_MAX}")
        if coef_source not in ("head", "probe"):
            raise ValueError(f"coef_source must be 'head' or 'probe', got {coef_source!r}")
        rows = self._cap(np.arange(len(v)), max_context, seed)
        X_ctx, xs, y_ctx, mu, sd = self._prepare(A[rows], v[rows], names, scale_y)
        _, sel, coef_out = self._forward(X_ctx, y_ctx)
        pip = torch.sigmoid(sel.float())[0].cpu().numpy()
        intercept = float(y_ctx.mean())  # no head models it; X is centred
        if coef_source == "head":
            if coef_out is None:
                raise RuntimeError("this checkpoint has no coefficient head; "
                                   "use coef_source='probe'")
            with torch.no_grad():
                eff = self.model.coefficient_posterior_mean(sel[0], coef_out[0])
            coef = eff.cpu().numpy().astype(float)
            source = f"head:{self.model.coef_head_mode}"
        else:
            intercept, coef = self._probe(X_ctx, y_ctx, seed)
            source = "probe"
        pairs = interaction_pairs(p, self.prior_cfg)
        res = FitResult(
            effect_names=names + [f"{names[i]}:{names[j]}" for i, j in pairs],
            pip=pip, coef=coef, intercept=intercept, x_mean=np.asarray(xs[0]),
            x_sd=np.asarray(xs[1]), y_mean=mu, y_sd=sd, n_context=len(rows),
            coef_source=source, pairs=pairs)
        if holdout is not None:
            res.heldout = self._heldout(A, v, names, holdout, seed, max_context, scale_y)
        self._context = (X_ctx, y_ctx, xs, mu, sd)
        return res

    def _probe(self, X_ctx: np.ndarray, y_ctx: np.ndarray, seed: int) -> tuple[float, np.ndarray]:
        from linearpfn.probe import extract_coefficients

        n = len(y_ctx)
        X_t = torch.from_numpy(X_ctx).float().to(self.device)
        y_t = torch.from_numpy(y_ctx).float().unsqueeze(0).to(self.device)

        def mean_fn(probes: np.ndarray) -> np.ndarray:
            Xp = torch.cat([X_t, torch.from_numpy(probes).float().to(self.device)]).unsqueeze(0)
            with torch.no_grad():
                return self.bar.mean(self.model(Xp, y_t, n).float())[0].cpu().numpy()

        res = extract_coefficients(np.random.default_rng(seed), X_ctx, mean_fn,
                                   cfg=self.prior_cfg)
        return float(res.coef[0]), np.asarray(res.coef[1:], dtype=float)

    def _heldout(self, A, v, names, frac, seed, max_context, scale_y) -> dict:
        if not 0 < frac < 1:
            raise ValueError("holdout must be a fraction in (0, 1)")
        ctx, q = split_rows(len(v), seed, frac)
        ctx = self._cap(ctx, max_context, seed)
        X_ctx, xs, y_ctx, mu, sd = self._prepare(A[ctx], v[ctx], names, scale_y)
        X_q, _ = zscore(A[q], xs)
        y_q = (v[q] - mu) / sd
        logits, _, _ = self._forward(X_ctx, y_ctx, X_q)
        lg = logits.float()
        with torch.no_grad():
            nll = float(self.bar.nll(lg, torch.from_numpy(y_q).float().unsqueeze(0)
                                     .to(self.device)).mean())
            pred = self.bar.mean(lg)[0].cpu().numpy()
        rmse = float(np.sqrt(np.mean((y_q - pred) ** 2)))
        return {"nll": nll, "rmse": rmse, "nll_raw": nll + float(np.log(sd)),
                "rmse_raw": rmse * sd, "n_context": len(ctx), "n_query": len(q),
                "query_rows": q}

    # ------------------------------------------------------------ prediction
    def predict(self, X_new, kind: str = "mean", levels=(0.05, 0.5, 0.95)):
        """Posterior predictive for new rows, given the rows of the last `fit`.

        kind: "mean" -> (m,), "quantiles" -> (m, len(levels)), "distribution" ->
        a `Predictive` (mean, quantile, cdf, nll), all in the units of y.
        """
        if self._context is None:
            raise RuntimeError("call fit(X, y) before predict")
        X_ctx, y_ctx, xs, mu, sd = self._context
        A, _ = _as_matrix(X_new, [f"x{j + 1}" for j in range(np.shape(X_new)[-1])])
        if A.shape[1] != X_ctx.shape[1]:
            raise ValueError(f"X_new has {A.shape[1]} columns, the fit had {X_ctx.shape[1]}")
        X_q, _ = zscore(A, xs)
        logits, _, _ = self._forward(X_ctx, y_ctx, X_q)
        dist = Predictive(self.bar, logits[0], mu, sd)
        if kind == "distribution":
            return dist
        if kind == "mean":
            return dist.mean()
        if kind == "quantiles":
            return dist.quantile(levels)
        raise ValueError(f"kind must be 'mean', 'quantiles' or 'distribution', got {kind!r}")
