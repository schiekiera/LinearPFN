"""Evaluation of a model during and after training (private helper of train).

Mean query NLL of the model vs the per-dataset Gaussian marginal baseline vs
the exact BMA reference on freshly sampled small-p datasets, plus the
selection AUC. Re-exported through `linearpfn.train`.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch

from linearpfn.generator import sample_batch
from linearpfn.model import BarDistribution, LinearPFNModel
from linearpfn.reference import fit_exact

__all__ = ["evaluate_marginal_gate"]


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the rank-sum (Mann-Whitney) statistic; numpy only."""
    pos = labels.astype(bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.empty(scores.size)
    ranks[np.argsort(scores)] = np.arange(1, scores.size + 1)
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate_marginal_gate(
    model: LinearPFNModel, bar: BarDistribution, config: dict, device: torch.device
) -> dict[str, float]:
    """Predictive check plus headroom: mean query NLL of the model, of the per-dataset
    Gaussian marginal baseline (mean/variance of context y, ddof=0), and of
    the exact BMA reference, on freshly sampled datasets with p <= eval.p_max.

    "gap_closed" = (marginal - model) / (marginal - exact): the fraction of
    the available headroom between the X-blind floor and the exact posterior
    that the model has closed.
    """
    from linearpfn.train import make_prior_config  # lazy: avoids the import cycle

    eval_cfg = config["eval"]
    prior_cfg = replace(make_prior_config(config["data"]), p_max=eval_cfg["p_max"])
    model.eval()
    model_nll_sum = baseline_nll_sum = exact_nll_sum = 0.0
    n_points = 0
    sel_probs: list[np.ndarray] = []
    sel_truth: list[np.ndarray] = []
    with torch.no_grad():
        for index in range(eval_cfg["n_datasets"]):
            batch = sample_batch(prior_cfg, 1, eval_cfg["seed"], index)
            split = batch.split_index
            X = batch.X.to(device)
            y = batch.y.to(device)
            y_ctx, y_query = y[:, :split], y[:, split:]
            logits, sel = model.forward_with_selection(X, y_ctx, split)
            sel_probs.append(torch.sigmoid(sel.float())[0].cpu().numpy())
            sel_truth.append(batch.gamma[0, 1:].numpy())
            model_nll_sum += bar.nll(logits.float(), y_query).double().sum().item()
            m = y_ctx.mean()
            v = y_ctx.var(correction=0).clamp_min(1e-12)
            baseline = 0.5 * torch.log(2 * math.pi * v) + (y_query - m) ** 2 / (2 * v)
            baseline_nll_sum += baseline.double().sum().item()
            X_np = batch.X[0].double().numpy()
            y_np = batch.y[0].double().numpy()
            # Context slices of full-dataset-standardized X deviate from
            # exact standardization by O(1/sqrt(n_ctx)); loosen the drift
            # guard accordingly (it still catches raw unstandardized data).
            post = fit_exact(
                X_np[:split], y_np[:split], prior_cfg, standardization_tol=1.0
            )
            exact_nll_sum += -post.predictive_logpdf(X_np[split:], y_np[split:]).sum()
            n_points += y_query.numel()
    model.train()
    model_nll = model_nll_sum / n_points
    marginal_nll = baseline_nll_sum / n_points
    exact_nll = exact_nll_sum / n_points
    return {
        "model_nll": model_nll,
        "marginal_nll": marginal_nll,
        "exact_nll": exact_nll,
        "gap_closed": (marginal_nll - model_nll) / (marginal_nll - exact_nll),
        "selection_auc": _binary_auc(np.concatenate(sel_probs), np.concatenate(sel_truth)),
    }


