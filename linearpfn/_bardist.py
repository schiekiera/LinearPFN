"""Bar (Riemann) distribution head (private helper of linearpfn.model).

Fixed support [-8, 8] in 512 equal bins plus two half-normal tail bins, per
the TabPFN recipe; re-exported through `linearpfn.model`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["BarDistribution"]

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


class BarDistribution(nn.Module):
    """Piecewise-constant predictive distribution over a fixed support.

    Bin layout (n_bins = n_inner_bins + 2 logits, in order):
    [0] left tail (-inf, lo]: density p0 * HalfNormal(tail_scale) in (lo - y);
    [1 .. n_inner_bins] equal-width inner bins on (lo, hi];
    [-1] right tail (hi, inf): density p_last * HalfNormal(tail_scale).
    """

    borders: Tensor

    def __init__(
        self,
        support: tuple[float, float] = (-8.0, 8.0),
        n_inner_bins: int = 512,
        tail_scale: float = 1.0,
    ) -> None:
        super().__init__()
        lo, hi = support
        self.n_inner_bins = n_inner_bins
        self.n_bins = n_inner_bins + 2
        self.width = (hi - lo) / n_inner_bins
        self.tail_scale = tail_scale
        self.register_buffer("borders", torch.linspace(lo, hi, n_inner_bins + 1))
        centers = torch.empty(self.n_bins, dtype=torch.float64)
        centers[1:-1] = (self.borders[:-1] + self.borders[1:]).double() / 2
        centers[0] = lo - tail_scale * _SQRT_2_OVER_PI  # E[lo - T], T ~ HalfNormal
        centers[-1] = hi + tail_scale * _SQRT_2_OVER_PI
        second = centers**2
        second[1:-1] += self.width**2 / 12.0  # uniform within inner bins
        second[0] = lo**2 - 2 * lo * tail_scale * _SQRT_2_OVER_PI + tail_scale**2
        second[-1] = hi**2 + 2 * hi * tail_scale * _SQRT_2_OVER_PI + tail_scale**2
        self.register_buffer("_centers", centers)
        self.register_buffer("_second_moments", second)

    def _log_halfnormal(self, t: Tensor) -> Tensor:
        s = self.tail_scale
        const = math.log(2.0) - math.log(s) - 0.5 * math.log(2.0 * math.pi)
        return const - t.clamp_min(0.0) ** 2 / (2.0 * s * s)

    def nll(self, logits: Tensor, y: Tensor) -> Tensor:
        """Negative log density at y; y has the shape of logits without the
        bin axis. This is the training loss (continuous NLL in nats)."""
        logp = torch.log_softmax(logits, dim=-1)
        borders = self.borders.to(y.dtype)
        idx = torch.searchsorted(borders, y.contiguous())
        inner = logp.gather(-1, idx.clamp(1, self.n_inner_bins).unsqueeze(-1)).squeeze(-1)
        inner = inner - math.log(self.width)
        left = logp[..., 0] + self._log_halfnormal(borders[0] - y)
        right = logp[..., -1] + self._log_halfnormal(y - borders[-1])
        log_density = torch.where(
            idx == 0, left, torch.where(idx > self.n_inner_bins, right, inner)
        )
        return -log_density

    def mean(self, logits: Tensor) -> Tensor:
        probs = torch.softmax(logits, dim=-1)
        return probs @ self._centers.to(probs.dtype)

    def variance(self, logits: Tensor) -> Tensor:
        probs = torch.softmax(logits, dim=-1)
        second = probs @ self._second_moments.to(probs.dtype)
        return second - self.mean(logits) ** 2

    def cdf(self, logits: Tensor, y: Tensor) -> Tensor:
        """P(Y <= y); y shaped (..., L) against logits (..., n_bins)."""
        probs = torch.softmax(logits, dim=-1)
        cum = probs.cumsum(-1)
        borders = self.borders.to(y.dtype)
        idx = torch.searchsorted(borders, y.contiguous())  # 0..n_inner_bins+1
        idx_inner = idx.clamp(1, self.n_inner_bins)
        below = cum.gather(-1, idx_inner - 1)  # mass strictly before the bin
        frac = (y - borders[idx_inner - 1]) / self.width
        inner = below + probs.gather(-1, idx_inner) * frac.clamp(0.0, 1.0)
        s = self.tail_scale
        phi_left = torch.special.ndtr((borders[0] - y) / s)
        left = probs[..., :1] * 2.0 * (1.0 - phi_left)
        phi_right = torch.special.ndtr((y - borders[-1]) / s)
        right = 1.0 - probs[..., -1:] * 2.0 * (1.0 - phi_right)
        return torch.where(idx == 0, left, torch.where(idx > self.n_inner_bins, right, inner))

    def quantile(self, logits: Tensor, levels: Tensor) -> Tensor:
        """Quantiles at `levels` (L,); returns (..., L)."""
        probs = torch.softmax(logits, dim=-1).clamp_min(1e-12)
        cum = probs.cumsum(-1)
        t = levels.to(probs.dtype).expand(*probs.shape[:-1], levels.shape[-1]).contiguous()
        k = torch.searchsorted(cum.contiguous(), t).clamp(0, self.n_bins - 1)
        borders = self.borders.to(probs.dtype)
        s = self.tail_scale
        eps = 1e-9
        # inner bins: uniform interpolation from the bin's left border
        k_inner = k.clamp(1, self.n_inner_bins)
        below = cum.gather(-1, k_inner - 1)
        inner = borders[k_inner - 1] + (t - below) / probs.gather(-1, k_inner) * self.width
        # left tail: t = p0 * 2 * (1 - ndtr((lo - y)/s))
        arg_l = (1.0 - t / (2.0 * probs[..., :1])).clamp(eps, 1.0 - eps)
        left = borders[0] - s * torch.special.ndtri(arg_l)
        # right tail: t = 1 - p_last + p_last * (2 * ndtr((y - hi)/s) - 1)
        a = ((t - (1.0 - probs[..., -1:])) / probs[..., -1:]).clamp(0.0, 1.0)
        arg_r = ((1.0 + a) / 2.0).clamp(eps, 1.0 - eps)
        right = borders[-1] + s * torch.special.ndtri(arg_r)
        return torch.where(k == 0, left, torch.where(k > self.n_inner_bins, right, inner))


