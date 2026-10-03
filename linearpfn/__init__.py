"""LinearPFN: amortized spike-and-slab regression — Bayesian variable selection without MCMC."""

__version__ = "0.1.0"
__all__ = ["LinearPFN"]


def __getattr__(name: str):
    # lazy, so that the torch-free modules (linearpfn.prior, ...) stay importable without torch
    if name == "LinearPFN":
        from linearpfn.api import LinearPFN

        return LinearPFN
    raise AttributeError(f"module 'linearpfn' has no attribute {name!r}")
