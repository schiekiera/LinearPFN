"""Benchmark and semi-synthetic infrastructure for LinearPFN.

Everything under bench/ is comparison tooling, NOT part of the linearpfn
package: it may use sklearn/pandas/matplotlib and shells out to Rscript, and
none of that becomes a package dependency. The package's own validation
against the exact reference lives in linearpfn.validate; this directory is
the methods-comparison layer around it.
"""
