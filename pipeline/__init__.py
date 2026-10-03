"""The manuscript pipeline: finished outputs -> numbers -> tables, figures,
LaTeX macros. No stage in this package fits a model or draws from the prior;
everything reads finished artifacts named in configs/pipeline.yaml.

The numbered stages in scripts/ use this package; they write their
numbers to reports/paper/ and the tables, figures and macros to paper/.
"""
