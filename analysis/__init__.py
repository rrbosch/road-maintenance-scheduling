"""Post-hoc analysis of optimization runs.

Consumes the per-run schema written by ``Src/Utils/results_io.py`` (``config.json``,
``progress.csv``, ``fronts.csv``, ``final_solutions.csv``, ``surrogate.csv``) and reproduces the
result figures (Pareto fronts, convergence-metric/iteration-vs-time confidence-interval plots,
sensitivity boxplots) plus the surrogate learning curve and pruning diagnostics.

Entry point: ``analysis/run_analysis.py``.
"""
