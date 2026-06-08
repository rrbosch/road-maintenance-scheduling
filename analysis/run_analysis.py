"""Driver for the analysis pipeline (replaces results_processing2.py::main).

Usage:
    python analysis/run_analysis.py <experiment_dir> [<experiment_dir> ...]

For each experiment directory it discovers runs (via config.json), computes per-generation Pareto
metrics from fronts.csv, builds a tidy table (one row per run/seed/generation), and writes the
figures + a metrics CSV under ``analysis/output/<experiment_name>/``.
"""
import os
import sys

import numpy as np
import pandas as pd

# allow running as a script from any cwd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis import load, plots
from analysis.metrics import DEFAULT_REFERENCE_POINT, METRIC_COLS, per_generation_metrics

_PROGRESS_COLS = ['time_cum', 'n_computed', 'exact_evals', 'lb_pruned',
                  'scenarios_materialized', 'n_estimated']


def build_tidy(runs, reference_point=DEFAULT_REFERENCE_POINT) -> pd.DataFrame:
    """One row per (run, generation): Pareto metrics joined with the progress diagnostics."""
    frames = []
    for run in runs:
        if run.fronts is None or run.fronts.empty:
            continue
        metrics = per_generation_metrics(run.fronts, reference_point)
        prog = run.progress.rename(columns={'iteration': 'generation'}) if not run.progress.empty else pd.DataFrame()
        keep = ['generation'] + [c for c in _PROGRESS_COLS if c in prog.columns]
        df = metrics.merge(prog[keep], on='generation', how='left') if not prog.empty else metrics
        df['iteration'] = df['generation']
        df['config_key'] = str(run.config_key())
        df['label'] = run.label
        df['seed'] = run.seed
        df['_config'] = [run.config] * len(df)  # carried for styling
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def to_time_grid(tidy, resolution=60, max_time=24 * 3600) -> pd.DataFrame:
    """Resample each (config, seed) series onto a regular wall-clock grid (last value <= grid t)."""
    if 'time_cum' not in tidy.columns:
        return pd.DataFrame()
    value_cols = [c for c in METRIC_COLS + ['n_computed', 'iteration'] if c in tidy.columns]
    grid = np.arange(0, max_time + resolution, resolution)
    out = []
    for (key, seed), grp in tidy.groupby(['config_key', 'seed']):
        grp = grp.sort_values('time_cum')
        idx = np.searchsorted(grp['time_cum'].values, grid, side='right') - 1
        valid = idx >= 0
        sub = grp.iloc[idx[valid]].copy()
        sub['time_grid'] = grid[valid]
        sub['config_key'] = key
        sub['seed'] = seed
        sub['_config'] = [grp.iloc[0]['_config']] * len(sub)
        sub['label'] = grp.iloc[0]['label']
        out.append(sub[['config_key', 'seed', 'time_grid', 'label', '_config'] + value_cols])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def analyze(experiment_dir, reference_point=DEFAULT_REFERENCE_POINT):
    name = os.path.basename(os.path.normpath(experiment_dir))
    out_dir = os.path.join(os.path.dirname(__file__), 'output', name)
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    runs = load.discover_runs(experiment_dir)
    if not runs:
        print(f"no runs found under {experiment_dir}")
        return

    tidy = build_tidy(runs, reference_point)
    if tidy.empty:
        print("no per-generation metrics could be built (empty fronts)")
        return
    tidy.drop(columns=['_config']).to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
    tidy_time = to_time_grid(tidy)

    plots.plot_final_pareto_fronts(runs, plots_dir, reference_point)
    plots.plot_metric_vs_iteration(tidy, plots_dir)
    plots.plot_pruning_diagnostics(tidy, plots_dir)
    plots.plot_final_metric_boxplots(tidy, plots_dir)
    plots.plot_surrogate_learning_curve(runs, plots_dir)
    if not tidy_time.empty:
        plots.plot_metric_vs_time(tidy_time, plots_dir)
        plots.plot_iteration_vs_time(tidy_time, plots_dir)

    print(f"analysis written to {out_dir}")


def main(argv):
    if len(argv) < 2:
        from Src.Utils.Utils import EXPERIMENTS_DIR
        print(f"usage: python analysis/run_analysis.py <experiment_dir> ...\n"
              f"(experiments live under {EXPERIMENTS_DIR})")
        return
    for exp in argv[1:]:
        analyze(exp)


if __name__ == "__main__":
    main(sys.argv)
