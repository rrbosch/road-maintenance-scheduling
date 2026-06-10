"""Figures for the analysis pipeline.

Reproduces the spec figures from ``results_processing2.py`` (Pareto fronts, metric-vs-iteration /
metric-vs-time confidence-interval plots, iteration-vs-time, sensitivity boxplots) and adds the two
telemetry figures from the logging half (surrogate learning curve, pruning diagnostics). Styling is
intentionally plain — the colour-key / font / Gantt polish lives in overhaul item 13.
"""
import os

import matplotlib
matplotlib.use("Agg")  # headless: write files, never block on a window
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.colors import to_rgba

from analysis.metrics import METRIC_COLS

_EVAL_COLOR = {'ApproximateEvaluator': 'tab:blue', 'LowerBoundEvaluator': 'tab:red',
               'StandardEvaluator': 'tab:green'}
_EVAL_LABEL = {'ApproximateEvaluator': 'LE', 'LowerBoundEvaluator': 'EP', 'StandardEvaluator': 'S'}
_LB_STYLE = {'XGBoost': '-', 'Heuristic': '--'}
_LB_LABEL = {'XGBoost': 'X', 'Heuristic': 'H'}
_FALLBACK_COLORS = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])


def style_for(config, idx=0):
    """(color, linestyle, label) for a run, mirroring results_processing2.get_label_and_style.

    Uses the evaluator→colour / lower_bound→linestyle / quantile→white-blend scheme when those
    fields are present, otherwise falls back to a plain colour cycle keyed by ``idx``.
    """
    evaluator = config.get('evaluator')
    lb = str(config.get('lower_bound'))
    q = config.get('lower_bound_quantile')
    if evaluator in _EVAL_COLOR:
        base = to_rgba(_EVAL_COLOR[evaluator])
        white = float(q) if (q is not None and str(q) not in ('nan', 'None')) else 0.0
        color = tuple(base[i] + (1 - base[i]) * 0.6 * white for i in range(3)) + (base[3],)
        linestyle = _LB_STYLE.get(lb, '-')
        label = f"{_EVAL_LABEL[evaluator]}|{_LB_LABEL.get(lb, '-')}|{q if q is not None else '-'}"
        return color, linestyle, label
    return _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)], '-', None


def _mean_ci(values, ci=0.90):
    """Mean and (lo, hi) confidence band across seeds for one x-position."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    m = values.mean()
    if len(values) < 2:
        return m, m, m
    half = stats.t.ppf(0.5 + ci / 2, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return m, m - half, m + half


def _plot_grouped_ci(tidy, x, y, out_path, xlabel, ylabel, title, logy=False):
    """Mean ± CI of ``y`` vs ``x``, one line per config group, averaged across seeds."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (key, grp) in enumerate(tidy.groupby('config_key')):
        cfg = grp.iloc[0]['_config']
        color, ls, label = style_for(cfg, idx)
        label = label or grp.iloc[0]['label']
        agg = grp.groupby(x)[y].apply(list)
        xs, means, los, his = [], [], [], []
        for xv, vals in agg.items():
            m, lo, hi = _mean_ci(vals)
            xs.append(xv); means.append(m); los.append(lo); his.append(hi)
        order = np.argsort(xs)
        xs = np.array(xs)[order]; means = np.array(means)[order]
        los = np.array(los)[order]; his = np.array(his)[order]
        ax.plot(xs, means, color=color, linestyle=ls, label=label)
        ax.fill_between(xs, los, his, color=color, alpha=0.2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if logy:
        ax.set_yscale('log')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_metric_vs_iteration(tidy, out_dir):
    for metric in METRIC_COLS + ['n_computed']:
        if metric not in tidy.columns:
            continue
        _plot_grouped_ci(tidy, 'iteration', metric,
                         os.path.join(out_dir, f"{metric.replace(' ', '_')}_vs_iteration.pdf"),
                         "iteration", metric, f"{metric} vs iteration (mean, 90% CI)")


def plot_metric_vs_time(tidy_time, out_dir):
    for metric in METRIC_COLS + ['n_computed']:
        if metric not in tidy_time.columns:
            continue
        _plot_grouped_ci(tidy_time, 'time_grid', metric,
                         os.path.join(out_dir, f"{metric.replace(' ', '_')}_vs_time.pdf"),
                         "wall-clock time (s)", metric, f"{metric} vs time (mean, 90% CI)")


def plot_iteration_vs_time(tidy_time, out_dir):
    _plot_grouped_ci(tidy_time, 'time_grid', 'iteration',
                     os.path.join(out_dir, "iteration_vs_time.pdf"),
                     "wall-clock time (s)", "iteration", "Iteration progress over time (mean, 90% CI)")


def plot_pruning_diagnostics(tidy, out_dir):
    cols = [c for c in ['exact_evals', 'lb_pruned', 'scenarios_materialized', 'n_estimated']
            if c in tidy.columns]
    for metric in cols:
        _plot_grouped_ci(tidy, 'iteration', metric,
                         os.path.join(out_dir, f"pruning_{metric}_vs_iteration.pdf"),
                         "iteration", metric, f"{metric} per generation (mean, 90% CI)")


def plot_surrogate_learning_curve(runs, out_dir):
    """MAPE and pinball loss vs cumulative simulations, one line per run that logged a surrogate."""
    for metric in ['mape', 'pinball_loss']:
        fig, ax = plt.subplots(figsize=(8, 5))
        any_data = False
        for idx, run in enumerate(runs):
            s = run.surrogate
            if s is None or s.empty or metric not in s.columns:
                continue
            any_data = True
            color, ls, label = style_for(run.config, idx)
            base = label or run.label
            # One line per surrogate model. Component (PLBE per-scenario lower bound) and schedule
            # (item-11 whole-schedule baseline) are logged in the same surrogate.csv, distinguished
            # by the 'model' column; older runs without it are treated as a single series.
            if 'model' in s.columns and s['model'].nunique() > 1:
                groups = [(f"{base} [{m}]", g) for m, g in s.groupby('model')]
            else:
                groups = [(base, s)]
            for series_label, g in groups:
                ax.plot(g['n_computed'], g[metric], color=color, linestyle=ls,
                        label=series_label, marker='o', markersize=3)
        if not any_data:
            plt.close(fig)
            continue
        ax.set_xlabel("cumulative simulations (n_computed)")
        ax.set_ylabel(metric)
        ax.set_title(f"Surrogate {metric} over the search")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"surrogate_{metric}.pdf"), bbox_inches='tight')
        plt.close(fig)


def plot_final_pareto_fronts(runs, out_dir, reference_point=None):
    """Scatter the final Pareto front of each run in (objective0, objective1) space."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, run in enumerate(runs):
        if run.fronts is None or run.fronts.empty:
            continue
        obj_cols = [c for c in run.fronts.columns if c != 'generation']
        last_gen = run.fronts['generation'].max()
        front = run.fronts[run.fronts['generation'] == last_gen][obj_cols].values
        color, ls, label = style_for(run.config, idx)
        ax.scatter(front[:, 0], front[:, 1], s=12, color=color, label=(label or run.label))
    obj_cols = [c for c in runs[0].fronts.columns if c != 'generation'] if runs else ['obj0', 'obj1']
    ax.set_xlabel(obj_cols[0]); ax.set_ylabel(obj_cols[1])
    ax.set_title("Final Pareto fronts")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "final_pareto_fronts.pdf"), bbox_inches='tight')
    plt.close(fig)


def plot_final_metric_boxplots(tidy, out_dir):
    """Boxplot of each final-generation metric across seeds, per config (sensitivity-style)."""
    final = tidy.sort_values('iteration').groupby(['config_key', 'seed']).tail(1)
    for metric in METRIC_COLS:
        if metric not in final.columns:
            continue
        groups, labels = [], []
        for key, grp in final.groupby('config_key'):
            groups.append(grp[metric].dropna().values)
            labels.append(grp.iloc[0]['label'])
        if not groups:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(groups, labels=labels)
        ax.set_ylabel(metric); ax.set_title(f"Final {metric} by configuration")
        ax.tick_params(axis='x', rotation=30)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"boxplot_{metric.replace(' ', '_')}.pdf"),
                    bbox_inches='tight')
        plt.close(fig)
