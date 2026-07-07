"""Robustness analysis for a multi-variant campaign (Experiment 3).

Experiment 3 runs the same three methods --- PLBE (``EP|X|q``), standard NSGA-II (``S|-|-``),
and the schedule-level SAEA baseline (``SS|X|0.5``) --- on several *variants* of the SF-76
problem (road-capacity scaling, construction-resource tightness, network topology). The question
is whether PLBE's advantage is robust across all of them.

Design notes
------------
* Metrics are computed on each run's **final** Pareto front (``fronts.csv`` last generation).
* Every variant is scored **against its own reference point** and its own pooled best-known
  front, because the variants live on very different objective scales (e.g. ``road capacity 0.9``
  is far more congested than ``road capacity 100``). A single fixed reference point would be
  meaningless across variants --- the same lesson learned on the Anaheim instance (Experiment 4).
* Because hypervolume therefore is *not* comparable across variants, the robustness story rests on
  **reference-independent** measures: two-set coverage, IGD+ (normalized per variant), Pareto-front
  size, and the unique-simulation reduction. Per-variant HV is still reported for completeness.

Outputs (under ``analysis/output/<experiment name>/``)
    metrics.csv            one row per (variant, config, seed): HV, IGD+, PF size, min dist,
                           unique sims, iterations.
    robustness_summary.csv one row per variant: PLBE-vs-baseline headline numbers + significance.
    significance.txt       per-variant Friedman + Mann-Whitney + two-set coverage.
    plots/variant_boxplots.pdf   grouped boxplots (HV, IGD+, PF size, unique sims) across variants.

Usage (run with the project's Python 3.11 env, which has numba/scipy):
    python analysis/analyze_variants.py "Experiments/E3 SF-76 variants"
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis import load
from analysis.metrics import pareto_metrics, return_non_dominated

# pretty variant names + display order (mirrors Table "Overview of the seven SF-76 problem variants")
VARIANT_NAMES = {
    'Sioux Falls road capacity 0.9': r'Road cap. 0.9$\times$',
    'Sioux Falls road capacity 1.1': r'Road cap. 1.1$\times$',
    'Sioux Falls road capacity 100': r'Road cap. 100$\times$',
    'Sioux Falls construction capacity 0.7': r'Constr. cap. 0.7$\times$',
    'Sioux Falls construction capacity 100': r'Constr. cap. 100$\times$',
    'Sioux Falls Less Connected': 'Less connected',
    'Sioux Falls More Connected': 'More connected',
}
VARIANT_ORDER = list(VARIANT_NAMES)
REF_INFLATION = 1.05  # reference point = 1.05 x pooled nadir (per variant)


def shorthand(config: dict) -> str:
    """Map a run config to the ``<evaluator>|<surrogate>|<quantile>`` shorthand."""
    ev = config.get('evaluator')
    if ev == 'StandardEvaluator':
        return 'S|-|-'
    if ev == 'ScheduleSurrogateEvaluator':
        return f"SS|X|{config.get('schedule_surrogate_quantile', 0.5)}"
    lb = 'X' if config.get('lower_bound') == 'XGBoost' else 'H'
    if ev == 'LowerBoundEvaluator':
        return f"EP|{lb}|{config.get('lower_bound_quantile')}"
    if ev == 'ApproximateEvaluator':
        return f"LE|{lb}|{config.get('lower_bound_quantile')}"
    return str(ev)


def final_front(run) -> np.ndarray:
    """Non-dominated points of a run's last-generation front, as an (n, 2) [SL, TTD] array."""
    f = run.fronts
    g = int(f['generation'].max())
    pts = f.loc[f['generation'] == g, ['SL', 'TTD']].to_numpy(dtype=float)
    return return_non_dominated(pts)


def igd_plus(approx: np.ndarray, reference: np.ndarray, ref_point: np.ndarray) -> float:
    """Normalized IGD+ from a reference set to an approximation (minimization).

    Objectives are scaled by ``ref_point`` first so the two axes are commensurate.
    For each reference point z, the modified distance counts only the objectives on which the
    nearest approximation point is *worse* than z.
    """
    A = approx / ref_point               # (n, 2)
    Z = reference / ref_point            # (m, 2)
    diff = np.maximum(A[None, :, :] - Z[:, None, :], 0.0)   # (m, n, 2)
    d = np.sqrt((diff ** 2).sum(axis=2))                    # (m, n)
    return float(d.min(axis=1).mean())


def coverage(A: np.ndarray, B: np.ndarray) -> float:
    """Two-set coverage C(A, B): fraction of B weakly dominated by some point of A (minimization)."""
    if len(B) == 0 or len(A) == 0:
        return 0.0
    le = np.all(A[None, :, :] <= B[:, None, :], axis=2)   # (|B|, |A|)
    lt = np.any(A[None, :, :] < B[:, None, :], axis=2)
    return float((le & lt).any(axis=1).mean())


def _stars(p):
    return '*' if (p is not None and p < 0.05) else 'n.s.'


def analyze(experiment_dir: str):
    name = os.path.basename(os.path.normpath(experiment_dir))
    out_dir = os.path.join(os.path.dirname(__file__), 'output', name)
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    runs = load.discover_runs(experiment_dir)
    # group runs by (variant case study, config shorthand)
    by_variant = {}
    for r in runs:
        if r.fronts is None or r.fronts.empty:
            continue
        cs = r.config.get('case_study')
        by_variant.setdefault(cs, {}).setdefault(shorthand(r.config), []).append(r)

    variants = [v for v in VARIANT_ORDER if v in by_variant] + \
               [v for v in by_variant if v not in VARIANT_ORDER]

    rows = []            # per (variant, config, seed)
    per_variant_fronts = {}  # (variant, config, seed) -> front, for coverage
    ep_label = None
    for cs in variants:
        cfgs = by_variant[cs]
        ep_label = next((c for c in cfgs if c.startswith('EP')), ep_label)
        # pooled best-known front + per-variant reference point
        pooled = np.vstack([final_front(r) for cfg in cfgs.values() for r in cfg])
        best_known = return_non_dominated(pooled)
        ref_point = pooled.max(axis=0) * REF_INFLATION
        for cfg, cfg_runs in cfgs.items():
            for r in cfg_runs:
                F = final_front(r)
                hv, _spread, mind, size = pareto_metrics(F, ref_point)
                prog = r.progress
                ncomp = float(prog['n_computed'].iloc[-1]) if 'n_computed' in prog else np.nan
                it = int(prog['iteration'].iloc[-1]) if 'iteration' in prog else len(prog)
                rows.append(dict(variant=cs, config=cfg, seed=r.seed, HV=hv,
                                 IGDplus=igd_plus(F, best_known, ref_point), PF_size=size,
                                 min_dist=mind, n_computed=ncomp, iterations=it))
                per_variant_fronts[(cs, cfg, r.seed)] = F

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)

    _write_significance(df, per_variant_fronts, variants, by_variant, out_dir, ep_label)
    _write_summary(df, per_variant_fronts, variants, out_dir, ep_label)
    _plot_boxplots(df, variants, plots_dir, ep_label)
    print(f"analysis written to {out_dir}  (PLBE arm = {ep_label})")
    return df


def _matched(df, cs, a, b, col):
    """Seed-matched value pairs of metric ``col`` for configs a, b on variant cs."""
    da = df[(df.variant == cs) & (df.config == a)].set_index('seed')[col]
    db = df[(df.variant == cs) & (df.config == b)].set_index('seed')[col]
    common = da.index.intersection(db.index)
    return da.loc[common].to_numpy(), db.loc[common].to_numpy()


def _write_significance(df, fronts, variants, by_variant, out_dir, ep):
    from scipy.stats import friedmanchisquare, mannwhitneyu
    lines = [f"Experiment 3 robustness significance  (PLBE arm = {ep})", "=" * 64, ""]
    for cs in variants:
        cfgs = [c for c in [ep, 'S|-|-', f'SS|X|0.5'] if c in df[df.variant == cs].config.values]
        lines.append(f"### {cs}")
        for col in ['HV', 'IGDplus', 'n_computed']:
            arrs, seeds = [], None
            for c in cfgs:
                s = df[(df.variant == cs) & (df.config == c)].set_index('seed')[col]
                arrs.append(s)
            common = arrs[0].index
            for s in arrs[1:]:
                common = common.intersection(s.index)
            mats = [s.loc[common].to_numpy() for s in arrs]
            if len(mats) >= 3 and len(common) >= 3:
                chi, p = friedmanchisquare(*mats)
                lines.append(f"  Friedman {col:9s}: chi2={chi:6.1f} p={p:.2e}  ({len(common)} blocks)")
        for base in ['S|-|-', 'SS|X|0.5']:
            if base not in df[df.variant == cs].config.values or ep is None:
                continue
            for col in ['HV', 'IGDplus', 'n_computed']:
                a, b = _matched(df, cs, ep, base, col)
                if len(a) >= 3:
                    try:
                        p = mannwhitneyu(a, b, alternative='two-sided').pvalue
                    except ValueError:
                        p = None
                    lines.append(f"  {ep} vs {base:8s} {col:9s}: med {np.median(a):.4g} vs "
                                 f"{np.median(b):.4g}  p={p:.2e} {_stars(p)}" if p is not None
                                 else f"  {ep} vs {base:8s} {col:9s}: identical")
        # two-set coverage (median over seeds)
        for base in ['S|-|-', 'SS|X|0.5']:
            if ep is None or base not in df[df.variant == cs].config.values:
                continue
            cov_ab, cov_ba = [], []
            seeds = df[(df.variant == cs) & (df.config == ep)].seed
            for sd in seeds:
                A = fronts.get((cs, ep, sd)); B = fronts.get((cs, base, sd))
                if A is not None and B is not None:
                    cov_ab.append(coverage(A, B)); cov_ba.append(coverage(B, A))
            if cov_ab:
                lines.append(f"  coverage C({ep},{base})={np.median(cov_ab):.3f}  "
                             f"C({base},{ep})={np.median(cov_ba):.3f}")
        lines.append("")
    with open(os.path.join(out_dir, 'significance.txt'), 'w') as f:
        f.write("\n".join(lines))


def _write_summary(df, fronts, variants, out_dir, ep):
    rows = []
    for cs in variants:
        d = df[df.variant == cs]
        def mean(c, col):
            v = d[d.config == c][col]
            return float(v.mean()) if len(v) else np.nan
        hv_ep, hv_s, hv_ss = mean(ep, 'HV'), mean('S|-|-', 'HV'), mean('SS|X|0.5', 'HV')
        sims_ep, sims_s = mean(ep, 'n_computed'), mean('S|-|-', 'n_computed')
        pf_ep, pf_s = mean(ep, 'PF_size'), mean('S|-|-', 'PF_size')
        cov = []
        for sd in d[d.config == ep].seed:
            A = fronts.get((cs, ep, sd)); B = fronts.get((cs, 'S|-|-', sd))
            if A is not None and B is not None:
                cov.append(coverage(A, B))
        rows.append(dict(
            variant=cs, HV_EP=hv_ep, HV_S=hv_s, HV_SS=hv_ss,
            HV_gain_pct=100 * (hv_ep - hv_s) / hv_s if hv_s else np.nan,
            IGDplus_EP=mean(ep, 'IGDplus'), IGDplus_S=mean('S|-|-', 'IGDplus'),
            PF_ratio_EPvsS=pf_ep / pf_s if pf_s else np.nan,
            sim_reduction_SvsEP=sims_s / sims_ep if sims_ep else np.nan,
            coverage_EP_over_S=float(np.median(cov)) if cov else np.nan,
        ))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'robustness_summary.csv'), index=False)


def _plot_boxplots(df, variants, plots_dir, ep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = [c for c in [ep, 'S|-|-', 'SS|X|0.5'] if c in df.config.unique()]
    colors = {ep: '#1f77b4', 'S|-|-': '#555555', 'SS|X|0.5': '#ff7f0e'}
    metrics = [('HV', 'Hypervolume'), ('IGDplus', 'IGD$^{+}$'),
               ('PF_size', 'Pareto-front size'), ('n_computed', 'Unique simulations')]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    labels = [VARIANT_NAMES.get(v, v) for v in variants]
    for ax, (col, title) in zip(axes.ravel(), metrics):
        nc = len(order); width = 0.8 / nc
        for j, c in enumerate(order):
            data = [df[(df.variant == v) & (df.config == c)][col].to_numpy() for v in variants]
            pos = np.arange(len(variants)) + (j - (nc - 1) / 2) * width
            bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True,
                            showfliers=False, medianprops=dict(color='black'))
            for box in bp['boxes']:
                box.set(facecolor=colors.get(c, 'gray'), alpha=0.75)
        ax.set_xticks(np.arange(len(variants)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_title(title)
        ax.grid(True, axis='y', ls=':', alpha=0.5)
        if col == 'n_computed':
            ax.set_yscale('log')
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors.get(c, 'gray'), alpha=0.75) for c in order]
    fig.legend(handles, order, loc='upper center', ncol=len(order), framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(plots_dir, 'variant_boxplots.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for exp in sys.argv[1:]:
        analyze(exp)
