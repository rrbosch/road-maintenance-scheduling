"""Post-hoc false-pruning rate estimation (E2 / item 12, log-and-replay variant).

PLBE prunes a candidate schedule when its ML-estimated *lower bound* objective is dominated by the
incumbent Pareto front, to skip an expensive traffic simulation. A *false prune* is a candidate
whose **true** objective vector ``F(x)`` is actually non-dominated by the incumbent front at the
moment it was pruned — i.e. the prune was a mistake.

The run-side logger (``LowerBoundEvaluator._maybe_log_pruned_sample``, gated by
``Config.false_pruning_log_prob``) records a reproducible random *sample* of pruned candidates —
each as its decision vector ``x`` plus a snapshot of the incumbent front at prune time — to
``pruned_sample.csv`` + ``pruned_sample_fronts.csv``, **without** any exact evaluation (so the run's
reported metrics are unaffected). This script replays that sample *offline*:

  1. discover runs that logged a ``pruned_sample.csv`` (reusing ``analysis/load.py`` conventions),
  2. for each run, instantiate the *same* ``Problem_py`` for its case study so ``F(x)`` is computed
     identically to the run (same ``Problem.evaluate`` path),
  3. exact-evaluate each sampled pruned ``x`` and test whether its true ``F(x)`` is non-dominated by
     the recorded incumbent front via pymoo's ``find_non_dominated`` (mirroring
     ``LowerBoundEvaluator._maybe_count_false_pruning``),
  4. report the false-pruning rate per (config, seed) and aggregated, with a binomial confidence
     interval (Wilson). When zero false prunes are observed, report the one-sided 95% upper bound.

Output: ``false_pruning.csv`` (per-run + aggregated rows) + ``false_pruning_summary.txt`` under
``analysis/output/<experiment>/``.

For large campaigns the replay is the expensive part (one exact traffic evaluation per sampled
prune), so two knobs keep it tractable:
  * ``--max-samples N`` deterministically sub-samples N pruned candidates per run (the false-prune
    rate is ~0, so a few thousand samples already give a tight CI — pooled 95% upper bound ~= 3/n),
  * ``--workers W`` replays the (independent) runs across W processes; each worker builds its own
    ``Problem`` so there is no shared state. Per-run results are checkpointed under
    ``analysis/output/<exp>/partial/`` so an interrupted/re-submitted job resumes.

Usage:
    python analysis/false_pruning.py "Experiments/<experiment_dir>" [<another> ...] \
        [--workers W] [--max-samples N] [--no-resume]
"""
import argparse
import functools
import hashlib
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except Exception:  # noqa: BLE001 - tqdm is optional; fall back to periodic prints
    tqdm = None
    _HAVE_TQDM = False

# allow running as a script from any cwd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymoo.util.nds.non_dominated_sorting import find_non_dominated

from analysis import load


# ----------------------------------------------------------------------------- confidence intervals
def wilson_interval(k, n, z=1.959963984540054):
    """Two-sided Wilson score interval for a binomial proportion (95% by default, z for 1-alpha/2).

    Robust near p=0/1 and for small n (unlike the normal-approx Wald interval). Returns
    ``(lo, hi)`` clipped to [0, 1]; ``(nan, nan)`` if ``n == 0``.
    """
    if n == 0:
        return float('nan'), float('nan')
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def one_sided_upper_95(k, n):
    """One-sided 95% upper confidence bound on a binomial rate.

    Uses the Clopper-Pearson exact upper limit (Beta quantile); for ``k == 0`` this reduces to the
    familiar rule-of-three-ish ``1 - 0.05**(1/n)`` (~3/n for large n). Returns nan if ``n == 0``.
    """
    if n == 0:
        return float('nan')
    if k >= n:
        return 1.0
    try:
        from scipy.stats import beta
        return float(beta.ppf(0.95, k + 1, n - k))
    except Exception:
        # closed form for the k=0 case; conservative fallback otherwise
        if k == 0:
            return 1.0 - 0.05 ** (1.0 / n)
        return min(1.0, (k + 3.0) / n)


# --------------------------------------------------------------------------------- run discovery/IO
def discover_pruned_runs(root):
    """Walk ``root`` for run directories that logged a ``pruned_sample.csv`` (the E2 sampler).

    Mirrors ``analysis/load.discover_runs`` (run identity comes from ``config.json``, not the path)
    but keys on the presence of ``pruned_sample.csv`` rather than ``fronts.csv``.
    """
    runs = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if 'config.json' not in filenames or 'pruned_sample.csv' not in filenames:
            continue
        try:
            runs.append(load.load_run(dirpath))
        except Exception as e:  # noqa: BLE001 - keep going past one bad run
            print(f"skipping {dirpath}: failed to load ({e})")
    print(f"discovered {len(runs)} run(s) with pruned_sample.csv under {root}")
    return runs


def _load_pruned_samples(directory):
    """Load (x_df, fronts_df) for one run; returns (None, None) if the sample file is empty."""
    x_path = os.path.join(directory, 'pruned_sample.csv')
    f_path = os.path.join(directory, 'pruned_sample_fronts.csv')
    if not os.path.exists(x_path):
        return None, None
    x_df = pd.read_csv(x_path)
    if x_df.empty:
        return None, None
    fronts_df = pd.read_csv(f_path) if os.path.exists(f_path) else pd.DataFrame()
    return x_df, fronts_df


# ----------------------------------------------------------------------------------- problem reuse
def _build_problem(config, _cache={}):
    """Instantiate (and cache) a ``Problem_py`` matching a run's config, for identical ``F(x)``.

    Building a problem runs the base traffic assignment (expensive), so we memoize per case study —
    the exact ``evaluate`` path is independent of the surrogate/quantile knobs, so all runs on the
    same case study can share one problem instance.
    """
    case_study = config.get('case_study', 'Sioux Falls Expanded')
    if case_study in _cache:
        return _cache[case_study]
    from Environments.env.Problem import Problem_py
    sims = set(config.get('sims', ['traffic']))
    objectives = set(config.get('objectives', ['SL', 'TTD']))
    problem = Problem_py(
        case_study, sims, objectives,
        lower_bound=config.get('lower_bound', 'XGBoost'),
        lower_bound_quantile=config.get('lower_bound_quantile', 0.2),
        traffic_cache_size=config.get('traffic_cache_size', 200_000),
        schedule_surrogate_quantile=config.get('schedule_surrogate_quantile', 0.5),
        seed=config.get('algo_seed', 0),
    )
    _cache[case_study] = problem
    return problem


def _true_F(problem, x):
    """Exact objective vector for a decision vector ``x`` (the same path the run uses)."""
    x = np.asarray(x, dtype=int)
    out = problem.evaluate(x, return_as_dictionary=True)
    return np.asarray(out['F'], dtype=float)


# --------------------------------------------------------------------------------------- core eval
def evaluate_run(run, max_samples=None):
    """Replay one run's sampled prunes; return a dict of counts.

    A *false prune* (mirrors ``_maybe_count_false_pruning``): the sampled pruned ``x`` has a finite
    true ``F(x)`` that is non-dominated by its recorded incumbent front (``find_non_dominated`` finds
    it surviving). Samples whose true ``F`` is non-finite (infeasible) are excluded from the
    denominator — pruning an infeasible candidate is never a mistake.

    ``max_samples`` (optional) caps the number of replayed rows per run: the ~0 false-prune rate
    only needs a few thousand samples for a tight CI, and each replayed row costs one exact traffic
    evaluation. Sub-sampling is deterministic (seeded on ``run.seed``) so it is reproducible. True
    ``F(x)`` depends only on ``x`` (given the shared base ``Problem``), so it is memoized per unique
    ``x`` to avoid re-simulating identical pruned schedules — the per-row false-prune test still
    uses each row's own recorded front, so correctness is unchanged.
    """
    x_df, fronts_df = _load_pruned_samples(run.directory)
    if x_df is None:
        return None
    n_total = len(x_df)

    # deterministic per-run sub-sample (seeded on the run's seed => reproducible across invocations)
    if max_samples is not None and n_total > max_samples:
        rng = np.random.default_rng(int(run.seed or 0))
        idx = np.sort(rng.choice(n_total, size=max_samples, replace=False))
        x_df = x_df.iloc[idx]

    problem = _build_problem(run.config)

    x_cols = [c for c in x_df.columns if c.startswith('x') and c[1:].isdigit()]
    x_cols = sorted(x_cols, key=lambda c: int(c[1:]))

    # index the front snapshots by sample_id
    if not fronts_df.empty:
        f_cols = sorted([c for c in fronts_df.columns if c.startswith('f') and c[1:].isdigit()],
                        key=lambda c: int(c[1:]))
        fronts_by_id = {sid: g[f_cols].to_numpy(dtype=float)
                        for sid, g in fronts_df.groupby('sample_id')}
    else:
        fronts_by_id = {}

    n_sampled = 0       # samples with a finite (feasible) true F => the CI denominator
    n_false = 0         # false prunes among them
    n_infeasible = 0    # excluded (true F non-finite)
    f_cache = {}        # x.tobytes() -> true F (avoid re-simulating identical schedules)
    # progress feedback over the (often many-thousand-row) replay loop — purely cosmetic
    n_rows = len(x_df)
    desc = f"  {run.label} seed {run.seed}"
    if _HAVE_TQDM:
        row_iter = tqdm(x_df.iterrows(), total=n_rows, desc=desc, unit="smp",
                        leave=False, dynamic_ncols=True)
    else:
        row_iter = x_df.iterrows()
    done = 0
    for _, row in row_iter:
        done += 1
        if not _HAVE_TQDM and (done % 2000 == 0 or done == n_rows):
            print(f"  {done}/{n_rows}", flush=True)
        sid = int(row['sample_id'])
        x = row[x_cols].to_numpy(dtype=int)
        key = x.tobytes()
        true_F = f_cache.get(key)
        if true_F is None:
            true_F = np.atleast_2d(_true_F(problem, x))
            f_cache[key] = true_F
        finite = bool(np.all(np.isfinite(true_F)))
        if not finite:
            n_infeasible += 1
            continue
        n_sampled += 1
        front = fronts_by_id.get(sid, np.empty((0, true_F.shape[1])))
        if front.shape[0] == 0:
            # no incumbent front recorded => nothing could dominate it => it would have survived
            is_false = True
        else:
            is_false = len(find_non_dominated(F=true_F, _F=np.atleast_2d(front))) > 0
        n_false += int(is_false)

    return {
        'directory': run.directory,
        'config_key': str(run.config_key()),
        'label': run.label,
        'seed': run.seed,
        'case_study': run.config.get('case_study'),
        'false_pruning_log_prob': run.config.get('false_pruning_log_prob'),
        'n_samples_total': n_total,
        'n_sampled_feasible': n_sampled,
        'n_infeasible_excluded': n_infeasible,
        'n_false_pruned': n_false,
    }


def _summarize(k, n):
    """(rate, wilson_lo, wilson_hi, upper95) for k false prunes out of n feasible sampled prunes."""
    rate = (k / n) if n else float('nan')
    lo, hi = wilson_interval(k, n)
    upper95 = one_sided_upper_95(k, n)
    return rate, lo, hi, upper95


# ----------------------------------------------------------------------------- per-run checkpointing
def _checkpoint_path(partial_dir, run):
    """Stable per-run checkpoint filename (seed + short hash of the directory, so seeds don't clash)."""
    h = hashlib.sha1(os.path.abspath(run.directory).encode()).hexdigest()[:8]
    return os.path.join(partial_dir, f"seed{run.seed}_{h}.json")


def _process_run(run, partial_dir, max_samples):
    """Replay one run and checkpoint its result to ``partial_dir``; returns the result dict (or None).

    Top-level (picklable) so it can be dispatched to a multiprocessing pool. Each worker builds its
    own ``Problem`` via the per-process memo in ``_build_problem`` — no shared state across workers.
    """
    res = evaluate_run(run, max_samples=max_samples)
    if res is not None:
        with open(_checkpoint_path(partial_dir, run), 'w') as f:
            json.dump(res, f)
    return res


def analyze(experiment_dir, workers=1, max_samples=2000, resume=True):
    name = os.path.basename(os.path.normpath(experiment_dir))
    out_dir = os.path.join(os.path.dirname(__file__), 'output', name)
    partial_dir = os.path.join(out_dir, 'partial')
    os.makedirs(partial_dir, exist_ok=True)

    runs = discover_pruned_runs(experiment_dir)
    if not runs:
        print(f"no runs with pruned_sample.csv found under {experiment_dir}")
        return

    # evaluate_run only needs config/directory/seed/label — drop the large fronts/progress/surrogate
    # frames load_run pulled in, so each Run pickles cheaply when dispatched to worker processes.
    for run in runs:
        run.fronts = run.progress = run.surrogate = pd.DataFrame()

    # resume: load already-checkpointed runs, replay only the rest
    per_run = []
    todo = []
    for run in runs:
        cp = _checkpoint_path(partial_dir, run)
        if resume and os.path.exists(cp):
            with open(cp) as f:
                per_run.append(json.load(f))
        else:
            todo.append(run)
    if per_run:
        print(f"resuming: {len(per_run)} run(s) already done, {len(todo)} to replay", flush=True)

    n_todo = len(todo)
    process = functools.partial(_process_run, partial_dir=partial_dir, max_samples=max_samples)
    if workers > 1 and n_todo > 1:
        print(f"replaying {n_todo} run(s) across {min(workers, n_todo)} worker(s)", flush=True)
        with mp.Pool(processes=min(workers, n_todo)) as pool:
            for i, res in enumerate(pool.imap_unordered(process, todo), start=1):
                if res is None:
                    continue
                print(f"[{i}/{n_todo}] done: {res['label']} seed {res['seed']} "
                      f"({res['n_sampled_feasible']} feasible, {res['n_false_pruned']} false)",
                      flush=True)
                per_run.append(res)
    else:
        for i, run in enumerate(todo, start=1):
            print(f"[run {i}/{n_todo}] {run.label} seed {run.seed}", flush=True)
            res = process(run)
            if res is not None:
                per_run.append(res)

    if not per_run:
        print("no non-empty pruned samples to evaluate")
        return

    df = pd.DataFrame(per_run)

    # per-(config, seed) rates
    df[['rate', 'wilson_lo', 'wilson_hi', 'upper95']] = df.apply(
        lambda r: pd.Series(_summarize(r['n_false_pruned'], r['n_sampled_feasible'])), axis=1)
    df['level'] = 'run'

    # aggregated per config_key (pool all seeds), and a grand total
    agg_rows = []
    for key, g in df.groupby('config_key'):
        k, n = int(g['n_false_pruned'].sum()), int(g['n_sampled_feasible'].sum())
        rate, lo, hi, upper95 = _summarize(k, n)
        agg_rows.append({'level': 'config', 'config_key': key, 'label': g['label'].iloc[0],
                         'case_study': g['case_study'].iloc[0],
                         'n_sampled_feasible': n, 'n_false_pruned': k,
                         'n_infeasible_excluded': int(g['n_infeasible_excluded'].sum()),
                         'rate': rate, 'wilson_lo': lo, 'wilson_hi': hi, 'upper95': upper95})
    k_all, n_all = int(df['n_false_pruned'].sum()), int(df['n_sampled_feasible'].sum())
    rate, lo, hi, upper95 = _summarize(k_all, n_all)
    agg_rows.append({'level': 'overall', 'config_key': 'ALL', 'label': name,
                     'case_study': '', 'n_sampled_feasible': n_all, 'n_false_pruned': k_all,
                     'n_infeasible_excluded': int(df['n_infeasible_excluded'].sum()),
                     'rate': rate, 'wilson_lo': lo, 'wilson_hi': hi, 'upper95': upper95})

    out = pd.concat([df.drop(columns=[c for c in df.columns if c.startswith('_')]),
                     pd.DataFrame(agg_rows)], ignore_index=True)
    csv_path = os.path.join(out_dir, 'false_pruning.csv')
    out.to_csv(csv_path, index=False)

    # short text summary
    lines = [f"False-pruning analysis for: {name}",
             f"runs evaluated: {len(per_run)}", ""]
    for r in agg_rows:
        if r['level'] == 'config':
            tag = f"config [{r['label']}]"
        else:
            tag = "OVERALL"
        n, k = r['n_sampled_feasible'], r['n_false_pruned']
        if k == 0:
            lines.append(f"{tag}: {k}/{n} false prunes; rate=0, "
                         f"95% one-sided upper bound = {r['upper95']:.3e} "
                         f"(excluded {r['n_infeasible_excluded']} infeasible)")
        else:
            lines.append(f"{tag}: {k}/{n} false prunes; rate={r['rate']:.3e}, "
                         f"95% Wilson CI = [{r['wilson_lo']:.3e}, {r['wilson_hi']:.3e}] "
                         f"(excluded {r['n_infeasible_excluded']} infeasible)")
    summary = "\n".join(lines)
    with open(os.path.join(out_dir, 'false_pruning_summary.txt'), 'w') as f:
        f.write(summary + "\n")
    print(summary)
    print(f"\nwritten to {csv_path}")


def main(argv):
    parser = argparse.ArgumentParser(
        description="Estimate PLBE's false-pruning rate by replaying logged pruned samples.")
    parser.add_argument("experiment_dirs", nargs="*",
                        help="experiment directories containing runs with pruned_sample.csv")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes (runs are independent; default 1)")
    parser.add_argument("--max-samples", type=int, default=2000,
                        help="cap replayed pruned candidates per run (default 2000; 0 = no cap)")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore existing per-run checkpoints and replay everything")
    cli = parser.parse_args(argv[1:])

    if not cli.experiment_dirs:
        from Src.Utils.Utils import EXPERIMENTS_DIR
        print(f"usage: python analysis/false_pruning.py <experiment_dir> ... "
              f"[--workers W] [--max-samples N] [--no-resume]\n"
              f"(experiments live under {EXPERIMENTS_DIR})")
        return

    max_samples = None if cli.max_samples in (0, None) else cli.max_samples
    for exp in cli.experiment_dirs:
        analyze(exp, workers=cli.workers, max_samples=max_samples, resume=not cli.no_resume)


if __name__ == "__main__":
    main(sys.argv)
