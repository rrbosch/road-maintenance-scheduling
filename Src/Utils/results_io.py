"""Per-run results writer — the on-disk schema the ``analysis/`` pipeline consumes.

One directory per run (``config.results_dir``) holds a small, fixed set of files instead of the
old per-generation ``F_<gen>.csv``/``X_<gen>.csv`` explosion:

  ``config.json``         run metadata (all ``Config`` attributes), machine-readable
  ``progress.csv``        one row per generation: timing, Pareto size, pruning diagnostics
  ``fronts.csv``          long format ``generation,<obj0>,<obj1>`` for *every* generation
  ``final_solutions.csv`` ``sol_idx,x0..xN`` start-time vectors of the latest front (for E3 Gantts)
  ``surrogate.csv``       one row per regressor retrain: ``n_computed,quantile,mape,pinball_loss``
  ``algo.pkl`` / ``algo_backup.pkl``  rolling pickles for crash-safe resume

``progress.csv`` and ``final_solutions.csv`` are rewritten in full each generation (cheap, and
resume-safe because they derive from the pickled ``algo``); ``fronts.csv``/``surrogate.csv`` are
appended.
"""
import json
import pickle
import shutil
from os import path

import numpy as np
import pandas as pd


def _jsonable(v):
    """Best-effort conversion of a Config attribute value to something json can serialize."""
    if isinstance(v, (set, frozenset)):
        return sorted((_jsonable(x) for x in v), key=str)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, type):
        return v.__name__
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def write_config(result_dir, config):
    """Write config.json once (machine-readable replacement for the old config.txt)."""
    cfg_path = path.join(result_dir, 'config.json')
    if path.exists(cfg_path):
        return
    data = {k: _jsonable(v) for k, v in config.__dict__.items()}
    with open(cfg_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def save_algo_pickle(result_dir, algo):
    """Atomically rotate algo.pkl -> algo_backup.pkl and write the fresh pickle (crash-safe)."""
    primary = path.join(result_dir, 'algo.pkl')
    backup = path.join(result_dir, 'algo_backup.pkl')
    temp = path.join(result_dir, 'algo_temp.pkl')
    with open(temp, 'wb') as f:
        pickle.dump(algo, f)
    if path.exists(primary):
        shutil.move(primary, backup)
    shutil.move(temp, primary)


def append_fronts(result_dir, generation, F, objective_names):
    """Append this generation's Pareto front to fronts.csv (long format, header if new)."""
    F = np.atleast_2d(np.asarray(F, dtype=float))
    df = pd.DataFrame(F, columns=list(objective_names))
    df.insert(0, 'generation', int(generation))
    fronts_path = path.join(result_dir, 'fronts.csv')
    df.to_csv(fronts_path, mode='a', header=not path.exists(fronts_path), index=False)


def write_final_solutions(result_dir, X):
    """(Re)write final_solutions.csv = start-time vectors of the latest front (for E3)."""
    X = np.atleast_2d(np.asarray(X))
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
    df.insert(0, 'sol_idx', range(X.shape[0]))
    df.to_csv(path.join(result_dir, 'final_solutions.csv'), index=False)


def write_progress(result_dir, log):
    """(Re)write progress.csv in full from the cumulative per-generation log list."""
    pd.DataFrame(log).to_csv(path.join(result_dir, 'progress.csv'), index=False)


def append_surrogate(result_dir, rows):
    """Append newly accumulated surrogate-accuracy rows to surrogate.csv."""
    if not rows:
        return
    sur_path = path.join(result_dir, 'surrogate.csv')
    pd.DataFrame(rows).to_csv(sur_path, mode='a', header=not path.exists(sur_path), index=False)


def write_generation(result_dir, config, algo, generation, log, F, X, objective_names,
                     surrogate_rows=()):
    """Persist everything for one generation: config (once), pickle, fronts, final X, progress.

    ``F``/``X`` are the current cumulative Pareto front's objective values / start-time vectors;
    ``log`` is the full per-generation record list; ``surrogate_rows`` are retrain-accuracy rows
    accumulated since the last call.
    """
    write_config(result_dir, config)
    save_algo_pickle(result_dir, algo)
    append_fronts(result_dir, generation, F, objective_names)
    write_final_solutions(result_dir, X)
    write_progress(result_dir, log)
    append_surrogate(result_dir, surrogate_rows)
