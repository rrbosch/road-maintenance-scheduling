from __future__ import print_function

import os
import pickle
import shutil
from collections import OrderedDict
from dataclasses import dataclass, field
from os import path
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

"""
Several utility functions for loading, saving and plotting
"""

# --- repo-anchored paths ---
# Locations are resolved relative to this file (Src/Utils/Utils.py), NOT os.getcwd(), so scripts
# work from any working directory. ROOT_DIR is the repo root (three parents up). When the project
# later becomes an installable package, only the parents[...] index here needs updating.
ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT_DIR / 'Environments' / 'input'      # case-study input data
EXPERIMENTS_DIR = ROOT_DIR / 'Experiments'           # generated run output


class FIFOCache:
    """A dict-like mapping that evicts the oldest-inserted entries once it grows past
    ``maxsize`` (first-in-first-out). Keys listed in ``pinned`` are never evicted.

    Used to cap the in-memory traffic-simulation cache (``TotalTravelDelay.results``) so long
    runs don't grow without bound. The baseline empty scenario ``frozenset()`` is pinned because
    it is read directly elsewhere as the network's base cost.

    Notes:
    - Pure FIFO: re-assigning an existing key keeps its original insertion position (it is not
      refreshed), so eviction order reflects insertion order, not access order.
    - ``maxsize=None`` disables eviction (legacy unbounded behavior).
    - Callers must not assume a value survives in the cache between writing and reading it (an
      ``update`` may evict older entries), so resolve values you need within a call into a local
      dict. ``maxsize`` is floored to a sane minimum to avoid pathological thrashing.
    """

    _MIN_MAXSIZE = 256

    def __init__(self, maxsize=200_000, pinned=()):
        self.pinned = set(pinned)
        if maxsize is not None:
            maxsize = max(int(maxsize), self._MIN_MAXSIZE)
        self.maxsize = maxsize
        self._d = OrderedDict()

    def _evict(self):
        if self.maxsize is None:
            return
        while len(self._d) > self.maxsize:
            for key in self._d:
                if key not in self.pinned:
                    del self._d[key]
                    break
            else:
                break  # everything left is pinned; nothing more to evict

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value):
        self._d[key] = value
        self._evict()

    def __contains__(self, key):
        return key in self._d

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return iter(self._d)

    def get(self, key, default=None):
        return self._d.get(key, default)

    def keys(self):
        return self._d.keys()

    def values(self):
        return self._d.values()

    def items(self):
        return self._d.items()

    def update(self, other):
        for key, value in dict(other).items():
            self._d[key] = value
        self._evict()


def convert_to_tuple(value):
    """Coerce a CSV cell (tuple/int/str-literal/NaN, or a whole Series) into a tuple of ints.

    Used when loading the case-study CSVs, whose link/list columns are stored as string literals
    like ``"(3, 4)"``. NaN / unparseable values become the empty tuple ``()``.
    """
    if isinstance(value, tuple):
        return value
    elif isinstance(value, pd.Series):
        a = value.apply(convert_to_tuple)
        return a
    elif pd.isna(value):
        return ()
    elif isinstance(value, int):
        return (value,)
    elif isinstance(value, str):
        try:
            _ = eval(value)
            if isinstance(_, int):
                return (_,)
            else:
                return _
        except:
            return ()
    else:
        return ()


# NOTE: per-generation result writing moved to Src/Utils/results_io.py (overhaul item 7); the old
# write_results (per-gen F_<gen>.csv/X_<gen>.csv + log.csv + config.txt) was removed.


def matrix_to_list_representation(x, problem):
    """Decision vector -> per-time-period project lists: ``x_list[t]`` = projects starting at t."""
    max_t = problem.input['general']['time periods']
    x_list = [[] for _ in range(max_t)]
    for index, time in enumerate(x):
        x_list[time].append(index)
    return x_list


def list_to_matrix_representation(x_list, problem):
    """Inverse of ``matrix_to_list_representation``: per-period project lists -> decision vector."""
    x = np.empty(problem.input['projects'].shape[0])
    for t, projects in enumerate(x_list):
        for project in projects:
            x[project] = t
    return x


def create_gantt_chart_csv_from_x(projects_path, x_path, gantt_path):
    """Expand a solutions file (one start-time vector per row) into a start/finish Gantt CSV."""
    from itertools import chain
    projects_df = pd.read_csv(projects_path)
    X = pd.read_csv(x_path, delimiter=";").to_numpy()

    solutions = range(X.shape[0])
    columns = [[f"Schedule {i} Start", f"Schedule {i} Finish"] for i in solutions]
    columns2 = list(chain(*columns))
    gantt_df = pd.DataFrame(data=None, index=range(X.shape[1]), columns=columns2)

    for x in range(X.shape[0]):
        start_times = X[x, :]
        end_times = start_times + projects_df['time periods']
        gantt_df[columns2[2*x]] = start_times
        gantt_df[columns2[2*x+1]] = end_times
    gantt_df.to_csv(path_or_buf=gantt_path, sep=";")


def flatten_list(nested_list):
    """Recursively flatten an arbitrarily nested list into a single flat list."""
    def flatten(lst):
        for item in lst:
            if isinstance(item, list):
                flatten(item)
            else:
                flat_list.append(item)

    flat_list = []
    flatten(nested_list)
    return flat_list


def frozensets_to_sparse_matrix(frozen_sets, num_cols):
    """One-hot encode scenarios (frozensets of project ids) as a sparse binary matrix.

    Each row is a scenario, each column a project; ``[i, j] = 1`` iff project ``j`` is in scenario
    ``i``. This is the feature matrix fed to the TTD surrogate regressor.
    """
    # Handle single frozenset input
    if isinstance(frozen_sets, frozenset):
        frozen_sets = [frozen_sets]

    # Create sparse matrix rows
    row_indices, col_indices = [], []
    for row_idx, fset in enumerate(frozen_sets):
        for elem in fset:
            row_indices.append(row_idx)
            col_indices.append(elem)

    # Create a sparse binary matrix
    num_rows = len(frozen_sets)
    data = np.ones(len(row_indices), dtype=np.int8)
    sparse_matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols))
    return sparse_matrix



def strict_pareto_optimal(parent_F: np.ndarray, child_f: np.ndarray) -> bool:
    """Did an offspring improve on its parents?

    Returns ``True`` when the offspring objective vector ``child_f`` is non-dominated with
    respect to its parents ``parent_F`` (one row per parent) -- i.e. no parent weakly dominates
    it. Used to credit the crossover/mutation operator that produced a useful offspring.
    Assumes minimization of every objective.
    """
    parent_F = np.atleast_2d(parent_F)
    for f in parent_F:
        # a parent that is <= the offspring in every objective weakly dominates it,
        # so the offspring is not an improvement over its parents
        if np.all(f <= child_f):
            return False
    return True


@dataclass
class LowerBoundInfo:
    """Lower-bound-screening state attached to a pymoo ``Individual`` as ``ind.data['lb']``.

    Written once per individual by ``Problem.get_lb`` (which fills ``F_lb`` and ``missing_info``);
    ``dominated`` is set afterwards by ``LowerBoundEvaluator.is_better_than_pareto_front``. This
    is the single owner of the LB state that used to live in the loose ``ind.data`` keys
    ``F_lb`` / ``F_mi`` / ``F_mi_size`` / ``dominated``.
    """
    F_lb: list                  # per-objective lower bounds, in problem.objectives order
    missing_info: list          # flat list of (sim_name, scenario, contribution) tuples for
                                # objective values still ESTIMATED (not exact); [] when exact
    dominated: bool = False      # True once the LB is found dominated by the current Pareto front

    @property
    def n_missing(self) -> int:
        """Number of still-estimated scenario contributions (was the ``F_mi_size`` key)."""
        return len(self.missing_info)


@dataclass
class OffspringOrigin:
    """Provenance of an offspring, attached as ``ind.data['origin']``.

    Lets ``CompositeCrossover``/``CompositeMutation.notify_successes`` credit the operators that
    produced an offspring which improved on its parents. Created by the crossover (which sets
    ``parent_F`` and ``crossover``); the mutation then records the operators it applied in
    ``mutations``. Replaces the loose ``ind.data`` keys ``parent F`` / ``crossover`` / ``mutation``.
    """
    parent_F: np.ndarray                            # parents' objective vectors (improvement test)
    crossover: int = -1                             # crossover operator id (-1 = copied through)
    mutations: list = field(default_factory=list)   # ids of mutation operators applied (empty if none)


if __name__ == "__main__":
    pass
