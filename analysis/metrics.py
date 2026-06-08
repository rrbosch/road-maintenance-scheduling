"""Pareto-front quality metrics (ported verbatim from the retained spec ``results_processing2.py``).

The metric definitions are unchanged so figures match the original pipeline:
* Hypervolume — area dominated between the (normalized) front and the reference point.
* Max Spread — Euclidean distance between the extreme normalized points.
* Min Distance to Origin — closest normalized point to (0, 0).
* Pareto Front Size — number of points.
Fronts are normalized by a per-case-study reference point before computing HV/distances.
"""
import numpy as np
import pandas as pd
from numba import njit

# Default reference point for Sioux Falls (Risk, TTD); other networks (e.g. Anaheim) need their own.
DEFAULT_REFERENCE_POINT = [2e3, 2e9]

METRIC_COLS = ["Hypervolume", "Max Spread", "Min Distance to Origin", "Pareto Front Size"]


@njit
def _calculate_hypervolume_numba(sorted_normalized_front):
    """Hypervolume of a 2-D minimization front already sorted ascending by the first objective."""
    hypervolume = 0.0
    prev_y = 1.0
    for i in range(len(sorted_normalized_front)):
        x = sorted_normalized_front[i, 0]
        y = sorted_normalized_front[i, 1]
        hypervolume += (1.0 - x) * (prev_y - y)
        prev_y = y
    return hypervolume


@njit
def _calculate_distances_numba(normalized_front):
    """Euclidean distance of each normalized point to the origin."""
    n_points = normalized_front.shape[0]
    distances = np.empty(n_points)
    for i in range(n_points):
        dist_sq = 0.0
        for j in range(normalized_front.shape[1]):
            dist_sq += normalized_front[i, j] ** 2
        distances[i] = np.sqrt(dist_sq)
    return distances


def return_non_dominated(arr):
    """Return the non-dominated rows of a 2-D minimization array (sort-and-sweep)."""
    sorted_arr = arr[np.argsort(arr[:, 0])]
    pareto = []
    min_second = np.inf
    for i in range(sorted_arr.shape[0]):
        if sorted_arr[i, 1] < min_second:
            pareto.append(sorted_arr[i])
            min_second = sorted_arr[i, 1]
    return np.array(pareto) if pareto else arr[:0]


def pareto_metrics(pareto_front, reference_point=DEFAULT_REFERENCE_POINT):
    """(Hypervolume, Max Spread, Min Distance to Origin, Pareto Front Size) for one front."""
    pareto_front = np.atleast_2d(np.asarray(pareto_front, dtype=float))
    reference_point = np.asarray(reference_point, dtype=float)
    normalized = np.clip(pareto_front / reference_point, 0, 1)
    sorted_norm = normalized[np.argsort(normalized[:, 0])]
    hypervolume = _calculate_hypervolume_numba(sorted_norm)
    max_spread = float(np.linalg.norm(sorted_norm[0, :] - sorted_norm[-1, :]))
    min_distance = float(np.min(_calculate_distances_numba(normalized)))
    size = len(pareto_front)
    return hypervolume, max_spread, min_distance, size


def per_generation_metrics(fronts_df, reference_point=DEFAULT_REFERENCE_POINT):
    """Compute the metrics for every generation from a run's long-format ``fronts.csv``.

    ``fronts_df`` has columns ``generation`` + the two objective columns. Returns a DataFrame
    indexed by generation with the METRIC_COLS.
    """
    obj_cols = [c for c in fronts_df.columns if c != 'generation']
    rows = []
    for gen, group in fronts_df.groupby('generation'):
        hv, spread, mind, size = pareto_metrics(group[obj_cols].values, reference_point)
        rows.append({'generation': int(gen), 'Hypervolume': hv, 'Max Spread': spread,
                     'Min Distance to Origin': mind, 'Pareto Front Size': size})
    return pd.DataFrame(rows).sort_values('generation').reset_index(drop=True)
