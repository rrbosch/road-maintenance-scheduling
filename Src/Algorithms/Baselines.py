import numpy as np
from scipy.stats import binom

from Src.Algorithms.Operators.Repair import TestRepair


def _repair_and_evaluate(problem, x, seed=0):
    """Repair a solution and evaluate it. Returns (x, F) or None if infeasible."""
    x = x.copy()
    repair = TestRepair(1000, seed)
    x = repair.scheduling_repair(problem, x)
    x = x.astype(int)
    out = {}
    problem._evaluate(x, out)
    F = np.array(out['F'])
    if np.any(np.isinf(F)):
        return None
    return x, F


def asap_by_deadline(problem, seed=0):
    """Schedule each project at the earliest feasible time, ordered by deadline."""
    projects = problem.input['projects']
    n = len(projects)
    order = projects['hard due date'].argsort().values

    x = np.zeros(n, dtype=int)
    for idx in order:
        x[idx] = 0  # earliest possible start

    result = _repair_and_evaluate(problem, x, seed)
    if result is None:
        return None
    return {'label': 'ASAP by deadline', 'x': result[0], 'F': result[1]}


def even_spread(problem, seed=0):
    """Distribute projects evenly across the time horizon, ordered by deadline."""
    projects = problem.input['projects']
    n = len(projects)
    xu = projects['hard due date'].values - projects['duration'].values + 1
    T_max = max(xu.max(), 1)

    order = projects['hard due date'].argsort().values
    x = np.zeros(n, dtype=int)
    for rank, idx in enumerate(order):
        t = int(rank * T_max / n)
        x[idx] = min(t, xu[idx])

    result = _repair_and_evaluate(problem, x, seed)
    if result is None:
        return None
    return {'label': 'Even spread', 'x': result[0], 'F': result[1]}


def threshold_risk_baselines(problem, thresholds=(0.01, 0.05, 0.1, 0.2, 0.5), seed=0):
    """For each threshold X, schedule project i at the first t where P(fail) > X."""
    projects = problem.input['projects']
    n = len(projects)
    xu = projects['hard due date'].values - projects['duration'].values + 1

    results = []
    for thresh in thresholds:
        x = np.zeros(n, dtype=int)
        for i in range(n):
            p = projects['p_decay'].iloc[i]
            k = projects['k_decay'].iloc[i]
            upper = xu[i]
            placed = False
            for t in range(upper + 1):
                prob_fail = 1 - binom.cdf(k=k, n=t, p=p)
                if prob_fail > thresh:
                    x[i] = min(t, upper)
                    placed = True
                    break
            if not placed:
                x[i] = upper

        result = _repair_and_evaluate(problem, x, seed)
        if result is not None:
            results.append({'label': f'p(fail) > {thresh}', 'x': result[0], 'F': result[1]})
    return results


def compute_all_baselines(problem, seed=0):
    """Compute all baseline rules and return list of {'label', 'x', 'F'} dicts."""
    baselines = []

    asap = asap_by_deadline(problem, seed)
    if asap is not None:
        baselines.append(asap)

    even = even_spread(problem, seed)
    if even is not None:
        baselines.append(even)

    baselines.extend(threshold_risk_baselines(problem, seed=seed))

    return baselines
