"""Exact Pareto solver (campaign E1 ground truth) — overhaul item 16.

`ExactParetoSolver` enumerates the **full feasible decision space** of a small scheduling instance
and returns the **exact** non-dominated (true Pareto) front, writing the standard `results_io.py`
schema so the `analysis/` package consumes it unchanged. It is the exact-optimum reference for
campaign E1: it lets us measure each method's true HV gap and PLBE's *actual* (not bounded)
false-pruning rate, and anchors the SF-9 -> SF-76 -> Anaheim size-scaling story at the small end.

**Intended only for the tiny enumerable instances SF-9 / SF-8** — not SF-76 or Anaheim.

Speed (≈1-2 min on SF-9: 711k feasible of 10.5M nominal). The naive "call Problem._evaluate on
every feasible schedule" is ~1 h (pandas `get_x_dict` ~5 ms/eval). Instead we exploit structure:

  * The number of **distinct traffic scenarios** (a scenario = the frozenset of projects ongoing in
    one period) is bounded by 2^P, *independent* of the number of feasible schedules. Traffic
    assignments (the only expensive op) are therefore <= 2^P; everything else is vectorized numpy.

Pipeline (see the module functions): (1) vectorized feasibility enumeration -> S_feas; (2) SL via a
per-project risk lookup table; (3) TTD via per-period scenario bitmasks + a cost LUT filled by <=2^P
cached assignments; (4) exact 2-objective non-dominated sweep; (5) write the single-"generation"
results schema.
"""
from os import path
from time import time

import numpy as np
from scipy.stats import binom

from Src.Utils import results_io

# Guard: the TTD cost LUT is indexed by a P-bit scenario mask (size 2^P), and full enumeration is
# only tractable for tiny instances anyway. Refuse anything that is clearly not SF-9/SF-8-sized.
MAX_PROJECTS = 20


class ExactParetoSolver:
    """Full-enumeration exact Pareto solver for tiny instances (SF-9 / SF-8). See module docstring."""

    def __init__(self, config):
        # Keep only lightweight, picklable state (config + a resume handle), never the big arrays —
        # so the rolling algo.pkl stays small (mirrors how NSGA2 is pickled).
        self.config = config
        self.problem = None
        self.elapsed_time = 0.0
        self.done = False

    # ---- enumeration helpers -------------------------------------------------
    @staticmethod
    def _cartesian(arrays):
        """Cartesian product of 1-D integer arrays -> (prod, k) matrix (int16)."""
        mesh = np.meshgrid(*arrays, indexing='ij')
        return np.stack([m.ravel() for m in mesh], axis=1).astype(np.int16)

    def _enumerate_feasible(self, dur, hard, cost, T, cap, budget):
        """Vectorized enumeration of all feasible start vectors.

        Constraints mirror `Problem.check_scheduling_constraints` exactly (g1 release is auto-met
        since every start >= 0; g2 due, g3 budget, g4 capacity). Chunked over project 0's start
        domain to bound peak memory to one chunk. Uses the half-open [start, start+duration) ongoing
        convention (item-15 fix). Returns (S_feas (N x P int16), total_enumerated).
        """
        P = len(dur)
        xu = (hard - dur + 1).astype(int)            # latest start so finish<=hard would allow; see g2
        domains = [np.arange(u + 1, dtype=np.int16) for u in xu]
        avail = budget * (np.arange(T) + 1)          # cumulative budget available by period
        feasible_chunks = []
        total = 0
        for s0 in domains[0]:
            if P > 1:
                rest = self._cartesian(domains[1:])              # (M, P-1)
                S = np.empty((rest.shape[0], P), dtype=np.int16)
                S[:, 0] = s0
                S[:, 1:] = rest
            else:
                S = np.array([[s0]], dtype=np.int16)
            total += S.shape[0]
            finish = S + dur
            ok = (finish <= hard).all(axis=1)                    # g2 due date
            # g4 capacity: peak simultaneous projects <= cap
            peak = np.zeros(S.shape[0], dtype=np.int16)
            for t in range(T):
                peak = np.maximum(peak, ((S <= t) & (t < finish)).sum(axis=1).astype(np.int16))
            ok &= peak <= cap
            # g3 budget: cumulative spend <= cumulative budget every period (no-op when disabled)
            if np.any(ok):
                spend = np.stack([np.where(S == t, cost, 0).sum(axis=1) for t in range(T)], axis=1)
                ok &= (np.cumsum(spend, axis=1) <= avail).all(axis=1)
            if np.any(ok):
                feasible_chunks.append(S[ok].copy())
        S_feas = (np.vstack(feasible_chunks) if feasible_chunks
                  else np.empty((0, P), dtype=np.int16))
        return S_feas, total

    # ---- objectives ----------------------------------------------------------
    @staticmethod
    def _compute_SL(S_feas, k_decay, p_decay, cost, xu):
        """Vectorized Tardiness (SL) via a per-project risk lookup table.

        Matches `Objectives.Tardiness.risk_per_project`: per started project the penalty probability
        is `1 - binom.cdf(k_decay, n=start, p=p_decay)` times `cost` (note: the live code passes
        k=k_decay to binom.cdf, so we replicate that exactly to reproduce the GA's SL values).
        """
        N, P = S_feas.shape
        SL = np.zeros(N, dtype=float)
        for j in range(P):
            s = np.arange(int(xu[j]) + 1)
            risk = (1 - binom.cdf(k_decay[j], s, p_decay[j])) * cost[j]   # table over start values
            SL += risk[S_feas[:, j]]
        return SL

    def _compute_TTD(self, S_feas, dur, env):
        """Vectorized TTD via per-period scenario bitmasks + a cost LUT.

        Each (schedule, period) ongoing-set is encoded as a P-bit integer. Unique masks (<=2^P) are
        costed once via the same path the GA uses (`TotalTravelDelay.results` cache +
        `TrafficSimulation.get_multiple_scenarios`), so values are identical. Then
        TTD = sum over periods of cost_lut[mask], reproducing `TotalTravelDelay.get_value`.
        """
        N, P = S_feas.shape
        T = env.input['general']['time periods']
        finish = S_feas + dur
        bitw = (1 << np.arange(P)).astype(np.int64)
        masks = np.empty((N, T), dtype=np.int64)
        for t in range(T):
            member = (S_feas <= t) & (t < finish)            # (N, P) bool
            masks[:, t] = member.astype(np.int64) @ bitw     # P-bit scenario id per schedule
        ttd = env.objectives['TTD']
        unique_masks = np.unique(masks)
        # decode masks -> frozensets, cost the misses once (cached for reuse)
        mask_to_fs = {int(m): frozenset(j for j in range(P) if (int(m) >> j) & 1)
                      for m in unique_masks}
        missing = [fs for m, fs in mask_to_fs.items() if fs not in ttd.results]
        if missing:
            new_costs = env.sims['traffic'].get_multiple_scenarios(missing)
            ttd.results.update(new_costs)
        cost_lut = np.zeros(int(unique_masks.max()) + 1, dtype=float)
        for m, fs in mask_to_fs.items():
            cost_lut[m] = ttd.results[fs]
        return cost_lut[masks].sum(axis=1)

    # ---- non-dominated front -------------------------------------------------
    @staticmethod
    def _pareto_front(F):
        """Return (front_F, front_idx) — exact non-dominated set (minimization, all objectives).

        2 objectives: O(n log n) skyline sweep on lexicographically-sorted unique rows. >2: pymoo
        fallback. `front_idx` indexes into the rows of `F` (first occurrence of each front point).
        """
        uniqueF, first_idx = np.unique(F, axis=0, return_index=True)   # lex-sorted by (F0, F1, ...)
        if uniqueF.shape[1] == 2:
            keep = np.zeros(len(uniqueF), dtype=bool)
            best_f1 = np.inf
            for i in range(len(uniqueF)):           # rows sorted by F0 asc, then F1 asc
                if uniqueF[i, 1] < best_f1:
                    keep[i] = True
                    best_f1 = uniqueF[i, 1]
            nd = np.nonzero(keep)[0]
        else:
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
            nd = NonDominatedSorting().do(uniqueF, only_non_dominated_front=True)
        return uniqueF[nd], first_idx[nd]

    # ---- driver --------------------------------------------------------------
    def get_res(self, env):
        """Enumerate, evaluate exactly, and write the true Pareto front (single 'generation').

        Idempotent: if progress.csv already exists in the results dir, returns immediately (so
        re-running the registry is a safe no-op, like NSGA2's spent-budget resume).
        """
        self.problem = env
        result_dir = self.config.results_dir
        if path.exists(path.join(result_dir, 'progress.csv')):
            print(f"ExactParetoSolver: results already present in {result_dir}; nothing to do.")
            self.done = True
            return

        t0 = time()
        proj = env.input['projects']
        dur = proj['duration'].values.astype(np.int64)
        hard = proj['hard due date'].values.astype(np.int64)
        cost = proj['cost'].values.astype(float)
        k_decay = proj['k_decay'].values.astype(np.int64)
        p_decay = proj['p_decay'].values.astype(float)
        xu = (hard - dur + 1).astype(int)
        T = int(env.input['general']['time periods'])
        cap = int(env.input['general']['construction teams'])
        budget = float(env.input['general']['budget'])
        P = len(dur)
        if P > MAX_PROJECTS:
            raise ValueError(f"ExactParetoSolver is for tiny instances only (P<={MAX_PROJECTS}); "
                             f"got P={P}. It is intended for SF-9/SF-8, not SF-76/Anaheim.")

        # 1) feasible decision space
        S_feas, total = self._enumerate_feasible(dur, hard, cost, T, cap, budget)
        n_feas = S_feas.shape[0]
        print(f"ExactParetoSolver: enumerated {total:,} schedules -> {n_feas:,} feasible "
              f"({100 * n_feas / max(total, 1):.2f}%)")
        if n_feas == 0:
            raise RuntimeError("No feasible schedules — check the instance (budget/teams/deadlines).")

        # 2)+3) exact objectives, in env.objectives key order
        obj_names = list(env.objectives.keys())
        columns = {}
        if 'SL' in env.objectives:
            columns['SL'] = self._compute_SL(S_feas, k_decay, p_decay, cost, xu)
        if 'TTD' in env.objectives:
            columns['TTD'] = self._compute_TTD(S_feas, dur, env)
        F = np.column_stack([columns[name] for name in obj_names])

        # 4) exact non-dominated front + a representative start vector per front point
        front_F, front_idx = self._pareto_front(F)
        front_X = S_feas[front_idx].astype(int)
        print(f"ExactParetoSolver: exact Pareto front = {len(front_F)} points; "
              f"unique sims = {env.sims['traffic'].n_computed}")

        # 5) write the standard results schema as one 'generation'
        self.elapsed_time = time() - t0
        log = [{
            'iteration': 0,
            'pareto_set_size': int(len(front_F)),
            'time': self.elapsed_time,
            'time_cum': self.elapsed_time,
            'n_computed': env.sims['traffic'].n_computed,
            'exact_evals': int(n_feas),
            'lb_pruned': 0,
            'scenarios_materialized': 0,
            'n_estimated': 0,
            'false_pruned': 0,
            'total_enumerated': int(total),
            'feasible': int(n_feas),
        }]
        self.done = True
        results_io.write_generation(
            result_dir, self.config, self, 0, log, front_F, front_X,
            objective_names=obj_names, surrogate_rows=(),
        )
        print(f"ExactParetoSolver: wrote results to {result_dir} in {self.elapsed_time:.1f}s")

    def resume(self):
        """Resume hook (run_single_instance.resume_run). One-shot, so this just re-runs (no-ops)."""
        self.get_res(self.problem)
