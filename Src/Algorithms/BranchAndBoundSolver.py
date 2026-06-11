"""Branch-and-bound exact Pareto solver (independent cross-check of `ExactParetoSolver`).

`BranchAndBoundSolver` finds the **exact** true Pareto front of a tiny scheduling instance
(SF-8 / SF-9) by a depth-first search over per-project start times that **prunes** dominated/
infeasible subtrees, instead of enumerating the whole feasible space the way
`ExactSolver.ExactParetoSolver` does. It writes the same `results_io.py` schema, so the
`analysis/` package consumes it unchanged. Its purpose is **verification + comparison**: it must
reproduce `ExactParetoSolver`'s front bit-for-bit (giving an independent ground-truth check), while
demonstrating how few schedules / traffic assignments a bounded search actually needs.

**Intended only for the tiny enumerable instances SF-9 / SF-8** — not SF-76 or Anaheim (the worst
case with no pruning is the full feasible set, and the per-period scenario machinery uses
P-bit/2^P-bounded frozensets).

Why the bounds are sound — both objectives are monotone:
  * **SL (Tardiness)** is additive across projects and increasing in each project's start, so a
    partial schedule's SL lower bound = (exact risk of assigned projects) + (each unassigned
    project's minimum possible risk, i.e. its risk at start 0 — a valid relaxation ignoring
    capacity/budget).
  * **TTD (TotalTravelDelay)** is non-decreasing in the per-period set of ongoing projects (crippling
    more links never lowers travel time), so the TTD of the *already-assigned* projects per period is
    a valid lower bound that tightens with depth and equals the exact TTD at a complete leaf.

Branching strategy (as specified): order projects by **slack low->high** (the most-constrained
project — fewest feasible start positions — first), the classic fail-first heuristic, so
capacity/deadline-infeasible regions are cut high in the tree. A two-stage bound check prunes
SL-dominated branches *before* paying for any traffic assignment.
"""
from os import path
from time import time

import numpy as np
from scipy.stats import binom

from Src.Utils import results_io

# Same guard as ExactParetoSolver: the per-period scenario frozensets and the 2^P scenario bound make
# this tractable only for tiny instances (and B&B's worst case is full enumeration anyway).
MAX_PROJECTS = 20

# Periodic progress dump cadence (seconds): on bigger tiny-instances (SF-10/SF-12) the search can run
# for minutes, so print a stats line roughly every REPORT_INTERVAL_SEC. The wall-clock check itself is
# only made every REPORT_EVERY_NODES explored nodes to keep `time()` off the hot path.
REPORT_INTERVAL_SEC = 300
REPORT_EVERY_NODES = 50_000


class BranchAndBoundSolver:
    """Branch-and-bound exact Pareto solver for tiny instances (SF-9 / SF-8). See module docstring."""

    def __init__(self, config):
        # Keep only lightweight, picklable state (config + a resume handle), never the big search
        # arrays — so the rolling algo.pkl stays small (mirrors ExactParetoSolver / NSGA2).
        self.config = config
        self.problem = None
        self.elapsed_time = 0.0
        self.done = False

    # ---- driver --------------------------------------------------------------
    def get_res(self, env):
        """Run the bounded search and write the exact Pareto front (single 'generation').

        Idempotent: if progress.csv already exists in the results dir, returns immediately (so
        re-running the registry is a safe no-op, like ExactParetoSolver / NSGA2).
        """
        self.problem = env
        result_dir = self.config.results_dir
        if path.exists(path.join(result_dir, 'progress.csv')):
            print(f"BranchAndBoundSolver: results already present in {result_dir}; nothing to do.")
            self.done = True
            return

        t0 = time()
        self._setup(env)

        # Run the depth-first bounded search from the root (nothing assigned): SL=0, TTD=T*base_cost.
        self._branch(0, 0.0, self.root_ttd)

        # Assemble the front in env.objectives key order; sort like ExactParetoSolver (lexicographic
        # by (SL, TTD) — the skyline of np.unique(F, axis=0)) so the two solvers' fronts line up.
        obj_names = list(env.objectives.keys())
        if self.front_F:
            F = np.array(self.front_F, dtype=float)
            X = np.array(self.front_X, dtype=int)
            order = np.lexsort(tuple(F[:, k] for k in range(F.shape[1] - 1, -1, -1)))
            front_F = F[order]
            front_X = X[order]
        else:
            raise RuntimeError("No feasible schedules — check the instance (budget/teams/deadlines).")

        print(f"BranchAndBoundSolver: exact Pareto front = {len(front_F)} points; "
              f"leaves evaluated = {self.leaves_evaluated:,}; nodes pruned (bound/infeasible) = "
              f"{self.nodes_pruned_bound:,}/{self.nodes_pruned_infeasible:,}; "
              f"unique sims = {env.sims['traffic'].n_computed}")

        # Write the standard results schema as one 'generation'.
        self.elapsed_time = time() - t0
        log = [{
            'iteration': 0,
            'pareto_set_size': int(len(front_F)),
            'time': self.elapsed_time,
            'time_cum': self.elapsed_time,
            'n_computed': env.sims['traffic'].n_computed,
            'exact_evals': int(self.leaves_evaluated),
            'lb_pruned': int(self.nodes_pruned_bound),
            'scenarios_materialized': 0,
            'n_estimated': 0,
            'false_pruned': 0,
            # B&B-specific diagnostics (write_progress keeps whatever keys the dicts carry).
            'nodes_explored': int(self.nodes_explored),
            'nodes_pruned_infeasible': int(self.nodes_pruned_infeasible),
            'nodes_pruned_bound': int(self.nodes_pruned_bound),
            'leaves_evaluated': int(self.leaves_evaluated),
        }]
        self.done = True
        results_io.write_generation(
            result_dir, self.config, self, 0, log, front_F, front_X,
            objective_names=obj_names, surrogate_rows=(),
        )
        print(f"BranchAndBoundSolver: wrote results to {result_dir} in {self.elapsed_time:.1f}s")

    # ---- setup ---------------------------------------------------------------
    def _setup(self, env):
        """Precompute instance arrays, the slack ordering, and the SL risk lookup tables."""
        proj = env.input['projects']
        self.dur = proj['duration'].values.astype(np.int64)
        hard = proj['hard due date'].values.astype(np.int64)
        self.cost = proj['cost'].values.astype(float)
        k_decay = proj['k_decay'].values.astype(np.int64)
        p_decay = proj['p_decay'].values.astype(float)
        # Latest feasible start: finish = start + dur <= hard (half-open [start, finish)), so the
        # domain is [0, hard - dur]. (ExactParetoSolver uses hard-dur+1 as a loop top then filters
        # finish<=hard; we bake the due-date constraint straight into the domain instead.)
        self.xu = (hard - self.dur).astype(int)
        self.T = int(env.input['general']['time periods'])
        self.cap = int(env.input['general']['construction teams'])
        self.budget = float(env.input['general']['budget'])
        self.P = len(self.dur)
        if self.P > MAX_PROJECTS:
            raise ValueError(f"BranchAndBoundSolver is for tiny instances only (P<={MAX_PROJECTS}); "
                             f"got P={self.P}. It is intended for SF-9/SF-8, not SF-76/Anaheim.")

        # (a) slack ordering: ascending number of feasible start positions => most-constrained first.
        slack = self.xu.astype(int)                           # range of starts (earliest=0)
        self.order = np.argsort(slack, kind='stable')

        # progress tracking for the periodic 5-min dump: per-depth branching factor (nominal start
        # options, in branch order) + cumulative product, plus the currently-explored start index per
        # depth. The explored-fraction is a mixed-radix "odometer" estimate over the nominal tree.
        self.branch_choices = np.array([self.xu[self.order[d]] + 1 for d in range(self.P)], dtype=float)
        self.cum_choices = np.cumprod(self.branch_choices)    # prod of choices for depths 0..d
        self.total_leaves = float(self.cum_choices[-1])       # nominal (unpruned) leaf count
        self.cur_idx = np.zeros(self.P, dtype=int)            # start index currently explored per depth
        self.t_start = time()
        self.last_report = self.t_start

        # (b) per-project SL risk lookup tables (exact match to ExactParetoSolver._compute_SL /
        #     Tardiness.risk_per_project: 1 - binom.cdf(k_decay, n=start, p=p_decay), times cost).
        self.risk_table = [
            (1 - binom.cdf(k_decay[j], np.arange(self.xu[j] + 1), p_decay[j])) * self.cost[j]
            for j in range(self.P)
        ]
        min_risk = np.array([rt.min() for rt in self.risk_table])   # risk at start 0 (relaxed LB)
        # suffix_min_risk[d] = sum of min_risk over projects not yet assigned at depth d (in branch order).
        self.suffix_min_risk = np.zeros(self.P + 1)
        for d in range(self.P - 1, -1, -1):
            self.suffix_min_risk[d] = self.suffix_min_risk[d + 1] + min_risk[self.order[d]]

        # available cumulative budget per period (no-op when budget is effectively disabled).
        self.avail_budget = self.budget * (np.arange(self.T) + 1)

        # (c) incremental DFS state.
        self.ttd = env.objectives['TTD']
        self.base_cost = self.ttd.results[frozenset()]        # seeded in Problem.set_objectives
        self.assigned = np.full(self.P, -1, dtype=int)        # depth-indexed chosen start
        self.cap_count = np.zeros(self.T, dtype=int)
        self.spend = np.zeros(self.T, dtype=float)
        self.period_set = [set() for _ in range(self.T)]      # assigned ongoing project ids per period
        self.period_cost = np.full(self.T, self.base_cost, dtype=float)
        # Root accumulators (passed DOWN the recursion as parameters, never mutated in place, so no
        # subtraction drift on backtrack — float-clean bounds, which matters at the exact-SL plateaus).
        self.root_ttd = float(self.base_cost) * self.T

        # incumbent Pareto front (parallel lists) + diagnostics.
        self.front_F = []                                     # list of [SL, TTD]
        self.front_X = []                                     # list of original-indexed start vectors
        self.nodes_explored = 0
        self.nodes_pruned_infeasible = 0
        self.nodes_pruned_bound = 0
        self.leaves_evaluated = 0

    # ---- search --------------------------------------------------------------
    def _branch(self, d, parent_sl, parent_ttd):
        """Assign a start to the depth-`d` project (in slack order), recursing over feasible starts.

        ``parent_sl`` / ``parent_ttd`` are the exact SL / TTD of the projects assigned at depths
        ``0..d-1`` — passed by value so each path accumulates by pure addition (no backtrack
        subtraction, hence float-clean bounds). The per-period state (``period_set``/``period_cost``/
        ``cap_count``/``spend``) is still mutated in place and reverted, but only by assignment.
        """
        p = int(self.order[d])          # original project id at this depth
        dur_p = int(self.dur[p])
        cost_p = float(self.cost[p])
        is_leaf = (d == self.P - 1)

        for s in range(self.xu[p] + 1):
            self.cur_idx[d] = s                  # odometer position at this depth (for progress %)
            self.nodes_explored += 1
            if self.nodes_explored % REPORT_EVERY_NODES == 0:
                self._maybe_report()
            finish = s + dur_p

            # 1) feasibility: capacity (g4) over the project's active periods.
            if np.any(self.cap_count[s:finish] + 1 > self.cap):
                self.nodes_pruned_infeasible += 1
                continue
            # budget (g3): only period s's spend changes; re-validate cumulative spend from s onward.
            if self.budget < 1e18:  # cheap skip when effectively disabled
                self.spend[s] += cost_p
                cum = np.cumsum(self.spend)
                budget_ok = np.all(cum[s:] <= self.avail_budget[s:])
                self.spend[s] -= cost_p
                if not budget_ok:
                    self.nodes_pruned_infeasible += 1
                    continue

            child_sl = parent_sl + self.risk_table[p][s]   # exact SL of depths 0..d

            # 2a) loose bound pre-check (NO new traffic sims): parent_ttd is a valid lower bound on
            #     this branch's TTD (monotone), and sl_lb adds the relaxed risk of all still-
            #     unassigned projects on top of the assigned SL.
            sl_lb = child_sl + self.suffix_min_risk[d + 1]
            if self._dominated_by_front(sl_lb, parent_ttd):
                self.nodes_pruned_bound += 1
                continue

            # 2b) apply the assignment: tighten the TTD bound by costing the now-larger per-period
            #     ongoing sets (cached; bounded by 2^P distinct scenarios).
            old_costs = {}
            self.spend[s] += cost_p
            self.assigned[d] = s
            child_ttd = parent_ttd
            for t in range(s, finish):
                self.cap_count[t] += 1
                old_costs[t] = self.period_cost[t]
                self.period_set[t].add(p)
                new_cost = self._scenario_cost(frozenset(self.period_set[t]))
                child_ttd += new_cost - self.period_cost[t]
                self.period_cost[t] = new_cost

            # re-check with the tightened TTD bound, then recurse or record the exact leaf.
            if self._dominated_by_front(sl_lb, child_ttd):
                self.nodes_pruned_bound += 1
            elif is_leaf:
                # complete schedule: record CLEAN exact objectives (summed the same way
                # ExactParetoSolver does — left-fold over projects for SL, np.sum over periods for
                # TTD) so the two solvers' front values match bit-for-bit and ties collapse.
                self.leaves_evaluated += 1
                self._offer_to_front(*self._leaf_objectives())
            else:
                self._branch(d + 1, child_sl, child_ttd)

            # 4) backtrack: revert the per-period mutations (assignment only, no arithmetic drift).
            for t in range(s, finish):
                self.cap_count[t] -= 1
                self.period_set[t].discard(p)
                self.period_cost[t] = old_costs[t]
            self.assigned[d] = -1
            self.spend[s] -= cost_p

    def _maybe_report(self):
        """Print a progress-stats line if at least REPORT_INTERVAL_SEC has elapsed since the last one.

        Reports nodes explored / pruned (bound + infeasible), leaves evaluated, current front size,
        unique traffic sims so far, and a mixed-radix estimate of how much of the nominal search tree
        has been covered (an *upper bound* on remaining work — pruning means the true fraction done is
        higher, so this is conservative).
        """
        now = time()
        if now - self.last_report < REPORT_INTERVAL_SEC:
            return
        self.last_report = now
        # odometer fraction over the nominal tree: sum_d cur_idx[d] / (prod of choices for 0..d).
        frac = float(np.sum(self.cur_idx / self.cum_choices))
        elapsed = now - self.t_start
        sims = self.problem.sims['traffic'].n_computed
        print(f"[B&B {elapsed/60:6.1f} min] explored={self.nodes_explored:,}  "
              f"pruned(bound/infeas)={self.nodes_pruned_bound:,}/{self.nodes_pruned_infeasible:,}  "
              f"leaves={self.leaves_evaluated:,}  front={len(self.front_F)}  sims={sims}  "
              f"tree-covered>={frac*100:.4f}% (of {self.total_leaves:,.0f} nominal leaves)",
              flush=True)

    def _leaf_objectives(self):
        """Exact (SL, TTD) of the current complete assignment, summed to match ExactParetoSolver.

        SL: left-fold of per-project risk-table lookups in original project order (mirrors
        ``_compute_SL``'s ``SL += risk[...]`` over j). TTD: ``np.sum`` over per-period costs (mirrors
        ``cost_lut[masks].sum(axis=1)``). Both reproduce the exact solver's arithmetic exactly.
        """
        x = self._current_x()
        sl = 0.0
        for j in range(self.P):
            sl += self.risk_table[j][x[j]]
        ttd = float(np.sum(self.period_cost))
        return sl, ttd, x

    def _scenario_cost(self, fs):
        """Exact cost of one scenario (frozenset of ongoing projects), via the shared TTD cache."""
        try:
            return self.ttd.results[fs]
        except KeyError:
            new = self.problem.sims['traffic'].get_multiple_scenarios([fs])
            self.ttd.results.update(new)
            return new[fs]

    # ---- incumbent Pareto front ----------------------------------------------
    def _dominated_by_front(self, sl, ttd):
        """True if some incumbent point weakly dominates (sl, ttd) componentwise (<= in both objs).

        Sound for pruning: every completion of a node is componentwise >= its lower bound, so a node
        whose LB is (weakly) dominated cannot yield a *new* (non-duplicate) front point.
        """
        for fsl, fttd in self.front_F:
            if fsl <= sl and fttd <= ttd:
                return True
        return False

    def _offer_to_front(self, sl, ttd, x):
        """Add an exact complete-solution point to the incumbent front, dropping dominated points."""
        # skip if weakly dominated by (or equal to) an existing point.
        for fsl, fttd in self.front_F:
            if fsl <= sl and fttd <= ttd:
                return
        # drop existing points now dominated by the new one.
        keep = [i for i, (fsl, fttd) in enumerate(self.front_F)
                if not (sl <= fsl and ttd <= fttd)]
        self.front_F = [self.front_F[i] for i in keep] + [[float(sl), float(ttd)]]
        self.front_X = [self.front_X[i] for i in keep] + [x]

    def _current_x(self):
        """Materialize the current complete assignment as an original-project-indexed start vector."""
        x = np.empty(self.P, dtype=int)
        for d in range(self.P):
            x[self.order[d]] = self.assigned[d]
        return x

    def resume(self):
        """Resume hook (run_single_instance.resume_run). One-shot, so this just re-runs (no-ops)."""
        self.get_res(self.problem)
