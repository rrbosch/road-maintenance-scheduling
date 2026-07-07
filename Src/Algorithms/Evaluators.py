"""Population evaluators — including the research core that avoids expensive traffic sims.

Three strategies, increasingly clever about *not* running the exact (expensive) evaluation:
* ``StandardEvaluator`` — plain pymoo: evaluate every individual exactly.
* ``LowerBoundEvaluator`` — for each feasible individual, compute a lower bound on its objectives;
  if that bound is already dominated by the current Pareto front, skip the exact evaluation
  entirely. Otherwise iteratively materialize the most impactful estimated traffic scenario and
  re-check, only doing the full exact evaluation once nothing is estimated anymore.
* ``ApproximateEvaluator`` — lower-bound-screen the whole population first, then exactly evaluate
  only the non-dominated survivors.
"""
from collections import defaultdict

import numpy as np
from pymoo.core.evaluator import Evaluator
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import find_non_dominated

from Src.Utils.Utils import LowerBoundInfo


class StandardEvaluator(Evaluator):
    """Plain exact evaluation of every individual (pymoo's default behavior)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LowerBoundEvaluator(Evaluator):
    """Skip exact evaluation of individuals whose lower bound is already Pareto-dominated.

    Per feasible individual: compute the LB, and while it is not dominated but still has estimated
    (not-exact) traffic scenarios, materialize the most impactful one to tighten the bound. Only
    when the bound is exact and still non-dominated do we run the full ``problem.evaluate``.
    """
    def __init__(self):
        super().__init__()
        self.pareto_front = None
        self.algorithm = None
        self.new_information_strategy = "impact"  # which estimated scenario to materialize first:
                                                  # the one with the largest contribution to the LB
        self.dominated_solutions = []  # LB-dominated solutions recorded per generation (for output)
        # E2 (item 12, log-and-replay): sampled pruned candidates (x + incumbent front snapshot) to
        # write to pruned_sample.csv each generation. Logging-only — see _record_dominated_solution.
        self.pruned_samples = []
        self._prune_log_rng = None  # lazily seeded from problem.seed_value (reproducible sampling)
        self._prune_sample_counter = 0  # monotone sample id across the whole run (survives resume)
        # Per-generation pruning diagnostics (read + reset each generation by NSGA2.get_res).
        self.reset_diagnostics()

    def reset_diagnostics(self):
        """Zero the per-generation pruning counters (telemetry for progress.csv)."""
        self.n_exact_evals = 0          # individuals given a full exact problem.evaluate
        self.n_lb_pruned = 0            # individuals discarded on their lower bound alone
        self.n_scenarios_materialized = 0  # estimated scenarios turned exact via add_scenario
        self.n_estimated = 0            # estimated scenario-contributions seen across LB computations
        self.n_false_pruned = 0         # E2 diagnostic: pruned solutions exact-eval would have kept

    def _eval(self, problem, pop, evaluate_values_of, **kwargs):
        # evaluate individuals one by one (each may trigger several scenario materializations)
        for ind in pop:
            x = ind.get("X")
            x_dict = problem.get_x_dict(x)
            G = problem.check_scheduling_constraints(x_dict)
            ind.set("G", G)
            if not ind.feasible:
                F = np.array([np.inf for _ in problem.objectives])
                ind.set("F", F)
                ind.set("H", [])
                ind.evaluated.update(('F', 'G', 'H'))
                continue
            else:
                # if feasible, start evaluating lower bounds
                while True:
                    self.evaluate_lower_bounds(ind, problem)
                    self.is_better_than_pareto_front(ind)

                    # if the individual is dominated based on the lb's, remove it
                    if ind.data['lb'].dominated:
                        # Record the dominated solution
                        self._record_dominated_solution(ind, iteration=self.algorithm.n_gen - 1 if self.algorithm else None)
                        break
                    if ind.data['lb'].n_missing > 0:
                        self.fill_missing_information(ind, problem)
                    else:
                        break
            if ind.data['lb'].dominated:
                F = np.array([np.inf for _ in problem.objectives])
                ind.set("F", F)
                ind.set("H", [])
                ind.evaluated.update(('F', 'G', 'H'))
                continue

            # once you're done updating lower bounds,
            x = ind.get("X")
            out = problem.evaluate(x, return_values_of=evaluate_values_of, return_as_dictionary=True, **kwargs)
            self.n_exact_evals += 1
            for key, item in out.items():
                ind.set(key, item)
            ind.evaluated.update(out.keys())
            new_opt = Population.merge(ind, self.algorithm.opt)
            if isinstance(new_opt, Individual):
                new_opt = Population(new_opt)
            self.algorithm.opt = new_opt
        new_pareto = update_true_pareto_front(self.algorithm.opt)
        self.algorithm.opt = new_pareto


    def evaluate_lower_bounds(self, ind, problem):
        # Compute the lower bound + missing information (stored on ind.data['lb']).
        problem.get_lb(ind)
        # diagnostic: how many scenario contributions are still estimated for this LB computation
        self.n_estimated += ind.data['lb'].n_missing

    def fill_missing_information(self, ind, problem):
        """Materialize the single most impactful estimated traffic scenario.

        Each missing-info entry is ``(sim, scenario, contribution)`` -- the estimated cost a
        traffic ``scenario`` (a frozenset of ongoing projects) contributes to this individual's
        lower bound. A scenario can appear in several time periods, so we sum its contributions
        and run the real assignment for the scenario with the largest total
        (``add_scenario``). On the next LB recomputation that scenario uses its exact cost, so the
        bound tightens and the loop in ``_eval`` re-checks domination. Only the ``traffic`` sim
        produces missing info (``Tardiness.get_lower_bound`` returns ``[]``).
        """
        contribution_by_scenario = defaultdict(float)
        for sim, scenario, contribution in ind.data['lb'].missing_info:
            if sim != 'traffic':
                raise NotImplementedError(f"missing-information handling for sim '{sim}' not implemented")
            contribution_by_scenario[scenario] += contribution

        if contribution_by_scenario:
            most_impactful = max(contribution_by_scenario, key=contribution_by_scenario.get)
            problem.objectives['TTD'].add_scenario(most_impactful, problem)
            self.n_scenarios_materialized += 1

    def is_better_than_pareto_front(self, ind):
        if self.algorithm.opt is None:
            ind.data['lb'].dominated = False
        else:
            F = np.atleast_2d(ind.data['lb'].F_lb)
            _F = np.atleast_2d(self.algorithm.opt.get("F"))
            ind_nd = find_non_dominated(F=F, _F=_F)
            if len(ind_nd) > 0:
                ind.data['lb'].dominated = False
            else:
                ind.data['lb'].dominated = True

    def _record_dominated_solution(self, ind, iteration=None):
        """Record a dominated (LB-pruned) solution for logging and file output."""
        self.n_lb_pruned += 1
        lb_info = ind.data.get('lb', None)
        dominated_record = {
            'X': ind.get("X").copy(),
            'F_lb': lb_info.F_lb.copy() if lb_info is not None and lb_info.F_lb is not None else None,
            'iteration': iteration,
        }
        self.dominated_solutions.append(dominated_record)
        # E2 (item 12, log-and-replay): metric-neutral sampling logger (no exact eval, no sim).
        self._maybe_log_pruned_sample(ind, iteration)
        # E2 (item 12): in the false-pruning diagnostic mode, check whether this just-pruned
        # solution would actually have survived an exact evaluation. (Inline, sim-consuming — kept
        # intact and complementary to the log-only sampler above.)
        self._maybe_count_false_pruning(ind)

    def _maybe_log_pruned_sample(self, ind, iteration):
        """LOG ONLY: with probability ``problem.false_pruning_log_prob``, snapshot this pruned
        candidate for post-hoc false-pruning replay (``analysis/false_pruning.py``).

        CRITICAL — metric neutrality. This method must never exact-evaluate, never run a traffic
        assignment, never touch/populate the scenario cache, and never mutate any run counter
        (``n_computed``, the per-gen diagnostics, iterations, timing). It only:
          1. draws a Bernoulli(p) from a *dedicated* seeded RNG (does not perturb numpy's global
             RNG, ``random``, or the algorithm RNG that drive the optimization), and
          2. on a hit, appends an in-memory record: the pruned decision vector ``x`` and a snapshot
             of the incumbent Pareto front objective vectors (``self.algorithm.opt.F``, already in
             memory — read, not modified) at prune time.
        Both ``x`` and the front are *copied* so later mutation of the population / front cannot
        corrupt a recorded sample. With the hook off (p=0) it returns immediately. The recorded
        rows are drained to ``pruned_sample.csv`` by NSGA2.get_res, mirroring ``dominated_solutions``.
        """
        problem = getattr(self.algorithm, 'problem', None)
        if problem is None:
            return
        p = getattr(problem, 'false_pruning_log_prob', 0.0)
        if not p or p <= 0.0:
            return
        # Lazily seed a dedicated RNG from the same algo_seed-derived seed surrogate_noise uses.
        if self._prune_log_rng is None:
            self._prune_log_rng = np.random.default_rng(getattr(problem, 'seed_value', 0))
        if p < 1.0 and self._prune_log_rng.random() >= p:
            return  # not sampled this time

        x = ind.get("X")
        # Snapshot the incumbent front objective vectors at prune time (copy so it's frozen).
        opt = getattr(self.algorithm, 'opt', None)
        if opt is not None:
            front_F = np.atleast_2d(np.asarray(opt.get("F"), dtype=float)).copy()
        else:
            front_F = np.empty((0, len(getattr(problem, 'objectives', [None, None]))))
        sample_id = self._prune_sample_counter
        self._prune_sample_counter += 1
        self.pruned_samples.append({
            'sample_id': sample_id,
            'iteration': iteration,
            'X': np.asarray(x).copy(),
            'front_F': front_F,
        })

    def clear_pruned_samples(self):
        """Clear the per-generation pruned-sample buffer after writing to file."""
        self.pruned_samples = []

    def _maybe_count_false_pruning(self, ind):
        """Count a *false prune*: a solution discarded by the LB/surrogate screen whose true
        objectives are non-dominated by the current Pareto front (i.e. it should have been kept).

        Only runs when ``problem.count_false_pruning`` is set — it exactly evaluates the pruned
        solution (extra sims), so it is a measurement mode for a few seeds, not for production.
        With a *valid* lower bound this can never happen (LB <= true F, so an LB dominated by the
        front implies the true F is dominated too); it becomes positive exactly when the surrogate
        over-predicts (e.g. injected noise, or too high a quantile) — the EP-vs-LE robustness study.
        """
        problem = getattr(self.algorithm, 'problem', None)
        if problem is None or not getattr(problem, 'count_false_pruning', False):
            return
        x = ind.get("X")
        out = problem.evaluate(x, return_as_dictionary=True)
        true_F = np.atleast_2d(np.asarray(out['F'], dtype=float))
        if not np.all(np.isfinite(true_F)):
            return  # infeasible / no real objective => not a false prune
        if self.algorithm.opt is None:
            self.n_false_pruned += 1  # nothing to dominate it yet => it would have been kept
            return
        _F = np.atleast_2d(self.algorithm.opt.get("F"))
        if len(find_non_dominated(F=true_F, _F=_F)) > 0:
            self.n_false_pruned += 1

    def clear_dominated_solutions(self):
        """Clear the dominated solutions list after writing to file."""
        self.dominated_solutions = []

class ApproximateEvaluator(LowerBoundEvaluator):
    """Population-level screen: lower-bound every individual, exactly evaluate only the survivors.

    Unlike ``LowerBoundEvaluator`` it does a single LB pass over the whole population (no iterative
    scenario refinement): LB-dominated individuals get ``F = inf``; the rest are evaluated exactly.
    """
    def _eval(self, problem, pop, evaluate_values_of, **kwargs):
        # pass 1: cheap lower-bound screen of the whole population
        for ind in pop:
            self.evaluate_lower_bounds(ind, problem)
            self.is_better_than_pareto_front(ind)

        # pass 2: exactly evaluate only the LB-non-dominated; mark the rest dominated (F = inf)
        for ind in pop:
            if ind.data['lb'].dominated:
                # Record the dominated solution
                self._record_dominated_solution(ind, iteration=self.algorithm.n_gen - 1 if self.algorithm else None)

                x_dict = problem.get_x_dict(ind.x)
                out = {'F': np.array([np.inf] * len(ind.data['lb'].F_lb)),
                       'G': problem.check_scheduling_constraints(x_dict),
                       'H': ind.H}
            else:
                x = ind.get("X")
                out = problem.evaluate(x, return_values_of=evaluate_values_of, return_as_dictionary=True, **kwargs)
                self.n_exact_evals += 1
                ind.evaluated.update(out.keys()) # This marks a solution as evaluated and will thus be skipped next time
            for key, item in out.items():
                ind.set(key, item)

        # Update the pareto optimal set
        # new_pareto = update_true_pareto_front(self.algorithm.opt, pop)
        # self.algorithm.opt = new_pareto


class ScheduleSurrogateEvaluator(LowerBoundEvaluator):
    """E1 control (revision item 11): standard surrogate-assisted NSGA-II via whole-schedule pre-selection.

    Implements the textbook surrogate-assisted-EA recipe — an **absolute-fitness regression**
    surrogate predicts each offspring's objective and only the most promising offspring are
    evaluated exactly: **individual-based evolution control by pre-selection** (Jin, Y. 2011,
    *Surrogate-assisted evolutionary computation*, Swarm & Evol. Comput. 1(2):61-70;
    Díaz-Manríquez et al. 2016, *A Review of Surrogate-Assisted Multiobjective Evolutionary
    Algorithms*, Comput. Intell. Neurosci. 2016:9420460). The surrogate here is a whole-schedule
    TTD point predictor (``Objectives.ScheduleLevelSurrogate``, mirroring Mao et al. 2021); "most
    promising" = predicted non-dominated w.r.t. the current Pareto front. SL (Tardiness) is
    closed-form, so it is screened exactly.

    This is the deliberate *control* against PLBE (``LowerBoundEvaluator``): unlike PLBE it screens
    with a **point estimate, not a lower bound** (no soundness guarantee) at the **whole-schedule**
    level with **no per-scenario progressive refinement**. Comparing the two isolates where PLBE's
    simulation savings come from (component-level surrogacy + progressive pruning), rather than from
    "using ML". Structurally it mirrors ``ApproximateEvaluator`` (screen the whole population, then
    exactly evaluate only the survivors).

    Shared progress.csv diagnostics take this evaluator's meaning: ``n_exact_evals`` = schedules
    given a full exact evaluation (the sims); ``n_lb_pruned`` = schedules pruned by the surrogate
    screen; ``n_scenarios_materialized`` = 0 (no per-scenario refinement); ``n_estimated`` =
    surrogate point predictions made.
    """

    def __init__(self):
        super().__init__()
        self.surrogate = None  # lazily created on first _eval (needs problem.n_var)

    def _ensure_surrogate(self, problem):
        """Lazily build the schedule surrogate and (re)bind its log to ``TTD.surrogate_log``.

        Rebinding every generation means NSGA2.get_res (which drains
        ``env.objectives['TTD'].surrogate_log``) picks up our rows with no change, and a resumed run
        binds to the freshly created problem's log.
        """
        if self.surrogate is None:
            from Environments.env.Objectives import ScheduleLevelSurrogate
            q = getattr(problem, 'schedule_surrogate_quantile', 0.5)
            self.surrogate = ScheduleLevelSurrogate(
                problem.n_var, quantile=q, model='XGBoost',
                surrogate_noise=getattr(problem, 'surrogate_noise', 0.0),
                noise_seed=getattr(problem, 'seed_value', 0))
        ttd = problem.objectives.get('TTD')
        if ttd is not None:
            self.surrogate.surrogate_log = ttd.surrogate_log

    def evaluate_lower_bounds(self, ind, problem):
        """Build the *predicted* objective vector F_hat = [<exact cheap objs>, TTD_pred].

        Stored on ``ind.data['lb']`` as a LowerBoundInfo (no missing info) so the inherited
        ``is_better_than_pareto_front`` screen works unchanged. Before the surrogate is trained the
        TTD prediction is ``-inf`` so the individual is treated as non-dominated and gets an exact
        evaluation (warming up the surrogate's training set).
        """
        x = ind.get("X")
        x_dict = problem.get_x_dict(x)
        F_hat = []
        for key, objective in problem.objectives.items():
            if key == 'TTD':
                pred = self.surrogate.predict([x])
                F_hat.append(float(pred[0]) if pred is not None else -np.inf)
                self.n_estimated += 1
            else:
                F_hat.append(objective.get_value(problem, x_dict))  # closed-form, exact (SL)
        ind.data['lb'] = LowerBoundInfo(F_lb=np.array(F_hat, dtype=float), missing_info=[])

    def _eval(self, problem, pop, evaluate_values_of, **kwargs):
        self._ensure_surrogate(problem)
        ttd_idx = list(problem.objectives.keys()).index('TTD')

        # pass 1: feasibility check + cheap surrogate screen of the whole population
        for ind in pop:
            x = ind.get("X")
            x_dict = problem.get_x_dict(x)
            G = problem.check_scheduling_constraints(x_dict)
            ind.set("G", G)
            if not ind.feasible:
                F = np.array([np.inf for _ in problem.objectives])
                ind.set("F", F)
                ind.set("H", [])
                ind.evaluated.update(('F', 'G', 'H'))
                ind.data['lb'] = LowerBoundInfo(F_lb=F, missing_info=[], dominated=True)
                continue
            self.evaluate_lower_bounds(ind, problem)
            self.is_better_than_pareto_front(ind)

        # pass 2: exactly evaluate predicted-non-dominated survivors; prune the rest (F = inf)
        for ind in pop:
            if not ind.feasible:
                continue
            if ind.data['lb'].dominated:
                self._record_dominated_solution(
                    ind, iteration=self.algorithm.n_gen - 1 if self.algorithm else None)
                F = np.array([np.inf] * len(ind.data['lb'].F_lb))
                ind.set("F", F)
                ind.set("H", [])
                ind.evaluated.update(('F', 'G', 'H'))
            else:
                x = ind.get("X")
                out = problem.evaluate(x, return_values_of=evaluate_values_of,
                                       return_as_dictionary=True, **kwargs)
                self.n_exact_evals += 1
                for key, item in out.items():
                    ind.set(key, item)
                ind.evaluated.update(out.keys())
                # learn: feed the exact (x -> true TTD) pair to the surrogate, grow the Pareto front
                self.surrogate.add_observation(x, out['F'][ttd_idx])
                new_opt = Population.merge(ind, self.algorithm.opt)
                if isinstance(new_opt, Individual):
                    new_opt = Population(new_opt)
                self.algorithm.opt = new_opt

        # retrain the surrogate on the shared n_computed cadence, refresh the cumulative front
        self.surrogate.maybe_retrain(problem.sims['traffic'].n_computed)
        new_pareto = update_true_pareto_front(self.algorithm.opt)
        self.algorithm.opt = new_pareto


def update_true_pareto_front(p_old, p_new=None):
    """
    Return the updated Pareto front from the union of p_old and optionally p_new.

    Parameters:
        p_old (Population): Existing population.
        p_new (Population, optional): New population to merge. If None, only p_old is used.

    Returns:
        Population: The updated Pareto front.
    """
    # Handle empty input
    if p_old is None:
        combined = p_new if p_new is not None else None
        return combined

    if p_new is None:
        combined = p_old
    else:
        combined = Population.merge(p_old, p_new)

    # Convert to list to safely index later
    combined_list = list(combined)

    # Filter for unique decision vectors
    X = np.array([ind.X for ind in combined_list])
    _, unique_indices = np.unique(X, axis=0, return_index=True)
    unique_individuals = [combined_list[i] for i in sorted(unique_indices)]

    # Extract objective values
    F = np.array([
        ind.F if ind.F is not None and len(ind.F) > 0 else np.array([np.inf] * 2)
        for ind in unique_individuals
    ])

    # Find non-dominated solutions
    nd_indices = find_non_dominated(F)
    pareto_front = Population([unique_individuals[i] for i in nd_indices])

    return pareto_front

