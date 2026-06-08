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
        # Per-generation pruning diagnostics (read + reset each generation by NSGA2.get_res).
        self.reset_diagnostics()

    def reset_diagnostics(self):
        """Zero the per-generation pruning counters (telemetry for progress.csv)."""
        self.n_exact_evals = 0          # individuals given a full exact problem.evaluate
        self.n_lb_pruned = 0            # individuals discarded on their lower bound alone
        self.n_scenarios_materialized = 0  # estimated scenarios turned exact via add_scenario
        self.n_estimated = 0            # estimated scenario-contributions seen across LB computations

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

