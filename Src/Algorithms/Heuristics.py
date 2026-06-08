"""Constructive heuristics that build a Pareto set without running the full NSGA-II loop.

Each heuristic sweeps a scalar trade-off knob (a weight or a budget on the risk-vs-delay
trade-off) and greedily constructs one schedule per knob value, yielding a spread of solutions
along the front. ``WeightedSlackHeuristic`` also backs the ``WeightedSlackSampling`` operator
(it seeds the NSGA-II population). Selected via ``Config.algo_name`` and dispatched in
``Config.set_algo``.

Output uses the same ``results_io.py`` schema as NSGA-II (overhaul item 7): a heuristic produces a
single final front, so it is recorded as one "generation" (``config.json`` + ``fronts.csv`` +
``final_solutions.csv``). No ``algo.pkl`` is written — heuristics run in one shot and don't resume.
"""
import os

import numpy as np
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from sortedcontainers import SortedDict

from Src.Algorithms.Operators.Repair import TestRepair
from Src.Utils import results_io


# Heuristic ideas explored: multiple random samples; spreading projects out; a greedy policy;
# doing nothing; allocating a slack budget per project (see IncreasingSlackHeuristic).

class Heuristic:
    """Base class: holds the run ``config`` and the shared (legacy-format) result writer."""
    def __init__(self, config):
        self.config = config

    def write_results(self, X, F, objective_names):
        """Keep the unique non-dominated rows and write them via the results_io schema (item 7).

        The front is recorded as a single generation (0): ``config.json`` + ``fronts.csv`` (objective
        values) + ``final_solutions.csv`` (start-time vectors). ``objective_names`` orders the
        ``fronts.csv`` columns and must match the column order of ``F`` (use
        ``list(env.objectives.keys())``, the same order NSGA-II uses).
        """
        # filter all dominated rows
        keep_idx = find_non_dominated(F)
        X = X[keep_idx]
        F = F[keep_idx]

        # filter all non-unique rows
        _, idx = np.unique(F, axis=0, return_index=True)
        X = X[idx]
        F = F[idx]

        # store the results (single-generation front; remove any stale fronts.csv from a prior run
        # since append_fronts appends and heuristics don't resume)
        result_dir = self.config.results_dir
        fronts_path = os.path.join(result_dir, 'fronts.csv')
        if os.path.exists(fronts_path):
            os.remove(fronts_path)
        results_io.write_config(result_dir, self.config)
        results_io.append_fronts(result_dir, 0, F, objective_names)
        results_io.write_final_solutions(result_dir, X)


class WeightedSlackHeuristic(Heuristic):
    """Greedy scalarized constructor: for a weight ``10**m`` it places projects (slack-ordered) one
    at a time at the start time minimizing ``risk + 10**m * delay``. ``build_res`` sweeps ``m`` by
    bisecting the largest gap in the realized trade-off, spreading ``tries`` solutions along the front.
    """
    def __init__(self, config, seed, order='fixed'):
        super().__init__(config)
        self.order = order
        self.tries = 60
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def get_res(self, env):
        solutions = self.build_res(env)

        # store the results
        X = np.array([x for x, _, _ in solutions.values()])
        F = np.array([out['F'] for _, out, _ in solutions.values()])
        self.write_results(X, F, list(env.objectives.keys()))


    def build_res(self, env, n_samples=None):
        """
        Generate solutions using different multipliers for the weighted objective.
        """
        if n_samples is None:
            n_samples = self.tries
        solutions = SortedDict()

        # first generate the extreme solutions
        def generate_new_solution(mult):

            exp_mult = 10 ** mult
            sol, out = self.construct_schedule(env, exp_mult)
            realized_m = np.log10(calculate_m(out))
            solutions[mult] = (sol, out, realized_m)

        small_m, big_m = -12, 1
        generate_new_solution(small_m)
        generate_new_solution(big_m)

        while len(solutions) < n_samples:
            # find m
            gaps = np.diff([i for _, _, i in solutions.values()])
            gaps[gaps < 0] = 0
            if gaps.sum() > 0:
                biggest_gap = np.random.choice(np.arange(gaps.shape[0]), p=gaps/gaps.sum()) # np.argmax(gaps)
            else:
                biggest_gap = np.random.choice(np.arange(gaps.shape[0]))
            m_inputs = np.array([i for i in solutions])
            m = np.mean([m_inputs[biggest_gap], m_inputs[biggest_gap+1]])
            generate_new_solution(m)
        return solutions


    def construct_schedule(self, env, multiplier, rollout_policy=None):
        if self.order == 'fixed':
            duration = env.input['projects']['duration']
            due_dates = env.input['projects']['hard due date']
            slack = due_dates - duration
            project_order = np.argsort(slack)
        elif self.order == 'random':
            project_order = np.arange(env.input['projects'].shape[0])
            self.rng.shuffle(project_order)
        else:
            raise KeyError
        schedule = np.full(project_order.shape[0], -1)  # -1 means unscheduled

        for pid in project_order:  # go through projects in a fixed order
            best_time = 0
            best_score = float('inf')

            for t in range(env.input['projects'].at[pid, 'hard due date']):  # and test the costs associated with each timeslot
                temp_schedule = schedule.copy()
                temp_schedule[pid] = t

                out = {}
                env._evaluate(temp_schedule, out, partial=True)
                obj1, obj2 = out['F']
                weighted_score = obj1 + multiplier * obj2

                # optionally estimate the remaining costs with a rollout policy
                if rollout_policy is not None:
                    weighted_score = rollout_policy.eval(temp_schedule, multiplier)

                # update the best timeslot if possible
                if weighted_score < best_score:
                    best_score = weighted_score
                    best_time = t

            schedule[pid] = best_time

        # Evaluate final schedule
        schedule = TestRepair(1000).scheduling_repair(env, schedule)
        out = {}
        env._evaluate(schedule, out)
        return schedule, out


class WeightedSlackHeuristicRollout(WeightedSlackHeuristic):
    def __init__(self, config):
        super().__init__(config)
        self.tries = 60


    def get_res(self, env):
        """
        Generate solutions using different multipliers for the weighted objective.
        """
        solutions = []
        out_list = []
        duration = env.input['projects']['duration']
        due_dates = env.input['projects']['hard due date']
        slack = due_dates - duration
        project_order = np.argsort(slack)
        rollout = SlackOrderRollout(env)

        # first generate the extreme solutions
        for m in [1e-6, 1e12]:
            sol, out = self.construct_schedule(env, project_order, m, rollout_policy=rollout)
            solutions.append(sol)
            out_list.append(out)

        # find the extreme realized values for m
        realized_small_m = calculate_m(out_list[0])
        realized_big_m = calculate_m(out_list[1])

        # calculate the range of multipliers
        base = (realized_big_m / realized_small_m) ** (1/(self.tries-3))
        multipliers = realized_small_m * base ** np.arange(self.tries)
        multipliers = multipliers[1:-1]
        for m in multipliers:
            sol, out = self.construct_schedule(env, project_order, m, rollout_policy=rollout)
            solutions.append(sol)
            out_list.append(out)
            print(f'Done with solution {len(out_list)}/{self.tries}')
            print(out_list)

        # store the results
        X = np.array(solutions)
        F = np.array([i['F'] for i in out_list])
        self.write_results(X, F, list(env.objectives.keys()))
        return solutions


class SlackOrderRollout:
    """Rollout policy: estimate a partial schedule's value by greedily completing it (slack order).

    Given a partially-built schedule, it fills the remaining projects in increasing-slack order at
    their best feasible start time, then returns the scalarized objective -- so the constructor can
    look ahead instead of judging a placement on its immediate cost alone.
    """
    def __init__(self, env):
        self.env = env
        self.repair = TestRepair(10)

    def eval(self, partial_schedule, mult):
        due_date = self.env.input['projects']['hard due date'].values
        duration = self.env.input['projects']['duration'].values
        unscheduled_projects = np.where(partial_schedule < 0)[0]
        slack = due_date[unscheduled_projects] - duration[unscheduled_projects]
        unscheduled_projects = unscheduled_projects[np.argsort(slack)]

        x = partial_schedule.copy()
        for pi in unscheduled_projects:
            feasible_starting_times = np.arange(due_date[pi] - duration[pi]) # placeholder
            best_t = 0
            best_score = np.inf
            for t in feasible_starting_times:
                x_dict = self.env.get_x_dict(x)
                if all(self.env.check_scheduling_constraints(x_dict) <= 0):
                    objs = self.env.get_lb(x)
                    obj1, obj2 = objs[0].F_lb
                    weighted_score = mult * obj1 + obj2
                    if weighted_score < best_score:
                        best_t = t
                        best_score = weighted_score
            x[pi] = best_t
        x = self.repair._do(self.env, x)
        objs = self.env.get_lb(x)
        obj1, obj2 = objs[0].F_lb
        weighted_score = mult * obj1 + obj2
        return weighted_score


def calculate_m(out):
    """Realized risk/delay ratio of a solution -- the trade-off weight it effectively corresponds to."""
    return out['F'][0] / max(out['F'][1], 1)


class IncreasingSlackHeuristic(Heuristic):
    """Budget-sweep constructor: allow a growing per-project risk budget and minimize delay within it.

    Instead of a weight, it caps how much risk each project may incur (a budget that accumulates as
    projects are placed) and picks the lowest-delay feasible start under that cap; sweeping the
    budget from 0 to ~infinity traces out the front.
    """
    def __init__(self, config):
        super().__init__(config)
        self.tries = 60

    def get_res(self, env):
        """
        Generate solutions using different multipliers for the weighted objective.
        """
        solutions = []
        out_list = []
        duration = env.input['projects']['duration']
        due_dates = env.input['projects']['hard due date']
        slack = due_dates - duration
        project_order = np.argsort(slack)

        # first generate the extreme solutions
        small_budget = 0
        big_budget = np.inf
        sol_1, out_1, mub_1 = self.construct_schedule(env, project_order, small_budget)
        sol_2, out_2, mub_2 = self.construct_schedule(env, project_order, big_budget)
        solutions.append(sol_1)
        solutions.append(sol_2)
        out_list.append(out_1)
        out_list.append(out_2)

        # find the extreme realized values for m
        realized_small_budget = mub_1 * slack.shape[0]
        realized_big_budget = mub_2 * slack.shape[0]

        # calculate the range of multipliers
        base = (realized_big_budget / realized_small_budget) ** (1/(self.tries-3))
        budgets = realized_small_budget * base ** np.arange(self.tries)
        budgets = budgets[1:-1]
        for b in budgets:
            sol, out, _ = self.construct_schedule(env, project_order, b)
            solutions.append(sol)
            out_list.append(out)

        # store the results

        X = np.array(solutions)
        F = np.array([i['F'] for i in out_list])
        self.write_results(X, F, list(env.objectives.keys()))
        return solutions

    def construct_schedule(self, env, project_order, budget):
        schedule = np.full(project_order.shape[0], -1)  # -1 means unscheduled
        budget_per_project = budget / project_order.shape[0]
        current_budget = budget_per_project
        max_used_budget = 0

        for i, pid in enumerate(project_order):  # go through projects in a fixed order
            best_time = 0
            best_score = float('inf')
            best_obj1 = 0

            for t in range(env.input['projects'].at[pid, 'hard due date']):  # and test the costs associated with each timeslot
                temp_schedule = schedule.copy()
                temp_schedule[pid] = t

                out = {}
                env._evaluate(temp_schedule, out, partial=True)
                obj1, obj2 = out['F']
                feasible = (obj1 < np.inf) and (obj2 < np.inf)
                # if the slack is lower than allowed, accept. Else, reject.
                if feasible:
                    if (obj1 <= current_budget) or (best_score == float('inf')):
                        if obj2 < best_score:
                            best_score = obj2
                            best_time = t
                            best_obj1 = obj1
                    else:
                        break

            schedule[pid] = best_time
            current_budget += budget_per_project
            max_used_budget = max(max_used_budget, best_obj1/i)

        # Evaluate final schedule
        schedule = TestRepair(1000).scheduling_repair(env, schedule)
        out = {}
        env._evaluate(schedule, out)
        return schedule, out, max_used_budget



if __name__ == '__main__':
    pass

