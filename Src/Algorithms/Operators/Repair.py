"""Constraint repair for schedules.

After sampling/crossover/mutation an individual may violate the scheduling constraints (release
date, hard due date, per-period budget, construction-team capacity). ``TestRepair`` nudges one
violated constraint at a time toward feasibility, re-checking after each fix, until feasible or
``max_tries`` is exhausted.
"""
import numpy as np
from pymoo.core.repair import Repair


class TestRepair(Repair):
    """Greedy, randomized repair: repeatedly fix one randomly-chosen violated constraint."""
    def __init__(self, max_tries = 10, seed=0):
        super().__init__()
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.max_tries = max_tries

    def _do(self, problem, X, **kwargs):
        if X.ndim == 1:
            X = self.scheduling_repair(problem, X)
        else:
            inputs = [(problem, X[i, :]) for i in range(X.shape[0])]
            X = [self.scheduling_repair(*i) for i in inputs]
            X = np.array(X)
        return X

    def scheduling_repair(self, problem, x, return_g=False, fixed=set()):
        """Repair one schedule ``x`` in place-ish, returning a (more) feasible vector.

        Each pass fixes one randomly chosen violated constraint. The ``iteration`` counter only
        advances when a pass fails to reduce the violation count, so productive repairs effectively
        get extra tries; ``fixed`` is a set of project ids the repair must not move.
        """
        iteration = 0
        max_tries = self.max_tries
        nr_violations = 1e20
        while iteration <= max_tries:
            # check which constraints are being violated
            x_dict = problem.get_x_dict(x)
            g = problem.check_scheduling_constraints(x_dict)
            violated_constraints = [i for i in range(len(g)) if g[i] > 0]
            if len(violated_constraints) == 0:
                break
            # if you're not making progress, add one to the max_tries
            if len(violated_constraints) >= nr_violations:
                iteration += 1
            else:
                iteration -= 1
            nr_violations = len(violated_constraints)
            # pick one constraint to repair and repair it.
            v = self.rng.choice(violated_constraints)
            x = self.scheduling_constraint_repair(problem, x, v, x_dict, fixed=fixed)
        if return_g:
            return x, g
        return x

    def scheduling_constraint_repair(self, problem, x, v, x_dict, fixed):
        violated_constraint = problem.constraints[v]
        constraint_type = problem.constraints[v]['constraint nr']
        if constraint_type == 1:
            # project has to start after release date
            viable_starting_times = find_viable_starting_times(violated_constraint['project'], problem, x_dict)
            x[violated_constraint['project']] = viable_starting_times[0]
        elif constraint_type == 2:
            # project has to start before absolute due date - project time
            viable_starting_times = find_viable_starting_times(violated_constraint['project'], problem, x_dict)
            x[violated_constraint['project']] = viable_starting_times[-1]
        elif constraint_type == 3:
            # money related constraint
            # take the latest started project on or before this time, and start it directly after this time period
            time_period = violated_constraint['time period']
            while True:
                started_projects = list(x_dict['projects'][x_dict['projects']['start'] == time_period].index.values)
                started_projects = [i for i in started_projects if i not in fixed]
                if len(started_projects) > 0:
                    project_to_reschedule = self.rng.choice(started_projects)
                    x[project_to_reschedule] = time_period + 1
                    break
                else:
                    time_period += -1
                if time_period < 0:
                    break
        elif constraint_type == 4:
            # available construction teams
            # select a random project ongoing at this time period and reschedule it directly before or after this time
            projects_at_time_period = list(x_dict['ongoing_projects'][violated_constraint['time period']])
            projects_at_time_period = [i for i in projects_at_time_period if i not in fixed]
            project_to_reschedule = self.rng.choice(list(projects_at_time_period))
            viable_starting_times = find_viable_starting_times(project_to_reschedule, problem, x_dict)
            movement = [np.abs(i - violated_constraint['time period']) for i in viable_starting_times]
            new_start_period = viable_starting_times[np.argmin(movement)]
            # new_start_period = self.rng.choice(viable_starting_times)
            x[project_to_reschedule] = new_start_period
        else:
            raise Exception('encountered incorrect constraint type.')
        return x


def find_viable_starting_times(project, problem, x_dict):
    ub = problem.input['projects']['hard due date'][project] - problem.input['projects']['duration'][project]
    viable_starting_dates = list(range(ub))
    # filter infeasible starting dates w.r.t available teams
    available_teams = [problem.input['general']['construction teams'] - len(x_dict['ongoing_projects'][t]) for t in range(len(x_dict['ongoing_projects']))]
    duration = problem.input['projects']['duration'][project]
    available_teams_violations = [-1 * min(np.min(np.array(available_teams[t:t + duration])), 1) for t in viable_starting_dates]
    min_teams_violation = np.min(available_teams_violations)
    viable_starting_dates = [i for i in viable_starting_dates if available_teams_violations[i] <= min_teams_violation]

    # filter infeasible starting dates w.r.t. budget
    available_budget2 = [problem.input['general']['budget'] - x_dict['spending'][0]]
    for i in range(1, len(x_dict['spending'])):
        modification = problem.input['general']['budget'] - x_dict['spending'][i]
        available_budget2.append(available_budget2[-1] + modification)
    project_cost = problem.input['projects']['cost'][project]
    budget_violations = [min(0, np.max(project_cost - available_budget2[t:])) for t in viable_starting_dates]
    min_budget_violation = np.min(budget_violations)
    viable_starting_dates = [viable_starting_dates[i] for i in range(len(viable_starting_dates)) if budget_violations[i] <=min_budget_violation]
    return viable_starting_dates
