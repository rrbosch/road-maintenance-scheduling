"""Initial-population samplers.

``WeightedSlackSampling`` (the configured default) seeds the population with constructive
heuristic schedules (``WeightedSlackHeuristic``) rather than random ones, giving NSGA-II a
feasible, decent starting front. ``IntegerRandomSampling`` and ``FeasibleRandomSampling`` are
plainer alternatives (uniform / constraint-aware random schedules).
"""
import numpy as np
from pymoo.operators.sampling.rnd import IntegerRandomSampling as IRS

from Src.Algorithms.Heuristics import WeightedSlackHeuristic


class IntegerRandomSampling(IRS):
    def __init__(self, seed=0):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        X = self.rng.integers(problem.xl, problem.xu+1, size=(n_samples, problem.n_var))
        return X


def find_starting_period(resource_use, release_date, due_date, execution_time):
    # go through all viable starting times.
    # Determine the max resource use (nr. of simultaneous projects) throughout project execution time if assigned to that starting time,
    # and assign to the earliest argmin(max_resource_use). A max_resource_use of 0 allows early termination.
    viable_starting_times = [t for t in range(release_date, due_date)]
    mru_list = []
    for t in viable_starting_times:
        max_resource_use = np.max([resource_use[t:t+execution_time]])
        if max_resource_use == 0:
            return t
        else:
            mru_list.append(max_resource_use)
    min_mru_idx = np.argmin(mru_list)
    return viable_starting_times[min_mru_idx]


class WeightedSlackSampling(IRS):
    def __init__(self, seed=0):
        super().__init__()
        self.seed = seed
        self.order = 'fixed'

    def _do(self, problem, n_samples, **kwargs):
        """
        Generate `n_samples` solutions using increasing slack and dynamic multipliers.
        Compatible with pymoo-style sampling interface.
        """
        heuristic = WeightedSlackHeuristic(None, order=self.order, seed=self.seed)
        solutions = heuristic.build_res(problem, n_samples)
        X = np.array([x for x, _, _ in solutions.values()])
        return X


class FeasibleRandomSampling(IRS):
    def __init__(self, seed=0):
        super().__init__()
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def _do(self, problem, n_samples, examples=None, **kwargs):
        """
        Generate n_samples solutions trying to satisfy budget and capacity constraints.
        """
        samples = []

        for _ in range(n_samples):
            if examples is not None and len(examples) > 0:
                # Pick a solution from examples
                idx = self.rng.choice(len(examples))
                e = examples[idx].copy()  # Copy to avoid modifying the original

                # Make the solution lose a random proportion of its values
                proportion = self.rng.random()
                mask = self.rng.random(len(e)) < proportion
                e[mask] = -1

                X = self._generate_single_sample(problem, e)
            else:
                X = self._generate_single_sample(problem)
            samples.append(X)

        return np.array(samples)


    def _generate_single_sample(self, problem, partial_x=None):
        """
        Generate a single solution by randomly planning projects while attempting
        to satisfy capacity and budget constraints.
        """
        n_projects = problem.n_var
        t_max = problem.input['general']['time periods']
        capacity = problem.input['general']['construction teams']
        budget_per_t = problem.input['general']['budget']

        # Initialize solution with -1 (not planned)
        if partial_x is not None:
            X = partial_x.copy()
        else:
            X = np.full(n_projects, -1, dtype=int)

        # Track available capacity and budget
        available_capacity = np.full(t_max, capacity, dtype=int)
        available_budget = np.array([budget_per_t * (i+1) for i in range(t_max)])

        # Get project ordering (random shuffle)
        project_order = self.rng.permutation(n_projects)

        for project_idx in project_order:
            duration = int(problem.input['projects'].at[project_idx, 'duration'])
            cost = problem.input['projects'].at[project_idx, 'cost']
            xl = int(problem.xl[project_idx])
            xu = int(problem.xu[project_idx])

            # Get all possible starting times in [xl, xu]
            possible_times = list(range(xl, xu + 1))
            self.rng.shuffle(possible_times)

            # Try to find a feasible starting time
            planned = False

            for project_time in possible_times:
                # Check if this time is feasible
                if self._is_feasible(project_time, duration, cost, available_capacity, available_budget, t_max):
                    # Plan the project
                    X[project_idx] = project_time
                    available_capacity[project_time:(project_time + duration)] -= 1
                    available_budget[project_time:] -= cost
                    planned = True
                    break

            # If all times are infeasible, just pick a random one anyway
            if not planned:
                project_time = self.rng.integers(xl, xu + 1)
                X[project_idx] = project_time
                # Still update the tracking arrays
                end_time = min(project_time + duration, t_max)
                available_capacity[project_time:end_time] -= 1
                available_budget[project_time:] -= cost
                # then assume everything is fine for now so we can continue with planning
                available_budget[available_budget < 0] = 0
                available_capacity[available_capacity < 0] = 0
        return X

    def _is_feasible(self, start_time, duration, cost, available_capacity, available_budget, t_max):
        """
        Check if planning a project at start_time is feasible given capacity and budget constraints.
        """
        # Check if project would extend beyond time horizon
        if start_time + duration > t_max:
            return False

        # Check capacity constraint: all time periods during project execution must have capacity
        if np.any(available_capacity[start_time:(start_time + duration)] < 1):
            return False

        # Check budget constraint: must have enough budget at start time
        if available_budget[start_time] < cost:
            return False

        return True