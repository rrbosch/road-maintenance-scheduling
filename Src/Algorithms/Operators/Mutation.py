"""Problem-specific mutation operators for the schedule decision vector.

Mirrors `Crossover.py`: elementary mutators (`N_Opt` shuffle, `GeometricMutation` random step,
and the objective-aware `TrafficBasedMutation`/`RiskBasedMutation`/`MOBasedMutation` that bias
which project to nudge by its traffic and/or tardiness contribution) are bundled in
`CompositeMutation`, which applies a random number of them per individual, each chosen in
proportion to its recent success rate. The objective-aware mutators normalize a per-project score
into a probability and fall back to uniform when the score sums to zero (degenerate plans).
"""
from copy import deepcopy

import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.core.variable import get

from Src.Utils.Utils import strict_pareto_optimal


class CompositeMutation(Mutation):
    """Adaptive ensemble of mutators (the mutation counterpart of `CompositeCrossover`).

    Each individual gets a geometric number of mutations; each is drawn weighted by the operators'
    recent success rates (`usage_counts`/`success_counts`, decayed by `tau`). The ids of the
    mutators applied to an individual are recorded on its `OffspringOrigin` so `notify_successes`
    can credit them.
    """
    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.mutations = [
            N_Opt(seed),
            GeometricMutation(seed),
            TrafficBasedMutation(seed),
            RiskBasedMutation(seed),
            MOBasedMutation(seed),
        ]
        self.tau = 0.9  # decay factor of the success-rate moving average
        self.usage_counts = [1000] * len(self.mutations)
        self.success_counts = [1000] * len(self.mutations)
        self._used_mutations = []
        super().__init__()

    def do(self, problem, pop, inplace=True, **kwargs):
        # NOTE: near-verbatim copy of pymoo's base Mutation.do(), re-implemented so we can record
        # which mutators were applied (self._used_mutations -> each offspring's OffspringOrigin)
        # for notify_successes. Re-sync if pymoo's do() changes upstream.
        if not inplace:
            pop = deepcopy(pop)

        n_mut = len(pop)
        X = pop.get("X")

        # per-individual mutation probability, then the mask of which individuals to mutate
        prob = get(self.prob, size=n_mut)
        mut = np.random.random(size=n_mut) <= prob

        Xp = self._do(problem, X[mut], **kwargs)
        pop[mut].set("X", Xp)

        # record, per mutated individual, the list of mutator ids that were applied
        mutated_inds = pop[mut]
        for ind, mut_log in zip(mutated_inds, self._used_mutations):
            # the OffspringOrigin was created by the crossover that produced this offspring
            ind.data["origin"].mutations = mut_log
        self._used_mutations = []
        return pop

    def _do(self, problem, X, **kwargs):
        """Apply a geometric number of success-weighted mutators to each individual.

        ``n_mutations`` per individual is ``Geometric(0.5) - 1`` (so often 0, occasionally
        several); each mutator is chosen with probability proportional to its success rate.
        Returns the mutated X and records the applied mutator ids in ``self._used_mutations``.
        """
        X_mut = X.copy()
        n_individuals = X.shape[0]
        self._used_mutations = []
        n_mutations = self.rng.geometric(p=0.5, size=n_individuals) - 1
        # choose mutators in proportion to their recent success rate
        success_rate = np.array(self.success_counts) / np.array(self.usage_counts)
        success_rate = success_rate / success_rate.sum()

        for i in range(n_individuals):
            x_mut_log = []
            for _ in range(n_mutations[i]):
                idx = self.rng.choice(np.arange(len(success_rate)), p=success_rate)
                mutator = self.mutations[idx]
                self.usage_counts[idx] += 1
                X_mut[i] = mutator._do(problem, X_mut[i][None, :], **kwargs)[0]
                x_mut_log.append(idx)
            self._used_mutations.append(x_mut_log)
        return X_mut

    def notify_successes(self, offspring):
        # Exponentially decay the running usage/success tallies (a moving average via tau), then
        # credit every mutation operator applied to an offspring that improved on its parents.
        self.usage_counts = [i * self.tau + 1 for i in self.usage_counts]
        self.success_counts = [i * self.tau + 1 for i in self.success_counts]
        for ind in offspring:
            # skip infeasible/dominated offspring (their objectives were set to inf)
            if ind.F is None or not np.all(np.isfinite(ind.F)):
                continue
            if strict_pareto_optimal(ind.data['origin'].parent_F, ind.F):
                # offspring that were not mutated keep the default empty mutations list
                for idx in ind.data['origin'].mutations:
                    self.success_counts[idx] += 1
        self.print_stats()

    def print_stats(self):
        for i, mut in enumerate(self.mutations):
            usage = self.usage_counts[i]
            success = self.success_counts[i]
            rate = success / usage if usage > 0 else 0.0
            print(f"{type(mut).__name__}: moving average success rate = {rate*100:.1f}%")


class N_Opt(Mutation):
    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, X, **kwargs):
        X = X.copy()
        n_individuals, n_variables = X.shape

        # Determine swap sizes for each individual
        swap_sizes = self.rng.geometric(p=0.5, size=n_individuals) + 1

        for i in range(n_individuals):
            size = min(swap_sizes[i], n_variables)  # avoid exceeding number of columns
            cols = self.rng.choice(n_variables, size=size, replace=False)
            shuffled = X[i, cols].copy()
            self.rng.shuffle(shuffled)
            X[i, cols] = shuffled
        return X


class GeometricMutation(Mutation):
    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, X, **kwargs):
        X = X.copy()
        xl, xu = problem.xl, problem.xu
        n_individuals, n_vars = X.shape

        for i in range(n_individuals):
            # Select a random index to mutate
            idx = self.rng.integers(0, n_vars)

            # Sample step from geometric distribution (1, 2, 3, ...)
            step = self.rng.geometric(p=0.5)

            # Randomly decide direction: +1 or -1
            direction = self.rng.choice([-1, 1])
            delta = direction * step

            # Apply mutation and clip to bounds
            X[i, idx] = np.clip(X[i, idx] + delta, xl[idx], xu[idx])
        return X


class TrafficBasedMutation(Mutation):
    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, X, **kwargs):
        X = X.copy()
        n_individuals, n_vars = X.shape

        for i in range(n_individuals):
            # Select a random index to mutate
            x_dict = problem.get_x_dict(X[i])
            # Degenerate plan with no ongoing projects in any period: nothing to
            # mutate on a traffic basis, and the time-period loop below would spin
            # forever looking for a non-empty project set. Skip this individual.
            if sum(len(s) for s in x_dict['ongoing_projects']) == 0:
                continue
            traffic_per_timeperiod, _ = problem.objectives['TTD'].get_lower_bound(problem, x_dict, decomposed=True)
            traffic_per_timeperiod = np.array(traffic_per_timeperiod)
            traffic_ps = traffic_per_timeperiod - traffic_per_timeperiod.min()
            # Guard against a zero / non-finite sum (e.g. flat traffic deltas):
            # fall back to a uniform distribution so rng.choice never sees NaN.
            total = traffic_ps.sum()
            if total > 0 and np.isfinite(total):
                traffic_ps = traffic_ps / total
            else:
                traffic_ps = np.full(traffic_ps.size, 1 / traffic_ps.size)
            done = False
            while not done:
                time_idx = self.rng.choice(np.arange(traffic_ps.shape[0]), p=traffic_ps)
                ongoing_projects = list(x_dict['ongoing_projects'][time_idx])
                if len(ongoing_projects) > 0:
                    done = True
            idx = self.rng.choice(list(x_dict['ongoing_projects'][time_idx]))

            # Randomly decide direction based on upper/lower limit and magnitude based on movement require to make project not happen on time t
            fraction = (X[i, idx] - problem.xl[idx]) / (problem.xu[idx] - problem.xl[idx])
            if self.rng.random() < fraction:
                difference = X[i, idx] - max(time_idx - problem.input['projects'].loc[idx, 'duration'] - 1, 0)
                mutation = -self.rng.geometric(p=1/difference)
            else:
                difference = time_idx + 1
                mutation = self.rng.geometric(p=1/difference)
            new_x = X[i, idx] + mutation
            # Apply mutation and clip to bounds
            new_x = min(max(problem.xl[idx], new_x), problem.xu[idx])
            X[i, idx] = new_x
        return X

class RiskBasedMutation(Mutation):
    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, X, **kwargs):
        X = X.copy()
        n_individuals, n_vars = X.shape

        for i in range(n_individuals):
            # Select a random index to mutate
            x_dict = problem.get_x_dict(X[i])
            risk_per_project, _ = problem.objectives['SL'].get_lower_bound(problem, x_dict, decomposed=True)
            risk_per_project = np.array(risk_per_project)
            ps = risk_per_project - risk_per_project.min()
            if ps.sum() > 0:
                ps = ps / ps.sum()
            else:
                ps = np.full(ps.size, 1/ps.size)
            idx = self.rng.choice(np.arange(n_vars), p=ps)

            # Apply mutation and clip to bounds
            X[i, idx] = max(0, X[i, idx] - 1)
        return X


class MOBasedMutation(Mutation):
    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, X, **kwargs):
        X = X.copy()
        n_individuals, n_vars = X.shape

        for i in range(n_individuals):
            # determine the contribution of each project to traffic
            x_dict = problem.get_x_dict(X[i])
            traffic_per_timeperiod, _ = problem.objectives['TTD'].get_lower_bound(problem, x_dict, decomposed=True)
            traffic_per_timeperiod = np.array(traffic_per_timeperiod)
            traffic_per_timeperiod = traffic_per_timeperiod - traffic_per_timeperiod.min()
            traffic_per_project = np.zeros(n_vars)
            for t in range(len(traffic_per_timeperiod)):
                for p in x_dict['ongoing_projects'][t]:
                    traffic_per_project[p] += traffic_per_timeperiod[t]
            # Guard against a zero / non-finite sum (e.g. no ongoing projects, or
            # flat traffic deltas): fall back to a uniform distribution.
            traffic_total = traffic_per_project.sum()
            if traffic_total > 0 and np.isfinite(traffic_total):
                traffic_per_project = traffic_per_project / traffic_total
            else:
                traffic_per_project = np.full(n_vars, 1 / n_vars)
            # determine the contribution of each project to risk
            risk_per_project, _ = problem.objectives['SL'].get_lower_bound(problem, x_dict, decomposed=True)
            risk_per_project = np.array(risk_per_project)
            if risk_per_project.sum() > 0:
                risk_per_project = risk_per_project / risk_per_project.sum()
                project_score = traffic_per_project + risk_per_project
            else:
                project_score = traffic_per_project
            # Guard the combined score normalization the same way.
            score_total = project_score.sum()
            if score_total > 0 and np.isfinite(score_total):
                project_score = project_score / score_total
            else:
                project_score = np.full(n_vars, 1 / n_vars)
            idx = self.rng.choice(np.arange(n_vars), p=project_score)
            # Apply mutation and clip to bounds
            mutation = self.rng.choice([-1, 1]) * self.rng.geometric(p=0.5)
            X[i, idx] = max(0, X[i, idx] + mutation)
        return X