"""Problem-specific crossover operators for the schedule decision vector.

Several elementary crossovers (`NoCrossover`, `NOptCrossover`, `CustomCrossover`/score-weighted,
`UniformCrossover`) are bundled in `CompositeCrossover`, which picks one per mating in proportion
to each operator's recent success rate (adaptive operator selection). "Success" = produced an
offspring that dominates its parents (`strict_pareto_optimal`), credited by `notify_successes`
which the `OperatorSuccessCallback` calls each generation.
"""
import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.variable import get

from Src.Utils.Utils import OffspringOrigin, strict_pareto_optimal


class NoCrossover(Crossover):
    """Pass-through: copy both parents unchanged (offspring then rely on mutation only)."""
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):
        return X

class UniformCrossover(Crossover):
    """Per-gene uniform crossover: each project's start time is swapped between parents w.p. 0.5."""
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        super().__init__(2, 2)

    def _do(self, _, X, **kwargs):
        Q = np.empty_like(X)
        n_matings, n_parents, n_var = X.shape
        for i in range(n_parents):
            mask = self.rng.random(n_var) < 0.5
            q_1 = X[1, i, :]
            q_2 = X[0, i, :]
            q_1[mask] = X[0, i, mask]
            q_2[mask] = X[1, i, mask]
            Q[0, i, :], Q[1, i, :] = q_1, q_2
        return Q

class CustomCrossover(Crossover): # more apt name is ScoreWeightedCrossover
    """Score-weighted crossover: bias gene inheritance toward the parent with the better project.

    For each parent it scores every project by its (normalized) contribution to traffic delay +
    risk (from the cheap lower bounds), then inherits each project's start time from whichever
    parent has the higher relative score for that project -- so genes flow from the parent that
    handles a given project better.
    """
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        super().__init__(n_parents=2, n_offsprings=2)

    def _do(self, problem, X, **kwargs):
        n_matings, n_parents, n_vars = X.shape
        Q = np.empty_like(X)
        for p in range(n_parents):
            project_score = np.zeros(shape=(n_matings, n_vars))
            for m in range(n_matings):
                x = X[m, p, :]
                x_dict = problem.get_x_dict(x)
                traffic_per_timeperiod, _ = problem.objectives['TTD'].get_lower_bound(problem, x_dict, decomposed=True)
                traffic_per_timeperiod = np.array(traffic_per_timeperiod)
                traffic_per_timeperiod = traffic_per_timeperiod - traffic_per_timeperiod.min()
                traffic_per_project = np.zeros(X.shape[2])
                for t in range(len(traffic_per_timeperiod)):
                    for i in x_dict['ongoing_projects'][t]:
                        traffic_per_project[i] += traffic_per_timeperiod[t]
                traffic_per_project = traffic_per_project / problem.input['projects']['duration'].values
                risk_per_project, _ = problem.objectives['SL'].get_lower_bound(problem, x_dict, decomposed=True)
                risk_per_project = np.array(risk_per_project)
                traffic_per_project = traffic_per_project / traffic_per_project.sum()
                risk_per_project = risk_per_project / risk_per_project.sum()
                project_score[m, :] = traffic_per_project + risk_per_project

            # create random weighted crossover
            weights = project_score[0] / (project_score[0]+project_score[1])
            for m in range(n_matings):
                mask = self.rng.random(weights.shape) < weights
                new_q = X[1, p, :]
                new_q[mask] = X[0, p, mask]
                Q[m, p, :] = new_q
        return Q


class NOptCrossover(Crossover):
    """Multi-point (n-opt) crossover over the time axis: swap contiguous time-period blocks.

    Picks a random (geometric) number of cut points along the planning horizon and alternately
    swaps the projects falling in each resulting time interval between the two parents.
    """
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):
        n_pop, n_var = X.shape[1], X.shape[2]
        Q = np.zeros(shape=(2, X.shape[1], X.shape[2]), dtype=int) - 1

        for i in range(X.shape[1]):
            n_opt = self.rng.geometric(p=0.5) + 1
            idx = np.sort(self.rng.choice(problem.input['general']['time periods'], size=n_opt, replace=False))
            if n_opt % 2 == 1:
                idx = np.concatenate(([0], idx))

            mask = np.zeros(n_var, dtype=bool)
            for j in range(int(len(idx)/2)):
                mask[idx[2*j]:idx[2*j+1]] = True
            x_1, x_2 = X[0, i, :], X[1, i, :]
            q_1, q_2 = x_2.copy(), x_1.copy()
            q_1[mask] = x_1[mask]
            q_2[mask] = x_2[mask]
            Q[0, i, :], Q[1, i, :] = q_1, q_2
        return Q

class CompositeCrossover(Crossover):
    """Adaptive ensemble of crossovers: per mating, pick one weighted by its recent success rate.

    ``usage_counts``/``success_counts`` are exponentially-decayed tallies (decay factor ``tau``)
    that form a moving-average success rate; ``notify_successes`` updates them each generation.
    Counts start equal (1000) so the first generations sample operators roughly uniformly.
    """
    def __init__(self, seed=None, **kwargs):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.crossovers = [
            NoCrossover(seed=seed),
            NOptCrossover(seed=seed),
            CustomCrossover(seed=seed),       # score-weighted (see ScoreWeightedCrossover note)
            UniformCrossover(seed=seed),
        ]

        # equal priors so early matings sample operators ~uniformly; decayed by tau over time
        self.usage_counts = [1000] * len(self.crossovers)
        self.success_counts = [1000] * len(self.crossovers)
        self._offspring_operator_ids = []
        self.tau = 0.9  # decay factor of the success-rate moving average (older results fade)
        super().__init__(n_parents=2, n_offsprings=2, **kwargs)

    def _do(self, problem, X, **kwargs):
        n_offsprings, n_matings, n_var = X.shape
        Q = np.empty_like(X)
        self._offspring_operator_ids = []
        success_rate = np.array(self.success_counts) / np.array(self.usage_counts)
        success_rate = success_rate / success_rate.sum()

        for j in range(n_matings):
            idx = self.rng.choice(np.arange(len(success_rate)), p=success_rate)
            crossover = self.crossovers[idx]
            self.usage_counts[idx] += 1

            mating_pair = X[:, j:j + 1, :]
            offspring = crossover._do(problem, mating_pair, **kwargs)  # shape: (n_offsprings, 1, n_var)
            Q[:, j:j + 1, :] = offspring

            # Store the index of the operator used for each offspring
            self._offspring_operator_ids.append(idx)

        return Q

    def do(self, problem, pop, parents=None, **kwargs):
        # NOTE: this is a near-verbatim copy of pymoo's base Crossover.do(), re-implemented for one
        # reason: to attach an OffspringOrigin (parents' F + which crossover was used) to every
        # offspring so notify_successes can later credit the operator. If pymoo's do() changes
        # upstream, re-sync this. The only project-specific lines are the OffspringOrigin block.
        if parents is not None:
            pop = [pop[mating] for mating in parents]

        pop_Fs = [[ind.get("F") for ind in mating] for mating in pop]
        pop_Xs = [[ind.get("X") for ind in mating] for mating in pop]

        n_parents, n_offsprings = self.n_parents, self.n_offsprings
        n_matings, n_var = len(pop), problem.n_var

        X = np.swapaxes(np.array(pop_Xs), 0, 1)
        F = np.swapaxes(pop_Fs, 0, 1)
        if self.vtype is not None:
            X = X.astype(self.vtype)

        Xp = np.empty(shape=(n_offsprings, n_matings, n_var), dtype=X.dtype)
        prob = get(self.prob, size=n_matings)
        cross = np.random.random(n_matings) < prob

        if np.any(cross):
            Q = self._do(problem, X[:, cross], **kwargs)
            assert Q.shape == (n_offsprings, np.sum(cross), problem.n_var)
            Xp[:, cross] = Q

        for k in np.flatnonzero(~cross):
            if n_offsprings < n_parents:
                s = np.random.choice(np.arange(n_parents), size=n_offsprings, replace=False)
            elif n_offsprings == n_parents:
                s = np.arange(n_parents)
            else:
                s = []
                while len(s) < n_offsprings:
                    s.extend(np.random.permutation(n_parents))
                s = s[:n_offsprings]
            Xp[:, k] = np.copy(X[s, k])

        off = []
        method = np.zeros(len(cross), dtype=int) - 1
        method[cross] = np.array(self._offspring_operator_ids)
        for n in range(n_matings):
            for o in range(n_offsprings):
                new_off = Individual()
                new_off.set("X", Xp[o, n, :])
                new_off.data['origin'] = OffspringOrigin(parent_F=F[:, n, :], crossover=method[n])
                off.append(new_off)

        self._offspring_operator_ids = []
        off = Population(off)
        return off

    def notify_successes(self, offspring):
        # Exponentially decay the running usage/success tallies (a moving average via tau), then
        # credit the crossover that produced each offspring that improved on its parents.
        self.usage_counts = [i * self.tau + 1 for i in self.usage_counts]
        self.success_counts = [i * self.tau + 1 for i in self.success_counts]
        for ind in offspring:
            # skip infeasible/dominated offspring (their objectives were set to inf)
            if ind.F is None or not np.all(np.isfinite(ind.F)):
                continue
            if strict_pareto_optimal(ind.data['origin'].parent_F, ind.F):
                crossover_idx = ind.data['origin'].crossover
                if crossover_idx >= 0:  # -1 marks offspring copied through without crossover
                    self.success_counts[crossover_idx] += 1
        self.print_stats()


    def print_stats(self):
        for i, crossover in enumerate(self.crossovers):
            usage = self.usage_counts[i]
            success = self.success_counts[i]
            rate = success / usage if usage > 0 else 0
            print(f"{type(crossover).__name__}: moving average success rate = {rate*100:.1f}%")


