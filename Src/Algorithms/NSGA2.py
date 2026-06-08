import random
from datetime import datetime
from time import time

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2 as NSGA2lib
from pymoo.util.misc import has_feasible

from Src.Utils import results_io
from Src.Algorithms.Evaluators import update_true_pareto_front

"""NSGA-II driver.

Subclasses pymoo's NSGA2 but replaces its run with a manual **ask / evaluator.eval / tell** loop
(`get_res`) so we can plug in the lower-bound evaluators, keep a cumulative true Pareto front, and
write per-generation results. pymoo's own termination is disabled in favour of a wall-clock budget.
"""


class NSGA2(NSGA2lib):
    def __init__(self, config, termination, **kwargs):
        self.config = config
        self.time = time()
        super().__init__(**kwargs, termination=termination)
        self.termination = termination
        self.rng = None
        self.evaluator.algorithm = self
        self.multiprocessing = False
        self.ready = False
        self.log = []
        self.elapsed_time = 0

    def get_res(self, env):
        """Run the manual NSGA-II loop until the wall-clock budget, writing results each generation."""
        # seed every RNG so a run is reproducible / resumable
        np.random.seed(self.config.algo_seed)
        self.rng = np.random.default_rng(self.config.algo_seed)
        random.seed(self.config.algo_seed)
        last_time = time()
        if not self.ready:
            # prepare the algorithm to solve the specific problem
            self.setup(problem=env, seed=self.seed)

        # Run for up to 24h of accumulated compute (survives resume via self.elapsed_time).
        TIME_BUDGET = 24*3600
        while self.elapsed_time < TIME_BUDGET:
            # neutralize pymoo's own termination so only the time budget stops us
            self.termination.perc = 0  # This is not temp
            self.termination.n_max_gen = 9999

            # one generation: ask for offspring, evaluate (possibly via LB screening), tell back
            pop = self.ask()
            self.evaluator.eval(env, pop)
            self.tell(infills=pop)

            # current feasible Pareto front
            pareto_set = [i for i in self.opt if i.feas]
            pareto_size = len(pareto_set)
            gen = self.n_gen - 1

            # update time tracking
            new_time = time()
            iteration_time = new_time - last_time
            last_time = new_time
            self.elapsed_time += iteration_time

            # per-generation log row: timing + pareto size + pruning diagnostics (item 11).
            # Counters default to 0 for evaluators that don't track them (e.g. StandardEvaluator).
            ev = self.evaluator
            self.log.append({
                'iteration': gen,
                'pareto_set_size': pareto_size,
                'time': iteration_time,
                'time_cum': self.elapsed_time,
                'n_computed': env.sims['traffic'].n_computed,   # = 'unique sims'
                'exact_evals': getattr(ev, 'n_exact_evals', 0),
                'lb_pruned': getattr(ev, 'n_lb_pruned', 0),
                'scenarios_materialized': getattr(ev, 'n_scenarios_materialized', 0),
                'n_estimated': getattr(ev, 'n_estimated', 0),
            })
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {self.config.results_dir}')
            print(self.log[-1])

            # current cumulative Pareto front (objective values F + start-time vectors X)
            res = self.result()
            if isinstance(res, list):
                F = np.array([ind.F for ind in res]); X = np.array([ind.X for ind in res])
            elif res.opt is not None:
                F = res.opt.get('F'); X = res.opt.get('X')
            else:
                F, X = None, None

            # drain new surrogate-accuracy rows (item 11 / E2 learning curve)
            ttd = env.objectives.get('TTD') if hasattr(env, 'objectives') else None
            surrogate_rows = []
            if ttd is not None and getattr(ttd, 'surrogate_log', None):
                written = getattr(self, '_n_surrogate_written', 0)
                surrogate_rows = ttd.surrogate_log[written:]
                self._n_surrogate_written = len(ttd.surrogate_log)

            if F is not None and len(F) > 0:
                results_io.write_generation(
                    self.config.results_dir, self.config, self, gen, self.log, F, X,
                    objective_names=list(env.objectives.keys()), surrogate_rows=surrogate_rows,
                )

            # reset per-generation evaluator state after writing
            if hasattr(ev, 'clear_dominated_solutions'):
                ev.clear_dominated_solutions()
            if hasattr(ev, 'reset_diagnostics'):
                ev.reset_diagnostics()

        # obtain the result objective from the algorithm
        res = self.result()
        return res, self.log

    def resume(self):
        env = self.problem
        print(f"Restarting the algortihm after iteration {self.n_gen - 1}")
        self.get_res(env)

    def _set_optimum(self, **kwargs):
        # Override pymoo's optimum tracking to accumulate a *cumulative* non-dominated set across
        # generations (update_true_pareto_front), rather than just this generation's rank-0 front.
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            nd_pop = self.pop[self.pop.get("rank") == 0]
            new_opt = update_true_pareto_front(self.opt, nd_pop)
            self.opt = new_opt
