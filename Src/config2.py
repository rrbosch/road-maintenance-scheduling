import os
from datetime import datetime
from os import path

import numpy as np

import Src.Utils.Utils as Utils
from Src.Algorithms import registry

"""
Build a run from a settings dict: resolve the operator/evaluator/algorithm choices (stored as
strings) into classes via Src.Algorithms.registry, wire up the NSGA-II stack, and return
(env, algo). Operator choices are strings so they can live in experiment grids / JSON / config.txt.
"""


class Config:
    def __init__(self, args):
        """
        Initialize the Config object for a specific experiment.

        Args:
        - experiment_name (str): Name of the experiment.
        - args (dict): Non-default experimental parameters used to create the directory structure.
        """
        # Set default values
        self.experiment_name = None
        self.problem = 'Problem_py'
        self.case_study = 'Sioux Falls Expanded'
        self.sims = {'traffic'}
        self.algo_name = 'NSGA2'
        self.algo_seed = 1
        self.pop_size = 100
        self.sampling = 'WeightedSlackSampling' # 'IntegerRandomSampling'
        self.repair = 'TestRepair'
        self.selection = 'elitist'
        self.crossover = 'CompositeCrossover' # 'SBX'
        self.mutation = 'CompositeMutation' # 'PolynomialMutation'
        self.termination = 'IterationTermination'
        self.termination_arg = 2000
        self.evaluator = 'ApproximateEvaluator'
        self.lower_bound = 'XGBoost'
        self.lower_bound_quantile = 0.2
        self.objectives = {'SL', 'TTD'}
        self.callback = 'OperatorSuccessCallback'
        self.start_time = datetime.now()
        # Max entries kept in the in-memory traffic-simulation cache (FIFO eviction beyond this).
        # None disables eviction (unbounded, legacy behavior).
        self.traffic_cache_size = 200_000

        # set args
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Invalid arg given as input for the config. key: {key}, value: {value}")
        del args['experiment_name']
        self.experiment_params = args  # Store parameters for logging/tracking purposes

        # Build the results directory path
        self.results_dir = self._build_results_dir()

        # Create the directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)

        print(self)

    def _build_results_dir(self) -> str:
        """Builds the nested directory path for the experiment (anchored to the repo, not cwd)."""
        base_dir = str(Utils.EXPERIMENTS_DIR)
        path_parts = [base_dir, self.experiment_name]

        # Create subdirectories based on the key-value pairs in kwargs
        for key, value in self.experiment_params.items():
            path_parts.append(f"{key}_{value}")

        # Join all parts into a valid path
        results_dir = os.path.join(*path_parts)
        return results_dir

    def __repr__(self):
        """Representation of the config to show key details."""
        return f"Config(experiment_name='{self.experiment_name}', params={self.experiment_params}, results_dir='{self.results_dir}')"

    def initialize(self):
        env = self.set_problem()
        algo = self.set_algo()
        np.random.seed(self.algo_seed)
        return env, algo

    def set_problem(self):
        if self.problem == "Problem_py":
            from Environments.env.Problem import Problem_py
            problem = Problem_py(self.case_study, self.sims, self.objectives, self.lower_bound, self.lower_bound_quantile, self.traffic_cache_size)
        else:
            raise KeyError
        return problem

    def set_algo(self):
        # Resolve each string choice to a class via the registry, then instantiate with the
        # arguments that category expects (samplers take seed, termination takes its arg, a
        # heuristic algorithm takes the config, the rest take no args).
        evaluator = registry.get(self.evaluator)()
        if "NSGA" in self.algo_name:
            sampling = registry.get(self.sampling)(seed=self.algo_seed)
            crossover = registry.get(self.crossover)()
            mutation = registry.get(self.mutation)()
            repair = registry.get(self.repair)()
            termination = registry.get(self.termination)(self.termination_arg)
            callback = registry.get(self.callback)()

            from Src.Algorithms.NSGA2 import NSGA2
            algo = NSGA2(config=self, termination=termination, pop_size=self.pop_size, sampling=sampling, crossover=crossover, mutation=mutation, repair=repair,
                         seed=self.algo_seed, evaluator=evaluator, callback=callback)
            return algo
        elif 'Heuristic' in self.algo_name:
            algo = registry.get(self.algo_name)(self)
            return algo
        else:
            raise KeyError("Called for an invalid/unimplemented algorithm.")

    def save_to_file(self, file_path):
        with open(file_path, 'w') as file:
            for key, value in self.__dict__.items():
                if isinstance(value, list):
                    value = ", ".join(map(str, value))
                elif isinstance(value, type):
                    value = value.__name__  # Get the class name as a string
                file.write(f"{key}: {value}\n")


if __name__ == '__main__':
    pass
