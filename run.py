import itertools
import os
import subprocess
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

from Src.config2 import Config

warnings.filterwarnings("ignore", category=RuntimeWarning)
"""Local experiment runner with hardcoded grids.

Each ``*_experiment`` function below defines a ``params`` dict (every value a list); its cartesian
product is expanded into one ``Config`` per combination and dispatched over a multiprocessing Pool
(or serially with ``python run.py single_thread``). Pick which grid runs by editing ``__main__``.
See docs/configuration.md and docs/running.md.
"""


def train_with_args(args):
    time.sleep(np.random.rand())
    config = Config(args)
    env, algo = config.initialize()
    algo.get_res(env)
    return True


def run_pooled_experiments(params, processes):
    keys = params.keys()
    if not all([isinstance(key, list) for key in params.values()]):
        raise TypeError("All parameters for a pooled experiment need to be lists, even 1-element items.")
    combinations = list(itertools.product(*params.values()))
    result = [dict(zip(keys, combination)) for combination in combinations]

    print(f"Running Experiment {params['experiment_name']}.")
    print(f"Experiment params: {params}")
    if any([i == "single_thread" for i in sys.argv]):
        print("Running in single-threaded mode.")
        _ = [train_with_args(i) for i in result]
    else:
        with Pool(processes=processes) as pool:
            _ = pool.map(train_with_args, result, chunksize=1)
    print('done')


def SF_expanded_standard_vs_ILB():
    params = {
        'experiment_name': ["Standard vs LBPM"],
        'evaluator': ["StandardEvaluator", "LowerBoundEvaluator"],
        'algo_seed': [i for i in range(10)],
    }
    run_pooled_experiments(params, 10)


def XGBoost_experiment():
    params = {
        'experiment_name': ["XGBoost"],
        'evaluator': ["LowerBoundEvaluator"],
        'lower_bound_quantile': [0.01, 0.02, 0.03, 0.04, 0.05],
        'algo_seed': [i for i in range(10)],
    }
    run_pooled_experiments(params, 10)


def population_size_experiment():
    params = {
        'experiment_name': ["Population size experiment"],
        'pop_size': [30, 40, 50, 60, 80, 100],
        'evaluator': ["StandardEvaluator"],
        'algo_seed': [i for i in range(30)],
    }
    run_pooled_experiments(params, processes=10)


def test_experiment_on_surf():
    params = {
        'experiment_name': ["test experiment"],
        'pop_size': [30, 40],
        'evaluator': ["StandardEvaluator"],
        'algo_seed': [i for i in range(3)],
        'termination_arg': [10]
    }
    run_pooled_experiments(params, 6)

def heuristic_experiment_toy():
    args = ...
    # args = Parser().get_parser().parse_args()
    args.case_study = 'Sioux Falls Expanded'
    args.algo_name = 'IncreasingSlackHeuristic'
    config = Config(args)
    env, algo = config.initialize()
    algo.get_res(env)


def heuristic_experiment():
    args = {
        'experiment_name': 'Heuristic SFE',
        'algo_name': 'Heuristic2'}
    config = Config(args)
    env, algo = config.initialize()
    algo.get_res(env)




if __name__ == "__main__":
    # Call `main.py` with extracted parameters
    python_path = sys.executable
    command = [python_path, "main.py"] + ["--expe_id=34"]
    # Run the command safely
    try:
        subprocess.run(command, check=True, env=os.environ)
    except subprocess.CalledProcessError as e:
        print(f"Error: main.py failed with exit code {e.returncode}")
