"""Standalone single-experiment dispatcher over a hardcoded cartesian grid.

``main(expe_id)`` expands the grid defined in ``population_size_experiment`` (edit it to change the
sweep) and runs the ``expe_id``-th combination. A simpler legacy alternative to the registry path
(``run_single_instance.py`` + ``hpc/registry.json``); prefer the registry for real sweeps.
"""
import argparse
import itertools
import warnings

from Src.config2 import Config

warnings.filterwarnings("ignore", category=RuntimeWarning)


def train_with_args(args):
    # time.sleep(np.random.rand())
    config = Config(args)
    env, algo = config.initialize()
    algo.get_res(env)


def population_size_experiment():
    params = {
        'experiment_name': ["Population size experiment"],
        'pop_size': [30, 40, 50, 60, 80, 100],
        'evaluator': ["StandardEvaluator"],
        'algo_seed': [i for i in range(30)],
    }
    return params


def main(expe_id):
    print(f'Running experiment with ID: {expe_id}')
    # Then turn experiment ID into a set of parameters
    params = population_size_experiment()
    # Then extract the params you want from that
    keys = params.keys()
    if not all([isinstance(key, list) for key in params.values()]):
        raise TypeError("All parameters for a pooled experiment need to be lists, even 1-element items.")
    combinations = list(itertools.product(*params.values()))
    result = [dict(zip(keys, combination)) for combination in combinations]
    args = result[expe_id]
    print(f"Experiment args: {args}")
    # Then run the experiment
    train_with_args(args)


if __name__ == "__main__":
    # Define argument: expe_id (integer)
    parser = argparse.ArgumentParser(description="Run experiment with SLURM array.")
    parser.add_argument(
        "--expe_id", type=int, help="Experiment ID (used for SLURM array jobs)",)
    # Parse arguments
    args = parser.parse_args()
    main(args.expe_id)
