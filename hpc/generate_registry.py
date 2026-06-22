# Generates the experiment "registry" (hpc/registry.json) that the runners / HPC tasks index into.
# Edit the grid at the bottom and run:  python hpc/generate_registry.py
import copy
import itertools
import json
import os
from pathlib import Path
from typing import Dict, Any, List

# The registry lives alongside this script, in the hpc/ folder.
_HPC_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = _HPC_DIR / "registry.json"
REGISTRY_COMBINATORIAL_PATH = _HPC_DIR / "registry_combinatorial.json"


def generate_experiment_runs_combinatorial(experiment_name: str, parameters: Dict[str, List[Any]]):
    """
    Generates all combinations of the given parameters and saves them to hpc/registry_combinatorial.json

    Args:
        experiment_name (str): The name of the experiment.
        parameters (Dict[str, List[Any]]): A dictionary of parameter names and their possible values.
    Returns:
        None
    """

    # Ensure 'algo_seed' is last in the order for correct sorting
    parameters = dict(sorted(parameters.items(), key=lambda kv: kv[0] != 'algo_seed'))

    # Generate all combinations, but sort by seed afterward
    keys, values = zip(*parameters.items())
    all_combinations = list(itertools.product(*values))

    # Sort so that seed varies slowest
    seed_index = keys.index('algo_seed')
    all_combinations.sort(key=lambda x: x[seed_index])

    combinations = [dict(zip(keys, v)) for v in all_combinations]

    # Create a list of experiment runs
    new_experiments = [{"experiment_name": experiment_name, "parameters": combo} for combo in combinations]

    # Write to JSON file
    output_file = str(REGISTRY_COMBINATORIAL_PATH)
    # Load existing experiments if file exists
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                existing_experiments = json.load(f)
            except json.JSONDecodeError:
                existing_experiments = []
    else:
        existing_experiments = []

    # Append only new unique experiments
    existing_set = {json.dumps(exp, sort_keys=True) for exp in existing_experiments}
    new_unique_experiments = [exp for exp in new_experiments if json.dumps(exp, sort_keys=True) not in existing_set]

    if new_unique_experiments:
        existing_experiments.extend(new_unique_experiments)
        with open(output_file, "w") as f:
            json.dump(existing_experiments, f, indent=4)
        print(f"Added {len(new_unique_experiments)} new experiment runs to '{output_file}'")
    else:
        print("No new experiment runs added (all were already present).")

def generate_experiment_runs(experiment_name: str, parameter_list: List, algo_seeds):
    experiments = []

    # Create a list of experiment runs
    for ps in parameter_list:
        for i in algo_seeds:
            pss = copy.deepcopy(ps)
            pss['algo_seed'] = i
            experiments_entry = {
                'experiment_name': experiment_name,
                'parameters': pss
            }
            experiments.append(experiments_entry)

    # Write to JSON file
    sorted_experiments = sorted(experiments, key=lambda x: x['parameters']['algo_seed'])
    output_file = str(REGISTRY_PATH)
    with open(output_file, "w") as f:
        json.dump(sorted_experiments, f, indent=4)
    print(f"Wrote {len(sorted_experiments)} experiments to {output_file}")


if __name__ == "__main__":
    """
    # Example Usage
    experiment_name = 'sensitivity experiment'
    parameters = []
    candidates = [{'evaluator': "LowerBoundEvaluator",
                       'lower_bound': "XGBoost",
                       'lower_bound_quantile': 0.2,
                       },
                  {'evaluator': "StandardEvaluator"}]
    for rc in [0.9, 1.1, 100]:
        for candidate in candidates:
            to_add = copy.deepcopy(candidate)
            to_add['case_study'] = f'Sioux Falls road capacity {rc}'
            parameters.append(to_add)

    for cc in [0.7, 100]:
        for candidate in candidates:
            to_add = copy.deepcopy(candidate)
            to_add['case_study'] = f'Sioux Falls construction capacity {rc}'
            parameters.append(to_add)
    for n in ['Less', 'More']:
        for candidate in candidates:
            to_add = copy.deepcopy(candidate)
            to_add['case_study'] = f'Sioux Falls {n} Connected'
            parameters.append(to_add)
    algo_seeds = [i for i in range(30)]
    generate_experiment_runs(experiment_name, parameters, algo_seeds)
    
    experiment_name = 'weighted slack heuristic experiment 2'
    parameters = [{'algo_name': 'WeightedSlackHeuristic'}]
    generate_experiment_runs(experiment_name, parameters, [i for i in range(1)])
    """
    # Shorthand (see ../CLAUDE.md "Experiment campaign"): <evaluator>|<surrogate>|<quantile>.
    #   * EP|*  -> LowerBoundEvaluator   (Elimination-Pruning PLBE)
    #   * LE|*  -> ApproximateEvaluator  (Lazy-Eval PLBE)
    #   * *|X|q -> XGBoost quantile lower bound at quantile q
    #   * *|H|- -> Heuristic (SubsetMaxRegressor) lower bound
    #   * S|-|- -> StandardEvaluator         (the control: plain NSGA-II, exact every eval)
    #   * SS|X|0.5 -> ScheduleSurrogateEvaluator (out-of-the-box SAEA: whole-schedule XGBoost
    #                 surrogate, median pre-selection; schedule_surrogate_quantile defaults to 0.5)
    #
    # ---- E1 on SF-12 (DONE) ----
    # Full 12-config grid (EP/LE x {XGBoost@{0.05,0.1,0.2,0.5}, Heuristic} + S|-|- + SS|X|0.5) x
    # 30 seeds = 360 runs. Its registry is archived as hpc/registry_exp0.json and the grid code is
    # in git history. Result: best PLBE variant = EP|X|0.2 (LowerBoundEvaluator, XGBoost, q=0.2),
    # carried forward as the PLBE arm of E2-E4 (see ../EXPERIMENTS.md).

    # ---- E2 on SF-76 (Sioux Falls Expanded): headline efficiency + EP-vs-LE + SAEA at scale ----
    # 8 configs x 30 seeds = 240 runs (single-threaded, 24 h TIME_BUDGET). Native-array dispatch:
    #   sbatch --array=0-14 hpc/submit_array.sh        # ceil(240/16)-1 = 14
    # PLBE arms: both EP (LowerBoundEvaluator) and LE (ApproximateEvaluator) swept at the SAME
    # q in {0.05,0.2,0.5}, giving matched EP-vs-LE pairs at every quantile (0.2 = headline EP|X|0.2).
    # Plus the S|-|- control and SS|X|0.5 SAEA.
    experiment_name = 'E2 SF-76'
    case_study = 'Sioux Falls Expanded'
    parameters = [
        # PLBE -- Elimination-Pruning (EP), XGBoost quantile sweep (0.2 = best-E1 PLBE)
        {'case_study': case_study, 'evaluator': 'LowerBoundEvaluator',
         'lower_bound': 'XGBoost', 'lower_bound_quantile': 0.05},              # EP|X|0.05
        {'case_study': case_study, 'evaluator': 'LowerBoundEvaluator',
         'lower_bound': 'XGBoost', 'lower_bound_quantile': 0.2},               # EP|X|0.2  (best-E1 PLBE)
        {'case_study': case_study, 'evaluator': 'LowerBoundEvaluator',
         'lower_bound': 'XGBoost', 'lower_bound_quantile': 0.5},               # EP|X|0.5
        # PLBE -- Lazy-Eval (LE), XGBoost quantile sweep -- identical to the EP arm for a direct
        # elimination-pruning-vs-lazy-eval comparison at every quantile
        {'case_study': case_study, 'evaluator': 'ApproximateEvaluator',
         'lower_bound': 'XGBoost', 'lower_bound_quantile': 0.05},              # LE|X|0.05
        {'case_study': case_study, 'evaluator': 'ApproximateEvaluator',
         'lower_bound': 'XGBoost', 'lower_bound_quantile': 0.2},               # LE|X|0.2  (best LE)
        {'case_study': case_study, 'evaluator': 'ApproximateEvaluator',
         'lower_bound': 'XGBoost', 'lower_bound_quantile': 0.5},               # LE|X|0.5
        # Baselines
        {'case_study': case_study, 'evaluator': 'StandardEvaluator'},          # S|-|-    (control)
        {'case_study': case_study, 'evaluator': 'ScheduleSurrogateEvaluator'}, # SS|X|0.5 (SAEA)
    ]
    algo_seeds = [i for i in range(30)]

    generate_experiment_runs(experiment_name, parameters, algo_seeds)


