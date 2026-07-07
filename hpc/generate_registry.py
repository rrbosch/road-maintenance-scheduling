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
REGISTRY_PATH = _HPC_DIR / "registry.json"          # active dispatch slot (submit_array.sh reads this)
REGISTRY_E1_PATH = _HPC_DIR / "registry_e1.json"
REGISTRY_E2_PATH = _HPC_DIR / "registry_e2.json"
REGISTRY_E3_PATH = _HPC_DIR / "registry_e3.json"
REGISTRY_E4_PATH = _HPC_DIR / "registry_e4.json"
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

def generate_experiment_runs(experiment_name: str, parameter_list: List, algo_seeds, output_path=REGISTRY_PATH):
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
    output_file = str(output_path)
    with open(output_file, "w") as f:
        json.dump(sorted_experiments, f, indent=4)
    print(f"Wrote {len(sorted_experiments)} experiments to {output_file}")


# Per-instance false-pruning sampling probabilities (item 12, log-and-replay). See the long note in
# __main__ for the derivation: sized from the observed total prunes/run of the carried-forward
# EP|X|0.05 arm (the variant reported in the manuscript) so the expected sampled-prune count is a
# few thousand+ while bounding disk / post-hoc cost.
FALSE_PRUNING_LOG_PROB = {
    'SF-12': 0.005,   # ~3.66M prunes/run -> ~18k sampled (k=0 -> 95% upper bound ~1.6e-4/run)
    'SF-76': 0.01,    # ~1.31M prunes/run -> ~13k sampled (k=0 -> 95% upper bound ~2.3e-4/run)
}


def build_false_pruning_redispatch_E1():
    """Re-dispatch the E1 (SF-12) PLBE arm with the metric-neutral false-pruning logger on (X=0.005).

    Only the carried-forward EP|X|0.05 arm is re-run (that is the variant whose false-pruning rate we
    report). The control / SAEA arms do no LB pruning of interest here, so they are omitted to save
    compute; add them if a baseline false-pruning column is wanted.
    """
    experiment_name = 'E1 SF-12 false-pruning'
    parameters = [{
        'case_study': 'Sioux Falls 12',
        'evaluator': 'LowerBoundEvaluator', 'lower_bound': 'XGBoost', 'lower_bound_quantile': 0.05,
        'false_pruning_log_prob': FALSE_PRUNING_LOG_PROB['SF-12'],
    }]
    generate_experiment_runs(experiment_name, parameters, algo_seeds=list(range(30)))


def build_false_pruning_redispatch_E2():
    """Re-dispatch the E2 (SF-76) PLBE arm with the metric-neutral false-pruning logger on (X=0.01)."""
    experiment_name = 'E2 SF-76 false-pruning'
    parameters = [{
        'case_study': 'Sioux Falls Expanded',
        'evaluator': 'LowerBoundEvaluator', 'lower_bound': 'XGBoost', 'lower_bound_quantile': 0.05,
        'false_pruning_log_prob': FALSE_PRUNING_LOG_PROB['SF-76'],
    }]
    generate_experiment_runs(experiment_name, parameters, algo_seeds=list(range(30)))


def build_false_pruning_redispatch_both():
    """E1 (SF-12) + E2 (SF-76) PLBE arm (EP|X|0.05) with the metric-neutral false-pruning logger on,
    written to ONE registry.json: 60 runs (30 SF-12 @ X=0.005 + 30 SF-76 @ X=0.01). Separate output
    dirs ('E1/E2 ... false-pruning'), so the existing E1/E2 result dirs are untouched."""
    groups = [
        ('E1 SF-12 false-pruning', 'Sioux Falls 12', FALSE_PRUNING_LOG_PROB['SF-12']),
        ('E2 SF-76 false-pruning', 'Sioux Falls Expanded', FALSE_PRUNING_LOG_PROB['SF-76']),
    ]
    experiments = []
    for name, case_study, xprob in groups:
        for i in range(30):
            experiments.append({'experiment_name': name, 'parameters': {
                'case_study': case_study,
                'evaluator': 'LowerBoundEvaluator', 'lower_bound': 'XGBoost', 'lower_bound_quantile': 0.05,
                'false_pruning_log_prob': xprob,
                'algo_seed': i,
            }})
    experiments = sorted(experiments, key=lambda x: x['parameters']['algo_seed'])
    with open(str(REGISTRY_PATH), "w") as f:
        json.dump(experiments, f, indent=4)
    print(f"Wrote {len(experiments)} experiments to {REGISTRY_PATH}")


# Headline trio carried through E3/E4: best PLBE (EP|X|0.05) vs the S|-|- control vs the SS|X|0.5 SAEA.
HEADLINE_TRIO = [
    {'evaluator': 'LowerBoundEvaluator', 'lower_bound': 'XGBoost', 'lower_bound_quantile': 0.05},  # EP|X|0.05
    {'evaluator': 'StandardEvaluator'},                                                            # S|-|-
    {'evaluator': 'ScheduleSurrogateEvaluator'},                                                   # SS|X|0.5
]
E4_SEEDS = 30  # full 30-seed campaign, matching E1-E3 (disk is no longer the constraint after the slim writer)


def build_e3_registry():
    """E3 (SF-76 variants): the headline trio across the 7 variant networks x 30 seeds = 630 runs.
    Written to registry_e3.json."""
    variants = [
        'Sioux Falls road capacity 0.9', 'Sioux Falls road capacity 1.1', 'Sioux Falls road capacity 100',
        'Sioux Falls construction capacity 0.7', 'Sioux Falls construction capacity 100',
        'Sioux Falls Less Connected', 'Sioux Falls More Connected',
    ]
    parameters = []
    for cs in variants:
        for c in HEADLINE_TRIO:
            entry = {'case_study': cs}
            entry.update(c)
            parameters.append(entry)
    generate_experiment_runs('E3 SF-76 variants', parameters, list(range(30)), output_path=REGISTRY_E3_PATH)


def build_e4_registry():
    """E4 (Anaheim-80): the headline trio on Anaheim x E4_SEEDS seeds. Written to registry_e4.json."""
    parameters = []
    for c in HEADLINE_TRIO:
        entry = {'case_study': 'Anaheim'}
        entry.update(c)
        parameters.append(entry)
    generate_experiment_runs('E4 Anaheim-80', parameters, list(range(E4_SEEDS)), output_path=REGISTRY_E4_PATH)


def build_e1_registry():
    """E1 (SF-12): the full 12-config down-select grid -- EP/LE x {XGBoost@{0.05,0.1,0.2,0.5},
    Heuristic} + S|-|- + SS|X|0.5 -- x 30 seeds = 360 runs, with the metric-neutral false-pruning
    logger enabled on the reported PLBE arm (EP|X|0.05, X=0.005). One campaign reproduces both the
    E1 comparison AND the false-pruning measurement (analyse with analysis/false_pruning.py).
    Written to registry_e1.json."""
    case = 'Sioux Falls 12'
    parameters = []
    for evaluator in ['LowerBoundEvaluator', 'ApproximateEvaluator']:        # EP, LE
        parameters.append({'case_study': case, 'evaluator': evaluator, 'lower_bound': 'Heuristic'})
        for q in [0.05, 0.1, 0.2, 0.5]:
            entry = {'case_study': case, 'evaluator': evaluator, 'lower_bound': 'XGBoost',
                     'lower_bound_quantile': q}
            if evaluator == 'LowerBoundEvaluator' and q == 0.05:              # EP|X|0.05 = reported arm
                entry['false_pruning_log_prob'] = FALSE_PRUNING_LOG_PROB['SF-12']
            parameters.append(entry)
    parameters.append({'case_study': case, 'evaluator': 'StandardEvaluator'})          # S|-|-
    parameters.append({'case_study': case, 'evaluator': 'ScheduleSurrogateEvaluator'})  # SS|X|0.5
    generate_experiment_runs('E1 SF-12', parameters, list(range(30)), output_path=REGISTRY_E1_PATH)


def build_e2_registry():
    """E2 (SF-76 / Sioux Falls Expanded): the 8-config grid -- EP/LE x q in {0.05,0.2,0.5} + S|-|-
    + SS|X|0.5 -- x 30 seeds = 240 runs, with the false-pruning logger on the EP|X|0.05 arm
    (X=0.01). Written to registry_e2.json."""
    case = 'Sioux Falls Expanded'
    parameters = []
    for evaluator in ['LowerBoundEvaluator', 'ApproximateEvaluator']:
        for q in [0.05, 0.2, 0.5]:
            entry = {'case_study': case, 'evaluator': evaluator, 'lower_bound': 'XGBoost',
                     'lower_bound_quantile': q}
            if evaluator == 'LowerBoundEvaluator' and q == 0.05:
                entry['false_pruning_log_prob'] = FALSE_PRUNING_LOG_PROB['SF-76']
            parameters.append(entry)
    parameters.append({'case_study': case, 'evaluator': 'StandardEvaluator'})
    parameters.append({'case_study': case, 'evaluator': 'ScheduleSurrogateEvaluator'})
    generate_experiment_runs('E2 SF-76', parameters, list(range(30)), output_path=REGISTRY_E2_PATH)


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
    # 30 seeds = 360 runs. Its registry is archived as hpc/registry_E1.json and the grid code is
    # in git history. Down-select (solution quality is the primary criterion): EP|X|0.05
    # (LowerBoundEvaluator, XGBoost, q=0.05) -- significantly better HV/IGD+ at scale (E2, paired
    # Wilcoxon); carried forward as the PLBE arm of E2-E4 (see ../EXPERIMENTS.md). EP|X|0.2 gives
    # ~16% more sim reduction at slightly lower quality (the efficiency-prioritized alternative).

    # ---- E2 on SF-76 (DONE) ----
    # 8 configs (EP/LE x q in {0.05,0.2,0.5} + S|-|- + SS|X|0.5) x 30 seeds = 240 runs on Sioux
    # Falls Expanded; registry archived as hpc/registry_E2.json. Result: PLBE dominates the Standard
    # front entirely and the SAEA front ~96% (two-set coverage), at ~85-89% fewer simulations; EP ~
    # LE; EP|X|0.05 is the quality-prioritized pick carried forward (see ../EXPERIMENTS.md).
    #
    # ---- E1/E2 false-pruning RE-DISPATCH (item 12, log-and-replay) ----
    # NB: the COMPLETED E1/E2 runs above did NOT log false-pruning samples (the
    # `false_pruning_log_prob` knob did not exist when they ran). To MEASURE the false-pruning rate
    # we re-dispatch the PLBE arm with the metric-neutral logger on. Chosen X (per-pruned-candidate
    # sampling prob), derived from the observed total prunes/run for the carried-forward EP|X|0.05 arm
    # (mean pruned/run: SF-12 ~3.66M, SF-76 ~1.31M):
    #   * SF-12: X=0.005  -> ~18k sampled prunes/run (per-run zero-rate 95% upper bound ~1.6e-4;
    #            pooled over 30 seeds ~9e-6). Disk ~1.4 MB/run (x file).
    #   * SF-76: X=0.01   -> ~13k sampled prunes/run (per-run upper bound ~2.3e-4; pooled ~8e-6).
    #            Disk ~4 MB/run (x file).
    # Both comfortably exceed "a few thousand" sampled prunes while bounding disk + post-hoc eval
    # cost. The logger is metric-neutral, so these re-runs reproduce E1/E2 metrics exactly *and* add
    # pruned_sample.csv; analyse with `python analysis/false_pruning.py <experiment_dir>`.
    # Uncomment the call you want (and adjust the experiment_name to avoid clobbering existing dirs):
    #   build_false_pruning_redispatch_E1()   # SF-12, X=0.005
    #   build_false_pruning_redispatch_E2()   # SF-76, X=0.01

    # ---- E3 on SF-76 variants: robustness across network structure / parameters ----
    # The headline trio -- best PLBE (EP|X|0.05, quality-prioritized) vs the S|-|- control vs the SS|X|0.5 SAEA -- across
    # 7 SF-76 variant networks x 30 seeds = 3 x 7 x 30 = 630 runs. Native-array dispatch:
    #   sbatch --array=0-39 hpc/submit_array.sh        # ceil(630/16)-1 = 39
    # (To also sweep EP-vs-LE / quantiles per variant, extend `configs` below -- but that multiplies
    #  the run count; E2 already established EP~LE and the quantile trade-off, so E3 keeps the trio.)
    # ---- E4 on Anaheim-80: transfer to a large realistic network (914 links) ----
    # The headline trio on Anaheim x 30 seeds = 90 runs. See build_e4_registry().
    #
    # Build all four canonical experiment registries -> registry_e{1,2,3,4}.json (see hpc/README.txt).
    # e1/e2 carry the metric-neutral false-pruning logger on the reported EP|X|0.05 arm, so one
    # campaign reproduces the comparison AND the false-pruning rate; e3/e4 are the headline trio.
    # (The build_false_pruning_redispatch_* helpers above remain as a standalone-fp alternative.)
    build_e1_registry()   # -> registry_e1.json  (E1 SF-12,          360 runs, + false-pruning arm)
    build_e2_registry()   # -> registry_e2.json  (E2 SF-76,          240 runs, + false-pruning arm)
    build_e3_registry()   # -> registry_e3.json  (E3 SF-76 variants, 630 runs)
    build_e4_registry()   # -> registry_e4.json  (E4 Anaheim-80,      90 runs)


