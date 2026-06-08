# Configuration & experiment grids

Every run is built from a single settings dict passed to **`Src/config2.py::Config`**. `Config`
holds all run settings as attributes with defaults, overrides them from the input dict, builds a
nested results directory, and `initialize()` returns `(env, algo)`. An experiment is just a set of
these dicts (a *grid*).

## Config parameters

Operator / evaluator / algorithm choices are stored as **strings** (so they live cleanly in grids,
JSON, and logs) and resolved to classes via `Src/Algorithms/registry.py`.

| Parameter | Default | Meaning / allowed values |
|---|---|---|
| `experiment_name` | `None` | **Required.** Top-level results folder name. |
| `case_study` | `'Sioux Falls Expanded'` | Subfolder under `Environments/input/`. See list below. |
| `objectives` | `{'SL', 'TTD'}` | Objective set: `SL` (tardiness/risk), `TTD` (travel delay). |
| `algo_name` | `'NSGA2'` | `NSGA2`, or a `*Heuristic` algorithm. |
| `algo_seed` | `1` | RNG seed (one run per seed). |
| `pop_size` | `100` | NSGA-II population size. |
| `evaluator` | `'ApproximateEvaluator'` | `StandardEvaluator` (exact), `LowerBoundEvaluator` (PLBE), `ApproximateEvaluator`. |
| `lower_bound` | `'XGBoost'` | TTD surrogate for the LB evaluators: `XGBoost` (quantile), `Heuristic` (`SubsetMaxRegressor`). |
| `lower_bound_quantile` | `0.2` | Quantile for the XGBoost lower bound (e.g. `0.05, 0.1, 0.2, 0.5`). |
| `sampling` | `'WeightedSlackSampling'` | Initial population sampler. |
| `crossover` | `'CompositeCrossover'` | Crossover operator. |
| `mutation` | `'CompositeMutation'` | Mutation operator. |
| `repair` | `'TestRepair'` | Repair operator. |
| `termination` | `'IterationTermination'` | Termination operator. |
| `termination_arg` | `2000` | See the gotcha below. |
| `callback` | `'OperatorSuccessCallback'` | Per-generation callback. |
| `traffic_cache_size` | `200_000` | Max entries in the in-memory traffic-result cache (`None` = unbounded). |
| `sims` | `{'traffic'}` | Active simulations (only `traffic` is used). |
| `problem` | `'Problem_py'` | Problem class (only one). |

The full set of registered operator/algorithm names lives in `Src/Algorithms/registry.py`
(`REGISTRY`). To add a new operator, import it there and add it to the list.

### ⚠️ `termination_arg` does **not** stop the run

`NSGA2.get_res` runs the manual ask/eval/tell loop until a **24 h wall-clock `TIME_BUDGET`**,
ignoring pymoo's own termination. `termination_arg` only names resume-skip files. For short test
runs, set a small `pop_size` and a short `termination_arg`, and stop the process manually — or rely
on the 24 h cap on HPC.

## Available case studies

Subfolders of `Environments/input/`:

- `Sioux Falls Expanded` — the main literature case study.
- `Anaheim` — larger literature case study (914 links / 416 nodes / 1,406 OD), 80 congestion
  corridors as projects (overhaul item 2).
- Parametric Sioux Falls variants: `Sioux Falls road capacity {0.9, 1.1, 100}`,
  `Sioux Falls construction capacity {0.7, 100}`.
- Structural Sioux Falls variants: `Sioux Falls Less Connected`, `Sioux Falls More Connected`.

## Defining an experiment grid

### Option A — hardcoded grids in `run.py` (local)

Each function in `run.py` builds a `params` dict where **every value is a list**; the cartesian
product is expanded into one run per combination, then dispatched via a `multiprocessing.Pool`
(or single-threaded). Example:

```python
def XGBoost_experiment():
    params = {
        'experiment_name': ["XGBoost"],
        'evaluator': ["LowerBoundEvaluator"],
        'lower_bound_quantile': [0.01, 0.02, 0.03, 0.04, 0.05],
        'algo_seed': [i for i in range(10)],
    }
    run_pooled_experiments(params, processes=10)
```

Select which function runs at the bottom of `run.py`, then `python run.py` (or
`python run.py single_thread`).

### Option B — the registry JSON (local pool or HPC)

For larger sweeps and Snellius, build a flat **registry** (`hpc/registry.json`): a list of
`{experiment_name, parameters}` entries, one per (config × seed). Edit the grid at the **bottom of
`hpc/generate_registry.py`**, then generate it:

```bash
python hpc/generate_registry.py     # writes hpc/registry.json
python hpc/count_experiments.py     # prints N and the HQ --array hint
```

The generator helper `generate_experiment_runs(name, parameter_list, algo_seeds)` expands each
parameter dict over the seeds. Example grid (this is the committed default — "experiment 1"):

```python
experiment_name = 'experiment 1'
parameters = []
for evaluator in ['LowerBoundEvaluator', 'ApproximateEvaluator']:
    parameters.append({'evaluator': evaluator, 'lower_bound': 'Heuristic'})
    for q in [0.05, 0.1, 0.2, 0.5]:
        parameters.append({'evaluator': evaluator, 'lower_bound': 'XGBoost',
                           'lower_bound_quantile': q})
parameters.append({'evaluator': 'StandardEvaluator'})
generate_experiment_runs(experiment_name, parameters, algo_seeds=range(30))
```

This produces `(2 × 5 + 1) × 30 = 330` registry entries. See [running.md](running.md) for how to
execute the registry.
