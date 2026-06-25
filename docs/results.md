# Results: where they're written & how to analyze them

## Where

All output lands under a nested directory built from the experiment name and the non-default
parameters:

```
Experiments/<experiment_name>/<key>_<value>/<key>_<value>/.../
```

For example, a run with `experiment_name='experiment 1'`, `evaluator='LowerBoundEvaluator'`,
`lower_bound='XGBoost'`, `lower_bound_quantile=0.2`, `algo_seed=3` writes to:

```
Experiments/experiment 1/evaluator_LowerBoundEvaluator/lower_bound_XGBoost/lower_bound_quantile_0.2/algo_seed_3/
```

`Experiments/` is **git-ignored** (large generated artifacts).

## The results schema (one run directory)

Written by `Src/Utils/results_io.py`, called from `NSGA2.get_res` once per generation. (This
replaced the old per-generation `F_<gen>.csv` / `X_<gen>.csv` explosion; **old results are not
supported** — overhaul item 7.)

| File | Contents |
|---|---|
| `config.json` | Machine-readable run metadata (all `Config` attributes). Replaces the old `config.txt`. |
| `progress.csv` | One row per generation: `iteration, time, time_cum, pareto_set_size, n_computed` + pruning diagnostics (`exact_evals, lb_pruned, scenarios_materialized, n_estimated`, and `false_pruned` — nonzero only in the E2 `count_false_pruning` diagnostic mode). |
| `fronts.csv` | Long format `generation,<obj0>,<obj1>` — the raw Pareto fronts, so any metric / reference point can be recomputed offline. Logged every `Config.fronts_log_interval` generations (default 10) plus the **final** front, since the full cumulative front written every generation dominated a run's on-disk size; set `fronts_log_interval=1` for the legacy every-generation behavior. Trajectory plots just sample more coarsely; final-state metrics are exact. |
| `final_solutions.csv` | `sol_idx, x0..xN` — start-time vectors of the latest front (feeds the E3 schedule/Gantt interpretation). |
| `surrogate.csv` | One row per regressor retrain: `n_computed, quantile, mape, pinball_loss, model` (the surrogate-accuracy learning curve). `model` is `component` (PLBE per-scenario lower bound) or `schedule` (the item-11 whole-schedule baseline). |
| `algo.pkl` | Rolling pickle for crash-safe resume, written atomically (temp + `os.replace`). |
| `output_log.txt` | Per-run log (when launched via `run_single_instance.py`). |

`n_computed` is the **unique-simulations** counter (total traffic assignments run); it also paces
regressor retraining (every +1000 assignments).

> **`ExactParetoSolver` runs** write the same schema as a single "generation" (the exact true Pareto
> front): `config.json`, `fronts.csv`, `final_solutions.csv`, `algo.pkl`, and a one-row
> `progress.csv` with two extra columns — `total_enumerated` and `feasible`. There is **no**
> `surrogate.csv`. The `analysis/` pipeline consumes these run dirs unchanged.

## Analyzing results

The `analysis/` package consumes **only** this schema — it discovers runs by walking for
`config.json` (no folder-name parsing, no per-generation file globbing).

```bash
python analysis/run_analysis.py "Experiments/<experiment_name>" ["Experiments/<another>" ...]
```

For each experiment directory it:

1. discovers runs via `config.json` (`analysis/load.py`),
2. computes per-generation Pareto metrics from `fronts.csv` (`analysis/metrics.py` — numba
   Hypervolume, Max Spread, Min Distance to Origin, Pareto Front Size, Unique Simulations; default
   reference point `[2e3, 2e9]`),
3. aggregates across seeds (mean + 90% CI) by iteration and by wall-time grid,
4. emits figures + a metrics CSV under `analysis/output/<experiment_name>/` (`analysis/plots.py`):
   final/super Pareto fronts, metric-vs-iteration and metric-vs-time CI plots, iteration-vs-time,
   sensitivity boxplots, plus the surrogate learning curve and pruning diagnostics.

> The `analysis/` package is the supported analysis tool. (Earlier legacy plotting scripts
> `results_processing2.py` / `results_processing_old.py` are not part of this published repo;
> `analysis/` ports their figures.) Figure-styling polish (color keys, fonts, E3 Gantts) is tracked
> separately as overhaul item 13.
