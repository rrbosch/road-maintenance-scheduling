# Experiment Setup — Computational Budget

## Decisions

| Parameter | Value | Rationale |
|---|---|---|
| Independent repetitions (seeds) | **30** | Matches the manuscript / CACAIE revision protocol; well above the statistical minimum |
| Threads per run | **1** | The NSGA-II inner loop (ask/eval/tell + traffic assignment) is serial; parallelism is across *runs*, not within |
| Wallclock per run | **24 h** | NSGA-II's `TIME_BUDGET` in `NSGA2.get_res`; the SLURM walltime matches it |

---

## Why these values

### Seed count
The manuscript reports results over **30 random seeds** with non-parametric significance testing
(`derrac_practical_2011`), which is what the experiment grids in `hpc/generate_registry.py` produce
(`algo_seeds = range(30)`). This is comfortably above the 3–10-seed minimum that
Agarwal et al. (2021, *"Deep RL at the Edge of the Statistical Precipice"*, NeurIPS) argue is
needed for reliable aggregate statistics, and it gives tight confidence intervals for the
hypervolume / min-distance / Pareto-size comparisons.

### Single-threaded runs
Unlike multi-worker RL training, a road-scheduling run is a single serial NSGA-II loop whose cost
is dominated by the traffic assignment (one Conjugate-Frank-Wolfe solve per unique scenario).
There is no within-run parallelism to exploit, so each HQ task reserves **one core** (`--cpus=1`)
and throughput comes from packing many runs onto a node.

### 24 h walltime
`NSGA2.get_res` runs until `TIME_BUDGET = 24*3600` seconds (it ignores `termination_arg`, which
only names resume files). The SLURM `--time=24:00:00` matches this. A run killed before its budget
is spent resumes from the rolling `algo.pkl` (`elapsed_time` is pickled), so no work is lost.

---

## Mapping to code

- **Seeds**: baked into the registry by `hpc/generate_registry.py`
  (`generate_experiment_runs(name, params, seeds)` → one `hpc/registry.json` entry per
  (config × seed), with `algo_seed` in each entry's `parameters`). No per-seed config files.
- **Dispatch**: `hpc/hq_task.sh` → `python hpc/run_task.py --expe_id=$HQ_TASK_ID` →
  `run_single_instance.process_experiment` (resume-or-start).
- **Single thread**: nothing to set in the config; ensure the HQ submit uses `--cpus=1`.

---

## Snellius node sizing

Standard Snellius CPU nodes have ~**128 cores** (AMD). With single-threaded tasks:

```
128 cores / 1 core per run = up to 128 concurrent runs per node
```

Size `submit.sh --nodes` to `ceil(num_experiments / 128)` so all runs start at t=0 and each gets
the full 24 h:

| Configs × seeds | Total runs | Nodes (24 h slot) |
|---|---|---|
| 9 × 30 (e.g. "experiment 1") | 270 | 3 |
| 5 × 30 | 150 | 2 |
| 1 × 30 | 30 | 1 |

**Memory:** each run keeps an in-memory traffic-result cache (`TotalTravelDelay.results`, a
`FIFOCache` capped at `traffic_cache_size`, default 200 000 entries ≈ 100–150 MB). 128 concurrent
runs ⇒ ~15–20 GB/node, well within a Snellius node's RAM. Lower `traffic_cache_size` (a `Config`
arg) if packing more runs per node.

> Note: `requirements.txt` still lists `SQLAlchemy` and `OpenMatrix`, which are no longer used
> (the SQLite cache was removed in overhaul item 1). They are harmless to install but can be pruned.
