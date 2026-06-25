# Experiment E1 on SF-12 — Snellius run guide

Step-by-step for the **first comparison experiment** on the `Sioux Falls 12` instance:
**standard NSGA-II** vs. the **PLBE variants** (different surrogate/quantile options) vs. an
**out-of-the-box SAEA** baseline. All runs are single-threaded with a 24 h `TIME_BUDGET`.

This guide is specific to E1/SF-12; the general HPC mechanics live in
[`snellius_manual.md`](snellius_manual.md) and [`experiment_setup.md`](experiment_setup.md).

---

## 1. What gets run

The grid is already defined in [`generate_registry.py`](generate_registry.py) (`experiment_name =
'E1 SF-12'`). **12 algorithm configurations × 30 seeds = 360 runs.** Shorthand
`<evaluator>|<surrogate>|<quantile>` (see `../CLAUDE.md` → "Experiment campaign"):

| # | Config | `evaluator` | `lower_bound` | `lower_bound_quantile` | What it is |
|---|---|---|---|---|---|
| 1 | `S\|-\|-` | `StandardEvaluator` | — | — | **Control:** plain NSGA-II, exact eval every solution |
| 2 | `SS\|X\|0.5` | `ScheduleSurrogateEvaluator` | (XGBoost) | (0.5 default) | **Out-of-the-box SAEA:** whole-schedule XGBoost surrogate, median pre-selection |
| 3–6 | `EP\|X\|q` | `LowerBoundEvaluator` | `XGBoost` | 0.05 / 0.1 / 0.2 / 0.5 | PLBE, elimination-pruning, XGBoost quantile bound |
| 7 | `EP\|H\|-` | `LowerBoundEvaluator` | `Heuristic` | — | PLBE, elimination-pruning, SubsetMaxRegressor bound |
| 8–11 | `LE\|X\|q` | `ApproximateEvaluator` | `XGBoost` | 0.05 / 0.1 / 0.2 / 0.5 | PLBE, lazy-eval, XGBoost quantile bound |
| 12 | `LE\|H\|-` | `ApproximateEvaluator` | `Heuristic` | — | PLBE, lazy-eval, SubsetMaxRegressor bound |

All 12 run under NSGA-II (`algo_name='NSGA2'`, default `pop_size=100`); only the **evaluator**
(and its surrogate) differs, so the comparison isolates the evaluation strategy.

To change the grid (e.g. fewer seeds for a smoke test, add quantiles), edit the `'E1 SF-12'` block
at the bottom of `generate_registry.py` and regenerate (step 4).

---

## 2. One-time: environment + code on Snellius

From a login node (only the first time, or after a Python-module change):

```bash
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
python3 -m venv ~/.local/venv_road           # skip if it already exists
source ~/.local/venv_road/bin/activate
cd ~/python_restructured/
pip install -r requirements.txt
```

**Sync the code + the SF-12 instance from your laptop** (run locally; the `Sioux Falls 12` folder
must be present or the runs can't load the case study):

```bash
# from the repo root on your laptop
scp -r Src/ Environments/ analysis/ hpc/ requirements.txt \
       main.py run.py run_single_instance.py run_in_IDE.py \
       <username>@snellius.surf.nl:~/python_restructured/
```

> The Snellius upload should EXCLUDE generated/junk dirs (`.git/`, `Experiments/`, `temp/`,
> `other/`, `plots/`, `analysis/output/`, `__pycache__/`, and the 4 GB
> `Environments/input/Sioux Falls Expanded/traffic_results.zip`). The `scp` line above already
> picks only what's needed; just confirm `Environments/input/Sioux Falls 12/` made it across.

---

## 3. Start the HyperQueue server (once per session)

```bash
module load 2023
module load HyperQueue/0.19.0
cd ~/python_restructured/
hq server info || nohup hq server start &
hq server info            # verify it's up
```

---

## 4. Build the registry and count it

```bash
source ~/.local/venv_road/bin/activate
python hpc/generate_registry.py     # writes hpc/registry.json: 360 entries for 'E1 SF-12'
python hpc/count_experiments.py     # -> "360 experiments" + the --array hint
```

Expected:
```
360 experiments
  -> hq submit --array 0-359 --cpus=1 --pin taskset hpc/hq_task.sh
```

---

## 5. Submit the task array

`--cpus=1` (each run is single-threaded), `--pin taskset` (bind each task to its core):

```bash
hq submit --array 0-359 --cpus=1 --pin taskset hpc/hq_task.sh
```

Note the returned `<job_id>`.

---

## 6. Submit SLURM workers

360 single-threaded tasks ÷ ~128 cores/node ⇒ **3 nodes** to have everything start at t=0 (and get
its full 24 h). The simplest way with the existing `submit.sh` (which starts **one** worker on
**one** node) is to **submit it three times** — each job adds a 128-core worker, and the HQ server
load-balances the 360 queued tasks across all connected workers:

```bash
sbatch hpc/submit.sh
sbatch hpc/submit.sh
sbatch hpc/submit.sh
squeue -u $USER
```

(Alternative: edit `submit.sh` to `#SBATCH --nodes=3` and launch one worker per node with
`srun --ntasks=3 --ntasks-per-node=1 hq worker start`. The 3×`sbatch` approach needs no edit and is
more robust to partial node availability.)

If you'd rather use **fewer nodes**, that's fine too: submit `submit.sh` once (1 node = 128
concurrent). The remaining ~232 tasks queue and start as the first wave finishes — slower
wall-clock, but each still gets its full 24 h, and the run is resume-safe (step 8).

> Before submitting, set `--mail-user` in `submit.sh` to your address if you want SLURM mail.

---

## 7. Monitor

```bash
hq job list                            # all HQ jobs + status
hq job progress <job_id>               # live progress bar
hq task list <job_id> | grep FAILED    # any failures
ls hpc/logs/ && tail -f hpc/logs/slurm_<slurm_job_id>.out
```

Per-run output lands under
`Experiments/E1 SF-12/case_study_Sioux Falls 12/.../algo_seed_<n>/` as the schema from overhaul
item 7: `config.json`, `progress.csv`, `fronts.csv`, `final_solutions.csv`, `surrogate.csv`
(PLBE/SAEA runs only), `algo.pkl`/`algo_backup.pkl`.

---

## 8. Resume after a timeout / partial failure

Each run targets the 24 h `TIME_BUDGET`; the SLURM walltime matches. If a worker is killed before a
task's budget is spent, just **resubmit the whole array** — `process_experiment` reloads the rolling
`algo.pkl` and continues (a run whose 24 h is already spent resumes to a fast no-op, so re-running
the array is safe):

```bash
hq submit --array 0-359 --cpus=1 --pin taskset hpc/hq_task.sh
sbatch hpc/submit.sh        # (x3 again if you want all nodes back)
```

---

## 9. Analyze

Pull the `Experiments/E1 SF-12/` tree back to your laptop and run the analysis package (it discovers
runs via `config.json`, so no path parsing):

```bash
python analysis/run_analysis.py        # -> analysis/output/E1 SF-12/
```

This emits the Hypervolume / Min-Distance-to-Origin / Pareto-size / unique-simulations comparisons
across the 12 configs (with seed-level significance), plus the surrogate learning curves and
pruning diagnostics. Compare against the **exact Pareto front** already produced by the
branch-and-bound solver (`Experiments/SF12_BnB/...`) for an exact HV gap and a measured
false-pruning rate.

---

## Notes / gotchas

- **Startup cost of the default sampler.** `sampling='WeightedSlackSampling'` does a heavy
  heuristic warm-start (≈ projects × periods partial evals per individual). On SF-12 (12×20) that's
  a one-time cost at `ask()` for the first generation — fine under the 24 h budget, but if you want
  quick generations for a smoke test, set `sampling='FeasibleRandomSampling'` in the grid.
- **Memory:** each run's traffic cache (`FIFOCache`, default `traffic_cache_size=200000`) is
  ~100–150 MB; 128 concurrent ⇒ ~15–20 GB/node, well within a Snellius node. Lower
  `traffic_cache_size` in the grid if packing more per node.
- **Off OneDrive only.** Never run this on the OneDrive-synced copy — the `get_res` write loop
  I/O-stalls there. Snellius `$HOME` is fine.
