# Snellius Run Manual

Step-by-step commands for running the road-renovation-scheduling experiments on Snellius via
**SLURM + HyperQueue (HQ)**. SLURM allocates whole nodes; HQ dispatches a flat array of
single-threaded tasks across the allocated cores (one task = one experiment from the queue).

Adjust the editable paths once: code lives at `~/python_restructured/`, the venv at
`~/.local/venv_road/` (set in `hpc/hq_task.sh` and `hpc/submit.sh`).

---

## Quick start

All commands from a login node, assuming the venv and code are already set up:

```bash
module load 2023
module load HyperQueue/0.19.0
cd ~/python_restructured/
nohup hq server start &
python hpc/count_experiments.py                                  # -> N and the --array hint
hq submit --array 0-{N-1} --cpus=1 --pin taskset hpc/hq_task.sh  # replace N-1
sbatch hpc/submit.sh
```

---

## 1. One-time: Python environment

Only the first time (or after a Python module change). This project's runs are single-threaded
and need `pymoo`, `numba`, `xgboost`, `scikit-learn`, `networkx`, `scipy` (see `requirements.txt`).

```bash
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

python3 -m venv ~/.local/venv_road          # skip if it already exists
source ~/.local/venv_road/bin/activate
cd ~/python_restructured/
pip install -r requirements.txt
```

---

## 2. Start the HyperQueue server

The HQ server runs on the login node and persists across sessions.

```bash
module load 2023
module load HyperQueue/0.19.0
nohup hq server start &
hq server info            # verify
```

---

## 3. Sync code + build the experiment queue

Sync the latest code to Snellius (from your laptop):

```bash
scp -r Src/ Environments/ analysis/ hpc/ requirements.txt \
       main.py run.py run_single_instance.py run_in_IDE.py \
       <username>@snellius.surf.nl:~/python_restructured/
```

The **registry** is `hpc/registry.json` (the flat list of experiments). Edit the grid at the
bottom of `hpc/generate_registry.py`, then generate it (locally or on Snellius):

```bash
cd ~/python_restructured/
source ~/.local/venv_road/bin/activate
python hpc/generate_registry.py    # writes hpc/registry.json (one entry per config x seed)
```

> `hpc/generate_registry.py` is the registry generator: `generate_experiment_runs(name,
> parameter_list, algo_seeds)` expands each parameter dict over the seeds and writes the flat JSON
> list to `hpc/registry.json`. Input data (`Environments/input/...`) and outputs (`Experiments/...`)
> are anchored to the repo, so tasks run correctly regardless of the worker's cwd.

---

## 4. Count experiments

```bash
python hpc/count_experiments.py
```

Example:
```
270 experiments
  -> hq submit --array 0-269 --cpus=1 --pin taskset hpc/hq_task.sh
```

---

## 5. Submit the HQ task array

`--cpus=1` because each run is single-threaded; `--pin taskset` binds each task to its core.

```bash
hq submit --array 0-269 --cpus=1 --pin taskset hpc/hq_task.sh    # replace 269 with N-1
```

Note the returned `<job_id>` for monitoring.

---

## 6. Submit the SLURM job (spawns workers)

Size `--nodes` in `hpc/submit.sh` to `ceil(N / cores_per_node)` (≈128 cores/node) so every task
starts at t=0 and gets the full 24 h budget. Then:

```bash
sbatch hpc/submit.sh
squeue -u $USER
```

When SLURM starts the job, the HQ worker connects to the server and tasks begin.

---

## 7. Monitor

```bash
hq job list                              # all HQ jobs and status
hq job progress <job_id>                 # live progress bar
hq task list <job_id> | grep FAILED      # failed tasks
hq task list <job_id> | grep RUNNING     # in-flight tasks
ls hpc/logs/                             # SLURM stdout/stderr per job
tail -f hpc/logs/slurm_<slurm_job_id>.out
```

Per-run output lands under `Experiments/<experiment_name>/<key_value>/.../` as the new schema
(`config.json`, `progress.csv`, `fronts.csv`, `final_solutions.csv`, `surrogate.csv`,
`algo.pkl`/`algo_backup.pkl` — see overhaul item 7).

---

## 8. Resuming after a timeout or partial failure

Each run targets NSGA-II's **24 h `TIME_BUDGET`**; the SLURM walltime is also 24 h. If a task is
killed before its budget is spent, just resubmit — `run_single_instance.process_experiment` reloads
the rolling `algo.pkl` and continues (the accumulated `elapsed_time` is pickled, so a run whose 24 h
is already spent resumes to a fast no-op). Resubmitting the **whole** array is therefore safe:

```bash
hq submit --array 0-{N-1} --cpus=1 --pin taskset hpc/hq_task.sh
sbatch hpc/submit.sh
```

To inspect failures: `hq task list <job_id> --filter failed`, then check the run's
`Experiments/.../output_log.txt`.

---

## Quick reference

| Action | Command |
|---|---|
| Check HQ server | `hq server info` |
| Start HQ server | `nohup hq server start &` |
| Build registry | `python hpc/generate_registry.py` |
| Count queue | `python hpc/count_experiments.py` |
| Submit task array | `hq submit --array 0-{N-1} --cpus=1 --pin taskset hpc/hq_task.sh` |
| Submit SLURM workers | `sbatch hpc/submit.sh` |
| Monitor | `hq job progress <job_id>` |
| Cancel all tasks | `hq job cancel all` |
| Stop HQ server | `hq server stop` |
