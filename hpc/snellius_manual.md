# Snellius Run Manual

Step-by-step commands for running the road-renovation-scheduling experiments on Snellius via a
**native SLURM job array** ([`hpc/submit_array.sh`](submit_array.sh)). There is no HyperQueue server
or worker daemon to keep alive, so the run **survives SSH/app disconnects** — a login-node HQ server
dying mid-run is what killed an earlier campaign. The older HQ-based workflow is preserved verbatim in
[`snellius_manual_old.md`](snellius_manual_old.md) (its scripts `submit.sh` / `hq_task.sh` are kept as
a fallback).

Each run is single-threaded, so an array element packs **16 experiments onto one 16-core slot** (the
Snellius minimum billed unit). Partition/sharing/QOS/billing facts that make this efficient are in
[`snellius_reference.md`](snellius_reference.md); the short version:

- **16 cores = minimum billed slot** (1/8 of a 128-core rome node). A 1-core array element would still
  be billed 16 cores — so we bundle 16 single-threaded runs per element → 100% core use, no waste.
- **Do NOT set `--mem-per-cpu`.** Each run needs ~100–150 MB; the default (~28 GiB/slot) keeps billing
  at the 16-core tier. An explicit `--mem-per-cpu` can silently tip billing to the next tier (2×).
- **128 jobs/user QOS.** 360 runs / 16 = **23 elements**, far under the cap, so every run starts at
  t=0. (A naive 360-element 1-core array would also waste 16× on billing *and* only run 128 at once.)

Adjust the editable paths once: code lives at `~/python_restructured/`, the venv at
`~/.local/venv_road/` (set in `hpc/submit_array.sh`).

---

## Quick start

All commands from a login node, with the venv and code already in place:

```bash
cd ~/python_restructured/
N=$(python hpc/count_experiments.py | awk 'NR==1{print $1}')   # e.g. 360
ELEMS=$(( (N + 15) / 16 - 1 ))                                  # ceil(N/16)-1, e.g. 22

sbatch --array=0 hpc/submit_array.sh             # smoke test: first bundle of 16 (or fewer)
sbatch --array=0-$ELEMS hpc/submit_array.sh      # full campaign, e.g. --array=0-22
squeue -u $USER                                  # watch it run
```

Results land under `Experiments/<experiment_name>/<key_value>/.../`. No `hq`, no separate worker
job, no server.

---

## 1. One-time: Python environment

Only the first time (or after a Python-module change). This project's runs are single-threaded and
need `pymoo`, `numba`, `xgboost`, `scikit-learn`, `networkx`, `scipy` (see `requirements.txt`).

```bash
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

python3 -m venv ~/.local/venv_road          # skip if it already exists
source ~/.local/venv_road/bin/activate
cd ~/python_restructured/
pip install -r requirements.txt
```

(`hpc/submit_array.sh` loads these modules and activates this venv itself per array element — you
don't need to activate it just to submit.)

---

## 2. Sync code + build the experiment queue

Sync the latest code to Snellius (from your laptop):

```bash
scp -r Src/ Environments/ analysis/ hpc/ requirements.txt \
       main.py run.py run_single_instance.py run_in_IDE.py \
       <username>@snellius.surf.nl:~/python_restructured/
```

The **registry** is `hpc/registry.json` (the flat list of experiments, one entry per config × seed).
Edit the grid at the bottom of `hpc/generate_registry.py`, then generate it (locally or on Snellius):

```bash
cd ~/python_restructured/
source ~/.local/venv_road/bin/activate
python hpc/generate_registry.py    # writes hpc/registry.json
```

> `generate_experiment_runs(name, parameter_list, algo_seeds)` expands each parameter dict over the
> seeds and writes the flat JSON list. Input data (`Environments/input/...`) and outputs
> (`Experiments/...`) are anchored to the repo, so tasks run correctly regardless of cwd.

---

## 3. Count experiments → array size

```bash
python hpc/count_experiments.py
```

It prints `N` (and an old HQ hint you can ignore). The array range is **`0-(ceil(N/16)-1)`** because
each element runs 16 experiments:

| N (runs) | `--array` | Elements |
|---|---|---|
| 360 (E1: 12 × 30) | `0-22` | 23 |
| 90 (5 × 30 ≈ E2) | `0-5` | 6 |
| 30 (1 × 30) | `0-1` | 2 |

(One-liner: `ELEMS=$(( (N + 15) / 16 - 1 ))`.)

---

## 4. Submit the array

The `#SBATCH` defaults in `hpc/submit_array.sh` are `--array=0-22`, `--cpus-per-task=16`,
`--time=25:00:00`, `--partition=rome`, **no `--mem-per-cpu`**. A CLI `--array` overrides the directive:

```bash
# Smoke test the first bundle (confirm it starts/resumes and writes Experiments/...):
sbatch --array=0 hpc/submit_array.sh

# Full run:
sbatch --array=0-22 hpc/submit_array.sh

# Any subset is fine, e.g. just the last bundle:
sbatch --array=22 hpc/submit_array.sh
```

`sbatch` prints a `<jobid>`. All 23 elements run concurrently (QOS allows 128 jobs/user; single-node
16-core jobs share nodes, so no whole-node waste). Inside each element, 16 experiments run in
parallel, each pinned to one core (`taskset`) with capped library threads.

> **Walltime headroom:** the walltime is **25 h** = the 24 h `TIME_BUDGET` + 1 h, so a run finishes
> its budget and flushes its final generation well before SLURM's hard kill. (Even without the
> headroom, the rolling per-gen `algo.pkl` means at most the last partial generation is lost and is
> recovered on resubmit.)

---

## 5. Monitor

```bash
squeue -u $USER                                                 # queued / running array elements
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS%12        # per-element state
tail -f hpc/logs/slurm_<jobid>_0.out                           # live log for array element 0
```

Per-element stdout/stderr: `hpc/logs/slurm_%A_%a.out` / `.err` (`%A` = array job id, `%a` = element
index). Each element's log interleaves its 16 experiments. Per-run output lands under
`Experiments/<experiment_name>/<key_value>/.../` as the item-7 schema (`config.json`, `progress.csv`,
`fronts.csv`, `final_solutions.csv`, `surrogate.csv`, `algo.pkl`/`algo_backup.pkl`).

**Disconnect test:** after submitting, close your session and reconnect — `squeue -u $USER` still
shows the array. That's the whole point of dropping the HQ server.

---

## 6. Resuming after a timeout or partial failure

Each run targets NSGA-II's **24 h `TIME_BUDGET`** (SLURM walltime 25 h = budget + 1 h flush). If an element is
killed before its runs' budgets are spent, just **resubmit the same array** —
`run_single_instance.process_experiment` reloads each run's rolling `algo.pkl` and continues (the
accumulated `elapsed_time` is pickled, so a run whose 24 h is already spent resumes to a fast no-op).
Resubmitting the **whole** array is therefore safe:

```bash
sbatch --array=0-22 hpc/submit_array.sh
```

To rerun only specific bundles (e.g. ones that hit a node failure), submit just those element
indices — element `e` covers experiments `16*e … 16*e+15`:

```bash
sbatch --array=3,7,12 hpc/submit_array.sh
```

To inspect failures: `sacct -j <jobid> --format=JobID,State,ExitCode`, then check the run's
`Experiments/.../output_log.txt`.

---

## 7. Pull results back

On your **laptop**:

```bash
rsync -av <username>@snellius.surf.nl:~/python_restructured/Experiments/ "Experiments/"
```

Then analyze locally with the `analysis/` package (run **off** the OneDrive tree — the `get_res`/
analysis loops I/O-stall on OneDrive).

---

## Quick reference

| Action | Command |
|---|---|
| Sync code up (laptop) | `scp -r Src/ Environments/ analysis/ hpc/ … snellius:~/python_restructured/` |
| Build registry | `python hpc/generate_registry.py` |
| Count queue | `python hpc/count_experiments.py` |
| Array size | `ELEMS=$(( (N + 15) / 16 - 1 ))` |
| Smoke test one bundle | `sbatch --array=0 hpc/submit_array.sh` |
| Submit full E1 | `sbatch --array=0-22 hpc/submit_array.sh` |
| Watch queue | `squeue -u $USER` |
| Per-element state | `sacct -j <jobid> --format=JobID,State,Elapsed` |
| Cancel | `scancel <jobid>` (or `scancel -u $USER`) |
| Pull results (laptop) | `rsync -av snellius:~/python_restructured/Experiments/ Experiments/` |
