# Running experiments

There are four entry points. All of them build a `Config` and call `algo.get_res(env)`; they
differ only in how the grid of runs is sourced and dispatched. See
[configuration.md](configuration.md) for how to define a grid.

> **Reminder:** each run executes until NSGA-II's **24 h `TIME_BUDGET`** (not `termination_arg`).
> For quick local tests, use a small `pop_size` and stop the process manually.

## 1. Local, hardcoded grids — `run.py`

Edit/select a grid function at the bottom of `run.py`, then:

```bash
python run.py                  # parallel via multiprocessing.Pool
python run.py single_thread    # serial (one run at a time)
```

## 2. Local, from the registry JSON

First build the registry (`hpc/registry.json`) as described in
[configuration.md](configuration.md). Then either run the whole registry in a pool:

```bash
python run_in_IDE.py           # runs all registry entries in a multiprocessing.Pool
```

or run a single entry by its 0-based id:

```bash
python run_single_instance.py --expe_id=<N> --json_file=hpc/registry.json
```

`--json_file` defaults to `hpc/registry.json` when omitted.

### Resume

`run_single_instance.py` is **resume-capable**. `process_experiment` reloads the rolling
`algo.pkl` (falling back to `algo_backup.pkl`) and continues. Because `get_res` runs to the 24 h
budget and the accumulated `elapsed_time` is pickled, a run whose budget is already spent resumes
to a fast no-op — so **re-running the registry is safe** (finished runs do nothing).

## 3. HPC / Snellius — SLURM + HyperQueue

The `hpc/` folder runs the registry on Snellius: SLURM allocates whole nodes, HyperQueue (HQ)
dispatches a flat array of single-threaded tasks (one task = one registry entry), with resume
across resubmissions.

```bash
python hpc/generate_registry.py     # build hpc/registry.json (edit the grid first)
python hpc/count_experiments.py     # -> N and the hq submit --array hint
# then submit the HQ array + SLURM workers (full steps in the manual)
```

Full procedure: **[../hpc/snellius_manual.md](../hpc/snellius_manual.md)**.
Computational-budget rationale (30 seeds, single-threaded, 24 h, node sizing):
**[../hpc/experiment_setup.md](../hpc/experiment_setup.md)**.

## 4. Standalone dispatcher — `main.py`

`python main.py --expe_id=<N>` indexes into a cartesian grid hardcoded in `main.py`. This is a
simpler legacy dispatcher; prefer the registry path (2/3) for real sweeps.

---

After a run, see [results.md](results.md) for where output is written and how to analyze it.
