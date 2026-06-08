# Multi-objective scheduling of road-construction projects

Research code for **multi-objective scheduling of road-construction projects on a traffic
network**. Given a set of construction projects (each closes or cripples certain road links for a
duration), it finds a Pareto-optimal set of schedules trading off two objectives:

- **SL** (`Tardiness`) — expected risk/penalty cost of starting projects late (a binomial-decay
  model).
- **TTD** (`TotalTravelDelay`) — extra network-wide travel time caused by links being under
  construction, computed via a traffic-assignment simulation.

A decision vector `x` holds **one integer start time per project** (`-1` = not planned). The
optimizer is **NSGA-II** (via `pymoo`). The research contribution is an evaluator that uses
ML-predicted **lower bounds** on TTD to skip the expensive traffic simulation for solutions that
are provably dominated (the *progressive lower-bound evaluator*, PLBE).

## Quickstart

```bash
# 1. Create and activate a Python 3.11 virtual environment, then install deps
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

# 2. Run a small local experiment (edit/select a grid function in run.py first)
python run.py single_thread

# 3. Analyze the results
python analysis/run_analysis.py "Experiments/<experiment_name>"
```

## Documentation

| Guide | Contents |
|---|---|
| [docs/setup.md](docs/setup.md) | Python version, virtual environment, dependencies |
| [docs/configuration.md](docs/configuration.md) | `Config` parameters and how to define an experiment grid |
| [docs/running.md](docs/running.md) | The entry points: running locally vs. on Snellius (HPC) |
| [docs/results.md](docs/results.md) | The on-disk results schema and the analysis pipeline |
| [hpc/snellius_manual.md](hpc/snellius_manual.md) | Step-by-step SLURM + HyperQueue run procedure |
| [hpc/experiment_setup.md](hpc/experiment_setup.md) | Computational budget (seeds, threads, walltime, node sizing) |

## Repository layout

```
Src/                  Optimizer: Config, NSGA-II, evaluators, operators, registry, I/O utils
Environments/
  env/                Problem model, objectives, traffic simulation + network (Frank-Wolfe TAP)
  input/<case>/       Case-study CSV inputs (Sioux Falls Expanded, Anaheim, variants)
analysis/             Post-hoc analysis pipeline (reads the results schema, emits figures)
hpc/                  Snellius SLURM + HyperQueue scaffolding + the experiment registry
Experiments/          Generated run output (git-ignored)
run.py                Local runner (hardcoded grids, parallel pool)
run_single_instance.py  Run one registry entry by id (resume-capable)
run_in_IDE.py         Run the whole registry locally in a pool
main.py               Standalone dispatcher over a cartesian grid
```
