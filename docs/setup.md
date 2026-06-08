# Environment setup

## Python version

**Python 3.11** (developed and verified on 3.11.9). The old `README.txt` said 3.10; that is
out of date.

## Virtual environment + dependencies

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
```

Key dependencies: `pymoo` (NSGA-II engine), `numba` (JIT-compiled traffic assignment),
`xgboost` (the TTD lower-bound surrogate), `scikit-learn`, `networkx`, `scipy`, `pandas`,
`matplotlib`.

> **Stale entries in `requirements.txt`:** `SQLAlchemy` and `OpenMatrix` are listed but no longer
> used — the SQLite result cache was removed (overhaul item 1). They are harmless to install but
> can be pruned.

There is **no build step and no packaging** — the code runs in place. (A future overhaul item
will convert it to an installable src-layout package.)

## Verifying the install

A quick check that the deps and imports resolve, runnable from any directory:

```bash
python -c "from Src.config2 import Config; print('imports OK')"
```

To smoke-test the traffic model on one case study, the `__main__` blocks of
`Environments/env/network.py` and `Environments/env/traffic_simulation.py` load a case study and
run a single traffic assignment (with plotting). Note `network.py`'s `__main__` has a hardcoded
case-study path — adjust it before running.

## Notes

- **No automated test suite.** Verification is done by running short experiments and inspecting
  the output (see [results.md](results.md)).
- On Snellius the environment is set up once via the modules + venv described in
  [../hpc/snellius_manual.md](../hpc/snellius_manual.md) (Python/3.11.3, `~/.local/venv_road`).
