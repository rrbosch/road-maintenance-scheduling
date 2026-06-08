"""Discover and load runs written in the new results schema.

A *run* is any directory under an experiment tree that contains a ``config.json`` (written by
``Src/Utils/results_io.write_config``). Run identity comes from the JSON, **not** from parsing the
folder path. Each run exposes its config, per-generation ``progress``, long-format ``fronts``, and
``surrogate`` accuracy log as DataFrames.
"""
import json
import os
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Run:
    directory: str
    config: dict
    progress: pd.DataFrame
    fronts: pd.DataFrame
    surrogate: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def params(self) -> dict:
        """The varied experiment parameters (what distinguishes runs within an experiment)."""
        return self.config.get('experiment_params', {}) or {}

    @property
    def seed(self):
        return self.config.get('algo_seed')

    def config_key(self, exclude=('algo_seed',)) -> tuple:
        """Hashable identity of this run's configuration, ignoring the random seed.

        Used to group seeds of the same configuration together for averaging.
        """
        return tuple(sorted((k, str(v)) for k, v in self.params.items() if k not in exclude))

    @property
    def label(self) -> str:
        parts = [f"{k}={v}" for k, v in sorted(self.params.items()) if k != 'algo_seed']
        return ", ".join(parts) if parts else os.path.basename(self.directory)


def _read_csv(p):
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def load_run(directory) -> Run:
    """Load a single run directory (must contain config.json + fronts.csv + progress.csv)."""
    with open(os.path.join(directory, 'config.json')) as f:
        config = json.load(f)
    progress = _read_csv(os.path.join(directory, 'progress.csv'))
    fronts = _read_csv(os.path.join(directory, 'fronts.csv'))
    surrogate = _read_csv(os.path.join(directory, 'surrogate.csv'))
    return Run(directory=directory, config=config, progress=progress, fronts=fronts,
               surrogate=surrogate)


def discover_runs(root) -> list[Run]:
    """Walk ``root`` and load every directory containing a config.json (skips incomplete runs)."""
    runs = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if 'config.json' not in filenames:
            continue
        if 'fronts.csv' not in filenames:
            print(f"skipping {dirpath}: config.json present but no fronts.csv (run not started?)")
            continue
        try:
            runs.append(load_run(dirpath))
        except Exception as e:  # noqa: BLE001 - keep going past one bad run
            print(f"skipping {dirpath}: failed to load ({e})")
    print(f"discovered {len(runs)} run(s) under {root}")
    return runs
