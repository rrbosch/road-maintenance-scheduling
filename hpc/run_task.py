"""HyperQueue entry point — called once per task by hq_task.sh.

Usage:
    python hpc/run_task.py --expe_id=$HQ_TASK_ID [--json_file hpc/registry.json]

Looks up entry[expe_id] in the registry JSON (built by ``hpc/generate_registry.py``) and runs it with
auto-resume: ``run_single_instance.process_experiment`` reloads the rolling ``algo.pkl`` and
continues, or starts fresh. The run itself stops at NSGA-II's 24 h ``TIME_BUDGET``; if SLURM kills
the worker earlier, the next submission resumes from the pickle (a run whose 24 h budget is already
spent resumes to a fast no-op, so resubmitting the whole array is safe).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# make the repo importable regardless of the worker's cwd
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from run_single_instance import process_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one experiment from the HPC queue JSON")
    parser.add_argument("--expe_id", type=int, required=True,
                        help="0-based index into the queue JSON (= $HQ_TASK_ID)")
    parser.add_argument("--json_file", default="hpc/registry.json",
                        help="registry JSON built by hpc/generate_registry.py "
                             "(default: hpc/registry.json, resolved from the repo root)")
    cli = parser.parse_args()

    json_path = Path(cli.json_file)
    if not json_path.is_absolute():
        json_path = _ROOT / json_path
    if not json_path.exists():
        print(f"ERROR: queue file not found: {json_path}")
        sys.exit(1)

    with open(json_path) as f:
        queue = json.load(f)
    n = len(queue)
    if not (0 <= cli.expe_id < n):
        print(f"ERROR: expe_id={cli.expe_id} out of range (queue has {n} entries)")
        sys.exit(1)

    entry = queue[cli.expe_id]
    print(f"[HPC] expe_id={cli.expe_id}/{n - 1}  experiment={entry.get('experiment_name')}  "
          f"params={entry.get('parameters')}", flush=True)

    # process_experiment re-reads the JSON and indexes [expe_id]; it resumes or starts the run.
    process_experiment(cli.expe_id, str(json_path))


if __name__ == "__main__":
    main()
