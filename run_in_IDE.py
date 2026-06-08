"""Run the whole experiment registry locally in a multiprocessing pool.

Reads ``hpc/registry.json`` and launches every entry as a separate ``run_single_instance.py``
subprocess (one per registry id), ``nr_of_threads`` at a time. Each subprocess is resume-capable,
so re-running this is safe. The local counterpart of the HPC dispatch in ``hpc/``.
"""
import json
import os
import subprocess
import sys
from multiprocessing import Pool


def run_experiment(args):
    """Runs a single experiment by calling run_single_instance.py with the given experiment ID."""
    expe_id, json_path = args
    python_path = sys.executable
    command = [python_path, "run_single_instance.py", f"--expe_id={expe_id}", f"--json_file={json_path}"]
    # Run the command safely
    try:
        subprocess.run(command, check=True, env=os.environ)
    except subprocess.CalledProcessError as e:
        print(f"Error: process failed with exit code {e.returncode}")

def main(nr_of_threads):
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hpc", "registry.json")
    with open(json_path, "r", encoding="utf-8") as f:
        experiment_list = json.load(f)
    nr_of_experiments = len(experiment_list)
    print(f"Starting {nr_of_experiments} experiments with {nr_of_threads} threads.")

    with Pool(nr_of_threads) as p:
        p.map(run_experiment, [(i, json_path) for i in range(nr_of_experiments)], chunksize=1)

if __name__ == "__main__":
    nr_of_threads = 10 # 64GB of RAM, +-4GB per thread
    main(nr_of_threads)
