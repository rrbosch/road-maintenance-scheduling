"""Run a single experiment from the registry by id, with resume support.

``process_experiment(expe_id, json_file)`` loads entry ``expe_id`` from the registry
(``hpc/registry.json`` by default), builds its ``Config``, and either resumes from the rolling
``algo.pkl`` (falling back to ``algo_backup.pkl``) or starts fresh. This is the entry point used by
the HPC dispatch (``hpc/run_task.py``) and by ``run_in_IDE.py``. CLI: ``--expe_id``, ``--json_file``.
"""
import argparse
import json
import logging
import os
import pickle
import re
import traceback
import warnings
from os import path

from Src.config2 import Config

warnings.filterwarnings("ignore", category=RuntimeWarning)
EXPERIMENT_LIST_OVERRIDE = None  # set to a path to override the default registry
# Default experiment registry (built by hpc/generate_registry.py), anchored to the repo root.
DEFAULT_REGISTRY = str(path.join(path.dirname(path.abspath(__file__)), 'hpc', 'registry.json'))

def start_run(config):
    """Simulate starting a new experiment."""
    print(f"Starting new run for: {config}")
    env, algo = config.initialize()
    algo.get_res(env)


def resume_run(config):
    """Simulate resuming an experiment from a saved state. Returns True if you have to start from scratch, if False automatically resumes."""
    pattern = re.compile(r"algo_(\d+)\.pkl")
    success = False

    # Find all files matching 'algo_<i>.pkl' and extract their indices
    indexed_files = []
    for filename in os.listdir(config.results_dir):
        match = pattern.match(filename)
        if match:
            indexed_files.append(int(match.group(1)))  # Extract numeric index

    # Sort in descending order to start from the highest index
    indexed_files.sort(reverse=True)
    indexed_files = [f'algo_{i}.pkl' for i in indexed_files]
    if not indexed_files:
        old_pickle = os.path.join(config.results_dir, 'algo.pkl')
        if os.path.exists(old_pickle):
            indexed_files.append('algo.pkl')
    if not indexed_files:
        return True

    # Try loading files in descending order
    for ifile in indexed_files:
        filepath = os.path.join(config.results_dir, ifile)
        try:
            with open(filepath, "rb") as file:
                state = pickle.load(file)
            print(f"Successfully loaded: {filepath}")
            success = True
            break
        except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
            print(f"Failed to load {filepath}: {e}, trying the next one...")
    if not success:
        print("All available pickle objects were corrupted, starting from fresh.")
        return True
    print(f"Resuming experiment with state: {state}")
    state.resume()
    return False  # do not start from scratch


def process_experiment(expe_id, json_file):
    """
    Loads an experiment from the registry (hpc/registry.json by default), checks if it's already
    run, and executes it.
    """
    if json_file is None:
        json_file = DEFAULT_REGISTRY

    # If no experiments to process, return
    if not os.path.exists(json_file):
        print("No to-do experiments found.")
        return

    # Load experiments
    with open(json_file, "r") as f:
        try:
            experiments = json.load(f)
        except json.JSONDecodeError:
            print("Error reading JSON file.")
            return

    if not experiments:
        print("No experiments left to run.")
        return

    # Take the first experiment
    experiment = experiments[expe_id]
    config = experiment['parameters']
    config['experiment_name'] = experiment['experiment_name']
    config = Config(config)
    LOG_FILE = os.path.join(config.results_dir, "output_log.txt")
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    # Check if already finished
    try:
        # Check for saved progress
        from_scratch = resume_run(config)
        if from_scratch:
            start_run(config)
    except Exception as e:
        error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        raise e


    # Remove the experiment from the list
    # experiments.pop(0)

    # Save the updated list back to the JSON file
    """
    with open(json_file, "w") as f:
        json.dump(experiments, f, indent=4)
    print(f"Experiment processed and removed: {experiment}")
    """


def load_latest_valid_pickle(directory):
    """Load algo.pkl, falling back to algo_backup.pkl if corrupted."""
    primary_path = path.join(directory, "algo.pkl")
    backup_path = path.join(directory, "algo_backup.pkl")

    # Try primary first
    if path.exists(primary_path):
        try:
            with open(primary_path, "rb") as f:
                data = pickle.load(f)
            print(f"Successfully loaded: {primary_path}")
            return data
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Failed to load {primary_path}: {e}, trying backup...")

    # Try backup
    if path.exists(backup_path):
        try:
            with open(backup_path, "rb") as f:
                data = pickle.load(f)
            print(f"Successfully loaded: {backup_path}")
            return data
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Failed to load {backup_path}: {e}")

    # No valid files found
    print("No valid pickle files found, starting fresh")
    return None

if __name__ == "__main__":
    if EXPERIMENT_LIST_OVERRIDE is not None:
        print("EXPERIMENT LIST FILE OVERWRITTEN IN run_single_instance.py")
    # Define argument: expe_id (integer)
    parser = argparse.ArgumentParser(description="Run experiment with SLURM array.")
    parser.add_argument(
        "--expe_id", type=int, help="Experiment ID (used for SLURM array jobs)",)
    parser.add_argument(
        "--json_file", type=str, default=None, help="Path to JSON experiment file")
    # Parse arguments
    args = parser.parse_args()
    json_file = args.json_file if args.json_file else EXPERIMENT_LIST_OVERRIDE
    process_experiment(args.expe_id, json_file)
