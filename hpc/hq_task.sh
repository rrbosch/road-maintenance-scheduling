#!/bin/bash
# Executed once per HyperQueue task.
# $HQ_TASK_ID is set by HQ to the 0-based task index (matches hpc/registry.json).
#
# Edit the two paths below to match your Snellius setup (see hpc/snellius_manual.md).

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source ~/.local/venv_road/bin/activate

cd ~/python_restructured/

python hpc/run_task.py --expe_id=$HQ_TASK_ID --json_file=hpc/registry.json
