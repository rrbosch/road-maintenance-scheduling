#!/bin/bash
# Native SLURM job-array dispatch — no HyperQueue, no daemon, disconnect-proof.
#
# Each array element is one 16-core slot (the Snellius minimum billed unit, 1/8 of a 128-core
# rome node) that runs BUNDLE=16 single-threaded experiments concurrently — the same packing the
# old HQ worker did, but driven by this script instead of a login-node server. Because there is no
# server/worker daemon, closing your SSH session or the app cannot orphan a running task (that is
# what killed an earlier HQ run); SLURM owns the whole lifecycle.
#
# Why bundle (vs. one experiment per array element):
#   * Billing: a 1-core element would still be billed the 16-core slot floor -> 16x overcharge.
#     16 experiments per 16-core slot = 100% core use, no waste, and the default memory tier
#     (~28 GiB/slot) stays below the threshold that silently tips billing to the next tier, so do
#     NOT set --mem-per-cpu (each run needs ~100-150 MB).
#   * Concurrency: 360 runs / 16 = 23 elements, well under the 128-job/user QOS cap, so every run
#     starts at t=0. A naive 360-element array would only run 128 at a time.
#
# Usage (from a login node, code + venv already in place; see hpc/snellius_manual.md):
#   cd ~/python_restructured/
#   python hpc/count_experiments.py                 # -> N (and the old HQ hint, ignore it)
#   sbatch --array=0-$(( (N + 15) / 16 - 1 )) hpc/submit_array.sh
#   # e.g. N=360 -> sbatch --array=0-22 hpc/submit_array.sh
#   squeue -u $USER                                 # watch it; survives disconnects
#
# Resume: just resubmit the same array. Each experiment's process_experiment reloads its rolling
# algo.pkl and continues; a run whose 24 h TIME_BUDGET is already spent resumes to a fast no-op.
# To rerun only stragglers, submit a subset, e.g.  sbatch --array=3,7,12 hpc/submit_array.sh

#SBATCH --job-name=road_sched
#SBATCH --array=0-22                 # ceil(N/16)-1; override on the CLI (sbatch --array=...)
#SBATCH --cpus-per-task=16           # 16 = minimum shareable/billed slot; fully used by BUNDLE runs
#SBATCH --ntasks=1
#SBATCH --time=25:00:00              # NSGA-II's 24 h TIME_BUDGET + 1 h headroom for the final flush
#SBATCH --partition=rome
#SBATCH --output=hpc/logs/slurm_%A_%a.out
#SBATCH --error=hpc/logs/slurm_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=robbert.bosch@pm.me
# NB: NO --mem-per-cpu — the default keeps billing at the 16-core slot.

BUNDLE=16

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source ~/.local/venv_road/bin/activate

cd ~/python_restructured/
mkdir -p hpc/logs

# Each road-scheduling run is single-threaded, but XGBoost/numba default to grabbing every core.
# Cap their thread pools AND pin each experiment to its own core with taskset so 16 co-located runs
# don't fight over the slot (mirrors the old `hq submit --cpus=1 --pin taskset`).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1

N=$(python hpc/count_experiments.py | awk 'NR==1{print $1}')
START=$(( SLURM_ARRAY_TASK_ID * BUNDLE ))

echo "[array ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}] running experiments ${START}..$(( START + BUNDLE - 1 )) of ${N}"

for k in $(seq 0 $(( BUNDLE - 1 ))); do
  EID=$(( START + k ))
  [ "$EID" -ge "$N" ] && break
  taskset -c "$k" python hpc/run_task.py --expe_id="$EID" --json_file=hpc/registry.json &
done
wait
