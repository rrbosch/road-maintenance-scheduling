#!/bin/bash
# SLURM job — post-hoc false-pruning replay (analysis/false_pruning.py), parallel over runs.
#
# Unlike hpc/submit.sh (the 24 h GA campaign via HyperQueue), this is a single multi-core job:
# analysis/false_pruning.py replays the 30 independent SF-76 runs across --workers processes.
#
# Procedure (from a Snellius login node; see hpc/snellius_manual.md for the module/venv setup):
#
#   # 1. Copy the completed E2 false-pruning run dirs to Snellius, OFF the OneDrive tree. Each run
#   #    dir must contain config.json + pruned_sample.csv + pruned_sample_fronts.csv, e.g.:
#   #      ~/python_restructured/Experiments/E2_fp/E2 SF-76 false-pruning/<...>/seed*/...
#   #    (rsync from the workstation; the get_res loop I/O-stalls on OneDrive, so don't run there.)
#   # 2. sbatch hpc/submit_false_pruning.sh
#   # 3. Watch hpc/logs/fp_<jobid>.out for per-run progress + the final pooled rate / 95% bound.
#   #    Output lands in analysis/output/E2 SF-76 false-pruning/ (false_pruning.csv + _summary.txt);
#   #    per-run checkpoints under .../partial/ let a re-submitted job resume where it stopped.

#SBATCH --job-name=fp_replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=hpc/logs/fp_%j.out
#SBATCH --error=hpc/logs/fp_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=robbert.bosch@pm.me

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source ~/.local/venv_road/bin/activate

cd ~/python_restructured/
mkdir -p hpc/logs

# One worker per replayed run; each runs numba (Dijkstra) + XGBoost single-threaded so 16 processes
# don't oversubscribe the 16 cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# ~2000 samples/run over 30 runs => pooled n~=60k => 95% one-sided upper bound ~5e-5 (k=0).
# Bump --workers to 30 to finish in one round if the partition/budget allows (same core-hours,
# ~half the wall-clock, but Snellius bills a 16-CPU minimum here).
python analysis/false_pruning.py "Experiments/E2_fp/E2 SF-76 false-pruning" \
    --workers "${SLURM_CPUS_PER_TASK}" --max-samples 2000
