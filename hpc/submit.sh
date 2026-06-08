#!/bin/bash
# SLURM job script — allocates a node and starts a HyperQueue worker that pulls tasks.
# Submit with: sbatch hpc/submit.sh
#
# Full procedure (run from a Snellius login node after `ssh`; details in hpc/snellius_manual.md):
#
#   cd ~/python_restructured/
#
#   # 1. Load modules and start the HQ server (skip if already running)
#   module load 2023 && module load HyperQueue/0.19.0
#   hq server info || nohup hq server start &
#
#   # 2. (Re)generate the registry, then count it
#   python hpc/generate_registry.py         # edit the grid at the bottom of generate_registry.py first
#   python hpc/count_experiments.py         # prints N and the --array hint
#
#   # 3. Submit the task array to HQ (update N to match step 2; runs are single-threaded → --cpus=1)
#   hq submit --array 0-{N-1} --cpus=1 --pin taskset hpc/hq_task.sh
#
#   # 4. Submit this SLURM job to spawn workers (size --nodes to ceil(N / cores-per-node))
#   sbatch hpc/submit.sh
#
#   # 5. Monitor
#   hq job list
#   hq job progress <job_id>
#   hq task list <job_id> | grep FAILED | wc -l
#
#   # Cleanup
#   hq job cancel all       # cancel pending/running tasks
#   hq server stop          # shut down when done

#SBATCH --job-name=road_sched
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=hpc/logs/slurm_%j.out
#SBATCH --error=hpc/logs/slurm_%j.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=robbert.bosch@pm.me

module load 2023
module load HyperQueue/0.19.0

mkdir -p ~/python_restructured/hpc/logs

# Each road-scheduling run is SINGLE-threaded (NSGA-II inner loop is serial), so HQ tasks reserve
# --cpus=1 and a 128-core node packs up to 128 concurrent runs. Size --nodes above to
# ceil(num_experiments / 128) so every run starts at t=0 and gets the full 24 h budget.
hq worker start &
wait
