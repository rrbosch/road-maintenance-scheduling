Experiment registries (hpc/) — road-renovation scheduling campaign
==================================================================

Each registry_e<N>.json is a JSON list of {experiment_name, parameters} entries — one entry per
RUN (algorithm config x random seed). They are built by generate_registry.py (edit it and run
`python hpc/generate_registry.py` to rebuild all four). Dispatch a registry with the native SLURM
job array; see snellius_manual.md / submit_array.sh. Algorithm shorthand <evaluator>|<surrogate>|
<quantile>: EP = LowerBoundEvaluator (elimination-pruning PLBE), LE = ApproximateEvaluator
(lazy-eval PLBE), S = StandardEvaluator (plain NSGA-II control), SS = ScheduleSurrogateEvaluator
(out-of-the-box SAEA baseline); X = XGBoost quantile lower bound, H = SubsetMaxRegressor bound.


registry_e1.json  Experiment 1 — SF-12 (Sioux Falls 12), the small ENUMERABLE instance.
                  Full down-select grid: EP/LE x {X@{0.05,0.1,0.2,0.5}, H} + S + SS
                  = 12 configs x 30 seeds = 360 runs. The reported PLBE arm (EP|X|0.05)
                  additionally logs false-pruning samples (false_pruning_log_prob=0.005), so one
                  campaign yields BOTH the comparison AND the false-pruning rate. Ground truth =
                  the exact Pareto front from the branch-and-bound solver (Src/Algorithms).
                  Analyse: analysis/run_analysis.py + analysis/false_pruning.py.

registry_e2.json  Experiment 2 — SF-76 (Sioux Falls Expanded), the headline instance.
                  EP/LE x q in {0.05,0.2,0.5} + S + SS = 8 configs x 30 seeds = 240 runs.
                  EP|X|0.05 arm logs false-pruning (false_pruning_log_prob=0.01). Establishes the
                  headline efficiency result, EP-vs-LE, and the SAEA comparison at scale.
                  Analyse: analysis/analyze_variants.py + analysis/false_pruning.py.

registry_e3.json  Experiment 3 — SF-76 variants (robustness across network structure/params).
                  Headline trio (EP|X|0.05 + S + SS) across 7 variant networks — road capacity
                  0.9/1.1/100, construction capacity 0.7/100, Less/More Connected — x 30 seeds
                  = 3 x 7 x 30 = 630 runs. Analyse: analysis/analyze_variants.py.

registry_e4.json  Experiment 4 — Anaheim-80 (transfer to a large realistic network, 914 links).
                  Headline trio x 30 seeds = 90 runs. Shows PLBE's advantage grows with network
                  size. Analyse: analysis/analyze_variants.py.

registry.json     ACTIVE DISPATCH SLOT — not an experiment. submit_array.sh reads hpc/registry.json,
                  so copy the registry you want to run into it first:
                      cp hpc/registry_e3.json hpc/registry.json
                      python hpc/count_experiments.py          # prints N + the --array range
                      sbatch --array=0-<N> hpc/submit_array.sh  # e.g. e3 -> --array=0-39

Rebuild every registry from the definitions:  python hpc/generate_registry.py
(registry_combinatorial.json is an unrelated helper for full cartesian-product sweeps.)
