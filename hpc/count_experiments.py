"""Print how many experiments are queued and the matching array range.

    python hpc/count_experiments.py [registry.json]

Defaults to hpc/registry.json (built by hpc/generate_registry.py).
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# Must match BUNDLE in hpc/submit_array.sh: the native array packs this many single-threaded
# experiments into each 16-core array element, so the array range is ceil(N/BUNDLE)-1, NOT N-1.
BUNDLE = 16


def main() -> None:
    jf = sys.argv[1] if len(sys.argv) > 1 else "hpc/registry.json"
    p = Path(jf)
    if not p.is_absolute():
        p = _ROOT / p
    if not p.exists():
        print(f"registry file not found: {p}")
        sys.exit(1)
    n = len(json.load(open(p)))
    print(f"{n} experiments")
    if n:
        last = (n + BUNDLE - 1) // BUNDLE - 1   # ceil(n/BUNDLE) - 1
        print(f"  native SLURM array ({BUNDLE} experiments bundled per 16-core element) -- USE THIS:")
        print(f"    sbatch --array=0-{last} hpc/submit_array.sh")
        print(f"  legacy HyperQueue (1 task/element, 1 cpu):")
        print(f"    hq submit --array 0-{n - 1} --cpus=1 --pin taskset hpc/hq_task.sh")
    else:
        print("  (nothing to run — regenerate with: python hpc/generate_registry.py)")


if __name__ == "__main__":
    main()
