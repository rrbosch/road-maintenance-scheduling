"""Print how many experiments are queued and the matching HQ array range.

    python hpc/count_experiments.py [registry.json]

Defaults to hpc/registry.json (built by hpc/generate_registry.py).
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


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
        print(f"  -> hq submit --array 0-{n - 1} --cpus=1 --pin taskset hpc/hq_task.sh")
    else:
        print("  (nothing to run — regenerate with: python hpc/generate_registry.py)")


if __name__ == "__main__":
    main()
