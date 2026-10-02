#!/usr/bin/env python
"""Build the frozen RAWVLA-Bench-Light LIBERO manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = ROOT / "experiments" / "simulation" / "libero"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rawvla_bench.manifest import build_libero_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PACKAGE_ROOT / "manifests" / "libero_manifest_50_init_states_10000_rollouts.json",
    )
    parser.add_argument("--benchmark-seed", type=int, default=20260813)
    parser.add_argument("--base-seeds-per-task", type=int, default=50)
    parser.add_argument("--tasks-per-suite", type=int, default=10)
    args = parser.parse_args()

    manifest = build_libero_manifest(
        benchmark_seed=args.benchmark_seed,
        base_seeds_per_task=args.base_seeds_per_task,
        tasks_per_suite=args.tasks_per_suite,
    )
    write_manifest(manifest, args.out)
    print(f"Wrote {len(manifest['entries'])} entries to {args.out}")


if __name__ == "__main__":
    main()
