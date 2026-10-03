#!/usr/bin/env python3
"""Summarize completion markers produced by the RoboTwin evaluation runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def marker_record(root: Path, marker: Path, status: str) -> dict[str, str]:
    relative = marker.relative_to(root)
    parts = relative.parts
    checkpoint = parts[0] if len(parts) > 1 else ""
    mode = parts[1] if len(parts) > 2 else ""
    seed = parts[2] if len(parts) > 3 and parts[2].startswith("seed_") else ""
    return {
        "checkpoint": checkpoint,
        "mode": mode,
        "seed": seed,
        "task": marker.stem,
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()

    records = [
        marker_record(args.result_root, path, status)
        for status, suffix in (("done", "*.DONE"), ("failed", "*.FAILED"))
        for path in sorted(args.result_root.rglob(suffix))
    ]
    summary = {
        "result_root": str(args.result_root),
        "completed": sum(record["status"] == "done" for record in records),
        "failed": sum(record["status"] == "failed" for record in records),
        "records": records,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
