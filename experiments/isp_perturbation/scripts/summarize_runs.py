#!/usr/bin/env python3
"""Summarize arbitrary LIBERO ISP-perturbation episode ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_episodes(path: Path) -> dict[tuple[object, object], bool]:
    episodes: dict[tuple[object, object], bool] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            key = (item["task_id"], item["episode_idx"])
            episodes[key] = bool(item["success"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively summarize episodes.jsonl files under an ISP experiment root."
    )
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    rows: list[tuple[str, int, int, bool]] = []
    for ledger in sorted(root.rglob("episodes.jsonl")):
        relative = ledger.parent.relative_to(root)
        if "control" in relative.parts or "libero_datasets" in relative.parts:
            continue
        episodes = read_episodes(ledger)
        rows.append(
            (
                relative.as_posix(),
                sum(episodes.values()),
                len(episodes),
                (ledger.parent / "job.done").is_file(),
            )
        )

    print(f"{'RUN':80} {'SUCCESS':>8} {'TOTAL':>8} {'SR':>9} {'DONE':>5}")
    total_success = total_episodes = 0
    for name, success, total, done in rows:
        rate = 100.0 * success / total if total else 0.0
        print(f"{name:80} {success:8d} {total:8d} {rate:8.2f}% {str(done):>5}")
        total_success += success
        total_episodes += total
    total_rate = 100.0 * total_success / total_episodes if total_episodes else 0.0
    print(f"\nALL {total_success}/{total_episodes} {total_rate:.2f}%")


if __name__ == "__main__":
    main()
