#!/usr/bin/env python3
"""Aggregate per-task RoboTwin2 clean training replay summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--expected-tasks", type=int, default=13)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    args = parser.parse_args()
    summaries = []
    for path in sorted(args.result_root.glob("*/task_summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    fields = [
        "task", "episodes_requested", "episodes_completed", "success_count", "failure_count",
        "success_rate", "complete", "mean_episode_psnr", "mean_episode_ssim",
        "original_image_key", "replay_image_key", "replay_mode",
    ]
    with (args.result_root / "task_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows({key: summary.get(key) for key in fields} for summary in summaries)
    completed = sum(int(summary.get("episodes_completed", 0)) for summary in summaries)
    successes = sum(int(summary.get("success_count", 0)) for summary in summaries)
    episode_psnr = []
    episode_ssim = []
    for path in sorted(args.result_root.glob("*/episodes/episode*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            continue
        psnr = record.get("metrics", {}).get("psnr", {}).get("mean")
        ssim = record.get("metrics", {}).get("ssim", {}).get("mean")
        if psnr is not None:
            episode_psnr.append(float(psnr))
        if ssim is not None:
            episode_ssim.append(float(ssim))
    expected = args.expected_tasks * args.episodes_per_task
    overall = {
        "tasks_found": len(summaries),
        "tasks_expected": args.expected_tasks,
        "episodes_completed": completed,
        "episodes_expected": expected,
        "success_count": successes,
        "failure_count": completed - successes,
        "success_rate": successes / completed if completed else 0.0,
        "mean_episode_psnr": float(np.mean(episode_psnr)) if episode_psnr else None,
        "mean_episode_ssim": float(np.mean(episode_ssim)) if episode_ssim else None,
        "complete": len(summaries) == args.expected_tasks and completed == expected,
        "tasks_incomplete": [summary["task"] for summary in summaries if not summary.get("complete")],
    }
    (args.result_root / "overall_summary.json").write_text(
        json.dumps(overall, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(overall, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
