#!/usr/bin/env python3
"""Build the strict successful-episode manifest for RoboTwin2 replay."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


VIEWS = ("head_camera", "left_camera", "right_camera")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def is_strict_success(record: dict[str, Any], replay_path: Path) -> bool:
    metrics = record.get("metrics", {})
    views = metrics.get("views", {})
    return (
        record.get("status") == "completed"
        and record.get("success") is True
        and replay_path.is_file()
        and metrics.get("all_view_frame_counts_match") is True
        and metrics.get("psnr", {}).get("mean") is not None
        and metrics.get("ssim", {}).get("mean") is not None
        and all(
            views.get(view, {}).get("frame_count_match") is True
            and int(views.get(view, {}).get("paired_frames", 0)) > 0
            for view in VIEWS
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    rows: list[dict[str, Any]] = []
    task_episodes: dict[str, list[int]] = {}
    for task_root in sorted(path for path in result_root.iterdir() if path.is_dir() and path.name != "logs"):
        selected: list[int] = []
        for record_path in sorted((task_root / "episodes").glob("episode*.json")):
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
                episode = int(record["episode"])
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
            replay_path = task_root / "replay_data" / f"episode{episode}.hdf5"
            if not is_strict_success(record, replay_path):
                continue
            metrics = record["metrics"]
            selected.append(episode)
            rows.append(
                {
                    "task": record["task"],
                    "episode": episode,
                    "seed": int(record["seed"]),
                    "source_hdf5": record["source_hdf5"],
                    "source_trajectory": record["source_trajectory"],
                    "replay_hdf5": str(replay_path),
                    "original_image_key": record["original_image_key"],
                    "replay_image_key": record["replay_image_key"],
                    "mean_psnr": float(metrics["psnr"]["mean"]),
                    "mean_ssim": float(metrics["ssim"]["mean"]),
                    "head_frames": int(metrics["views"]["head_camera"]["paired_frames"]),
                    "left_frames": int(metrics["views"]["left_camera"]["paired_frames"]),
                    "right_frames": int(metrics["views"]["right_camera"]["paired_frames"]),
                    "frame_counts_match": True,
                }
            )
        task_episodes[task_root.name] = sorted(selected)

    rows.sort(key=lambda row: (row["task"], row["episode"]))
    manifest = {
        "schema_version": 1,
        "selection": (
            "status=completed AND success=true AND replay_hdf5 exists AND "
            "PSNR/SSIM present AND all three view frame counts match"
        ),
        "successful_episode_count": len(rows),
        "task_count": len(task_episodes),
        "tasks": {
            task: {"count": len(episodes), "episode_ids": episodes}
            for task, episodes in task_episodes.items()
        },
    }
    atomic_text(
        args.output_dir / "successful_episodes.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    atomic_text(
        args.output_dir / "successful_episodes.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )

    csv_path = args.output_dir / "successful_episodes.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    print(f"successful_episodes={len(rows)} tasks={len(task_episodes)} output={args.output_dir}")


if __name__ == "__main__":
    main()
