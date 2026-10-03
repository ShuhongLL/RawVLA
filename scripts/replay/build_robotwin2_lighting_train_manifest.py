#!/usr/bin/env python3
"""Freeze one random five-level lighting sample for each successful clean replay."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_ROOT = ROOT / "third_party" / "RoboTwin"
TRAIN_SEED = 20260824
RAW_WHITE_LEVEL = 3.5
RAW_SENSOR_SHOT_NOISE = 2.5e-5
RAW_SENSOR_READ_NOISE = 3.90625e-8


def stable_uint32(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def successful_records(clean_replay_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(clean_replay_root.glob("*/results.jsonl")):
        task = path.parent.name
        replay_dir = path.parent / "replay_data"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            episode = int(record["episode"])
            if (
                record.get("status") == "completed"
                and bool(record.get("success"))
                and (replay_dir / f"episode{episode}.hdf5").is_file()
            ):
                records.append({"task_name": task, **record})
    records.sort(key=lambda item: (str(item["task_name"]), int(item["episode"])))
    return records


def build_manifest(
    *, clean_replay_root: Path, dataset_root: Path, seed: int
) -> dict[str, Any]:
    lighting = runpy.run_path(str(ROBOTWIN_ROOT / "envs" / "lighting.py"))
    domains = tuple(lighting["LIGHTING_DOMAINS"])
    configs = lighting["LIGHTING_DOMAIN_CONFIGS"]
    source_records = successful_records(clean_replay_root)
    target_base, target_remainder = divmod(len(source_records), len(domains))
    target_counts = {
        domain: target_base + int(index < target_remainder)
        for index, domain in enumerate(domains)
    }
    records_by_task: dict[str, list[dict[str, Any]]] = {}
    for source in source_records:
        records_by_task.setdefault(str(source["task_name"]), []).append(source)
    guaranteed_per_domain = sum(len(records) // len(domains) for records in records_by_task.values())
    residual_capacity = {
        domain: target_counts[domain] - guaranteed_per_domain for domain in domains
    }
    assigned_domains: dict[tuple[str, int], str] = {}
    for task, task_records in sorted(records_by_task.items()):
        per_domain, task_remainder = divmod(len(task_records), len(domains))
        rng = np.random.default_rng(stable_uint32("robotwin2_train_domain_balance", seed, task))
        tie_order = list(domains)
        rng.shuffle(tie_order)
        tie_rank = {domain: rank for rank, domain in enumerate(tie_order)}
        residual_domains = sorted(
            domains, key=lambda domain: (-residual_capacity[domain], tie_rank[domain])
        )[:task_remainder]
        for domain in residual_domains:
            residual_capacity[domain] -= 1
        labels = [domain for domain in domains for _ in range(per_domain)] + residual_domains
        rng.shuffle(labels)
        for source, domain in zip(task_records, labels):
            assigned_domains[(task, int(source["episode"]))] = str(domain)
    if any(residual_capacity.values()):
        raise RuntimeError(f"Could not balance domain assignment: {residual_capacity}")

    entries = []
    for index, source in enumerate(source_records):
        task = str(source["task_name"])
        episode = int(source["episode"])
        dataset_seed = int(source["seed"])
        rng = np.random.default_rng(
            stable_uint32("robotwin2_train_ev", seed, task, episode, dataset_seed)
        )
        domain = assigned_domains[(task, episode)]
        sample_unit = float(rng.random())
        lighting_ev, _, _ = configs[domain].at(sample_unit)
        relative_dir = Path(task) / "aloha-agilex_clean_50"
        source_hdf5 = dataset_root / relative_dir / "data" / f"episode{episode}.hdf5"
        source_trajectory = dataset_root / relative_dir / "_traj_data" / f"episode{episode}.pkl"
        if not source_hdf5.is_file() or not source_trajectory.is_file():
            raise FileNotFoundError(f"Missing source data for {task} episode {episode}")
        entries.append(
            {
                "entry_index": index,
                "task_name": task,
                "episode": episode,
                "dataset_seed": dataset_seed,
                "source_hdf5": str(source_hdf5),
                "source_trajectory": str(source_trajectory),
                "clean_replay_record": str(
                    clean_replay_root / task / "episodes" / f"episode{episode}.json"
                ),
                "lighting_domain": domain,
                "lighting_strategy": "environment_only",
                "lighting_sample_unit": sample_unit,
                "lighting_ev": float(lighting_ev),
                "lighting_scale": float(2.0**lighting_ev),
                "raw_white_level": RAW_WHITE_LEVEL,
                "raw_sensor_noise_enabled": True,
                "raw_sensor_shot_noise": RAW_SENSOR_SHOT_NOISE,
                "raw_sensor_read_noise": RAW_SENSOR_READ_NOISE,
                "noise_seed": stable_uint32(
                    "robotwin2_train_raw_noise", seed, task, episode, dataset_seed, domain
                ),
            }
        )

    counts = Counter(entry["lighting_domain"] for entry in entries)
    tasks = Counter(entry["task_name"] for entry in entries)
    return {
        "benchmark": "rawvla-bench-light-robotwin2-v1-train-state-copy",
        "version": 1,
        "seed": seed,
        "source_protocol": "clean_rgb_action_replay_success_only",
        "paired_raw_protocol": "copy_clean_simulator_state_then_render_without_action",
        "lighting_domains": list(domains),
        "lighting_strategy": "environment_only",
        "lighting_distribution": "task-stratified_balanced_random_domain_then_uniform_ev",
        "ev_ranges": {domain: list(configs[domain].ev_range) for domain in domains},
        "raw_sensor": {
            "representation": "three_channel_linear_float32",
            "range": [0.0, 1.0],
            "quantized": False,
            "black_level_corrected": True,
            "white_level": RAW_WHITE_LEVEL,
            "shot_noise": RAW_SENSOR_SHOT_NOISE,
            "read_noise": RAW_SENSOR_READ_NOISE,
        },
        "clean_replay_root": str(clean_replay_root),
        "dataset_root": str(dataset_root),
        "num_entries": len(entries),
        "task_counts": dict(sorted(tasks.items())),
        "domain_counts": {domain: counts[domain] for domain in domains},
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-replay-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    args = parser.parse_args()
    manifest = build_manifest(
        clean_replay_root=args.clean_replay_root.resolve(),
        dataset_root=args.dataset_root.resolve(),
        seed=args.seed,
    )
    if manifest["num_entries"] != 633:
        raise ValueError(f"Expected 633 successful replay episodes, got {manifest['num_entries']}")
    atomic_json(args.output.resolve(), manifest)
    print(json.dumps({key: manifest[key] for key in ("num_entries", "task_counts", "domain_counts")}, indent=2))


if __name__ == "__main__":
    main()
