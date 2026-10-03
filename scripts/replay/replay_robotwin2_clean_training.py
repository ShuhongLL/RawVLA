#!/usr/bin/env python3
"""Replay RoboTwin2 clean training trajectories and measure image fidelity.

This is expert-trajectory replay, not VLA policy evaluation.  Every episode
loads the saved ``left_joint_path``/``right_joint_path``, executes those paths
through RoboTwin's physics/controller stack, saves the newly rendered camera
products, and compares replay ``default_isp`` against the original training
``rgb`` at matching frame indices.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import pickle
import sys
import traceback
from typing import Any

import cv2
import h5py
import numpy as np
from skimage.metrics import structural_similarity
import yaml


VIEWS = ("head_camera", "left_camera", "right_camera")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def decode_hdf_image(value: Any) -> np.ndarray:
    if isinstance(value, (bytes, np.bytes_)):
        encoded = np.frombuffer(value, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("OpenCV could not decode HDF5 image bytes")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray(value)
    if image.ndim == 1 and image.dtype == np.uint8:
        decoded = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("OpenCV could not decode HDF5 uint8 buffer")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape={image.shape}, dtype={image.dtype}")
    return image


def as_float01(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        result = image.astype(np.float32) / np.float32(255.0)
    elif image.dtype == np.float32:
        result = image
    else:
        result = image.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Image contains NaN or infinity")
    if result.size and (float(result.min()) < 0.0 or float(result.max()) > 1.0):
        raise ValueError(
            f"Image must be in [0, 1], got [{float(result.min())}, {float(result.max())}]"
        )
    return result


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(value)]
    return {
        "count": len(values),
        "finite_count": len(finite),
        "mean": float(np.mean(finite)) if finite else None,
        "min": float(np.min(finite)) if finite else None,
        "max": float(np.max(finite)) if finite else None,
    }


def compare_hdf5_images(
    source_path: Path,
    replay_path: Path,
    original_key: str,
    replay_key: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"views": {}}
    all_psnr: list[float] = []
    all_ssim: list[float] = []
    with h5py.File(source_path, "r") as source, h5py.File(replay_path, "r") as replay:
        for view in VIEWS:
            source_ds = source[f"observation/{view}/{original_key}"]
            replay_ds = replay[f"observation/{view}/{replay_key}"]
            paired = min(len(source_ds), len(replay_ds))
            psnr_values: list[float] = []
            ssim_values: list[float] = []
            for index in range(paired):
                original = as_float01(decode_hdf_image(source_ds[index]))
                replayed = as_float01(decode_hdf_image(replay_ds[index]))
                if original.shape != replayed.shape:
                    raise ValueError(
                        f"Image shape mismatch view={view} frame={index}: "
                        f"source={original.shape}, replay={replayed.shape}"
                    )
                mse = float(np.mean(np.square(original - replayed, dtype=np.float32)))
                psnr = math.inf if mse == 0.0 else float(-10.0 * math.log10(mse))
                ssim = float(
                    structural_similarity(
                        original,
                        replayed,
                        data_range=1.0,
                        channel_axis=-1,
                    )
                )
                psnr_values.append(psnr)
                ssim_values.append(ssim)
            all_psnr.extend(psnr_values)
            all_ssim.extend(ssim_values)
            result["views"][view] = {
                "original_frames": len(source_ds),
                "replay_frames": len(replay_ds),
                "paired_frames": paired,
                "frame_count_match": len(source_ds) == len(replay_ds),
                "psnr": summarize_values(psnr_values),
                "ssim": summarize_values(ssim_values),
            }
    result["psnr"] = summarize_values(all_psnr)
    result["ssim"] = summarize_values(all_ssim)
    result["all_view_frame_counts_match"] = all(
        view_result["frame_count_match"] for view_result in result["views"].values()
    )
    return result


def merge_cache_to_hdf5(cache_dir: Path, hdf5_path: Path) -> None:
    from envs.utils.pkl2hdf5 import (
        append_data_to_structure,
        create_hdf5_from_dict,
        load_pkl_file,
        parse_dict_structure,
    )

    indexed = sorted(
        (int(path.stem), path)
        for path in cache_dir.glob("*.pkl")
        if path.stem.isdigit()
    )
    if not indexed:
        raise FileNotFoundError(f"No replay frame PKLs found in {cache_dir}")
    expected = list(range(len(indexed)))
    actual = [index for index, _ in indexed]
    if actual != expected:
        raise ValueError(f"Replay cache is not contiguous: expected={expected}, actual={actual}")
    data = parse_dict_structure(load_pkl_file(str(indexed[0][1])))
    for _, path in indexed:
        append_data_to_structure(data, load_pkl_file(str(path)))
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = hdf5_path.with_name(f".{hdf5_path.name}.tmp.{os.getpid()}")
    with h5py.File(temporary, "w") as output:
        create_hdf5_from_dict(output, data)
    os.replace(temporary, hdf5_path)


def build_robotwin_args(robotwin_root: Path, output_task_root: Path, task: str) -> dict[str, Any]:
    from envs import CONFIGS_PATH
    from script.collect_data import get_embodiment_config

    with (robotwin_root / "task_config" / "demo_clean.yml").open("r", encoding="utf-8") as stream:
        args = yaml.safe_load(stream)
    with Path(CONFIGS_PATH, "_embodiment_config.yml").open("r", encoding="utf-8") as stream:
        embodiment_types = yaml.safe_load(stream)
    embodiment = args["embodiment"]
    if len(embodiment) != 1:
        raise ValueError(f"Expected one shared embodiment, got {embodiment}")
    robot_file = embodiment_types[embodiment[0]]["file_path"]
    args.update(
        {
            "task_name": task,
            "task_config": "demo_clean",
            "save_path": str(output_task_root),
            "left_robot_file": robot_file,
            "right_robot_file": robot_file,
            "dual_arm_embodied": True,
            "left_embodiment_config": get_embodiment_config(robot_file),
            "right_embodiment_config": get_embodiment_config(robot_file),
            "embodiment_name": str(embodiment[0]),
            "need_plan": False,
            "render_freq": 0,
            "save_data": True,
            "eval_mode": False,
            "eval_video_log": False,
        }
    )
    return args


def load_completed_records(record_dir: Path, replay_data_dir: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for path in sorted(record_dir.glob("episode*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            episode = int(record["episode"])
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
        replay_path = replay_data_dir / f"episode{episode}.hdf5"
        if record.get("status") == "completed" and replay_path.is_file():
            records[episode] = record
    return records


def write_task_outputs(task_root: Path, task: str, requested: int) -> dict[str, Any]:
    records = []
    for path in sorted((task_root / "episodes").glob("episode*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    records.sort(key=lambda record: int(record["episode"]))
    fields = [
        "task", "episode", "seed", "status", "success", "failure", "source_hdf5",
        "source_trajectory", "replay_hdf5", "original_image_key", "replay_image_key",
        "mean_psnr", "mean_ssim", "frame_counts_match",
    ]
    csv_path = task_root / "results.csv"
    csv_temporary = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with csv_temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metrics = record.get("metrics", {})
            writer.writerow(
                {
                    "task": record.get("task"),
                    "episode": record.get("episode"),
                    "seed": record.get("seed"),
                    "status": record.get("status"),
                    "success": record.get("success"),
                    "failure": record.get("failure", ""),
                    "source_hdf5": record.get("source_hdf5"),
                    "source_trajectory": record.get("source_trajectory"),
                    "replay_hdf5": record.get("replay_hdf5"),
                    "original_image_key": record.get("original_image_key"),
                    "replay_image_key": record.get("replay_image_key"),
                    "mean_psnr": metrics.get("psnr", {}).get("mean"),
                    "mean_ssim": metrics.get("ssim", {}).get("mean"),
                    "frame_counts_match": metrics.get("all_view_frame_counts_match"),
                }
            )
    os.replace(csv_temporary, csv_path)
    atomic_text(
        task_root / "results.jsonl",
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
    )
    completed = [record for record in records if record.get("status") == "completed"]
    successes = sum(bool(record.get("success")) for record in completed)
    psnr = [record["metrics"]["psnr"]["mean"] for record in completed if record["metrics"]["psnr"]["mean"] is not None]
    ssim = [record["metrics"]["ssim"]["mean"] for record in completed if record["metrics"]["ssim"]["mean"] is not None]
    summary = {
        "task": task,
        "episodes_requested": requested,
        "episodes_completed": len(completed),
        "episodes_missing": sorted(set(range(requested)) - {int(record["episode"]) for record in completed}),
        "success_count": successes,
        "failure_count": len(completed) - successes,
        "success_rate": successes / len(completed) if completed else 0.0,
        "complete": len(completed) == requested,
        "mean_episode_psnr": float(np.mean(psnr)) if psnr else None,
        "mean_episode_ssim": float(np.mean(ssim)) if ssim else None,
        "original_image_key": "rgb",
        "replay_image_key": "default_isp",
        "replay_mode": "clean_training_trajectory_closed_loop_physics",
    }
    atomic_json(task_root / "task_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--episode-indices",
        default="",
        help="Comma-separated episode indices for this shard (default: all episodes).",
    )
    parser.add_argument("--original-image-key", default="rgb")
    parser.add_argument("--replay-image-key", default="default_isp")
    parser.add_argument("--clear-cache-every", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    robotwin_root = cli.robotwin_root.resolve()
    dataset_root = cli.dataset_root.resolve()
    output_root = cli.output_root.resolve()
    if cli.episodes <= 0:
        raise ValueError("--episodes must be positive")
    selected_episodes = (
        [int(value) for value in cli.episode_indices.split(",") if value.strip()]
        if cli.episode_indices
        else list(range(cli.episodes))
    )
    if not selected_episodes:
        raise ValueError("--episode-indices must select at least one episode")
    if len(selected_episodes) != len(set(selected_episodes)):
        raise ValueError("--episode-indices contains duplicates")
    if any(episode < 0 or episode >= cli.episodes for episode in selected_episodes):
        raise ValueError(f"--episode-indices must be within [0, {cli.episodes})")
    os.chdir(robotwin_root)
    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(robotwin_root / "script"))
    os.environ.setdefault("ROBOTWIN_ISP_AXIS", "none")

    from sapien.render import clear_cache as sapien_clear_cache
    from script.collect_data import class_decorator

    source_task_root = dataset_root / cli.task / "aloha-agilex_clean_50"
    output_task_root = output_root / cli.task
    record_dir = output_task_root / "episodes"
    replay_data_dir = output_task_root / "replay_data"
    error_dir = output_task_root / "errors"
    for directory in (record_dir, replay_data_dir, error_dir):
        directory.mkdir(parents=True, exist_ok=True)

    seed_path = source_task_root / "seed.txt"
    seeds = [int(value) for value in seed_path.read_text(encoding="utf-8").split()]
    if len(seeds) < cli.episodes:
        raise ValueError(f"{seed_path} contains {len(seeds)} seeds, need {cli.episodes}")
    completed = load_completed_records(record_dir, replay_data_dir)
    print(
        f"task={cli.task} resume_completed={len(completed)}/{cli.episodes} "
        f"shard_episodes={selected_episodes}",
        flush=True,
    )
    args = build_robotwin_args(robotwin_root, output_task_root, cli.task)
    task_env = class_decorator(cli.task)
    errors: list[int] = []

    for episode in selected_episodes:
        if episode in completed:
            print(f"skip completed episode={episode}", flush=True)
            continue
        source_hdf5 = source_task_root / "data" / f"episode{episode}.hdf5"
        source_trajectory = source_task_root / "_traj_data" / f"episode{episode}.pkl"
        replay_hdf5 = replay_data_dir / f"episode{episode}.hdf5"
        if not source_hdf5.is_file() or not source_trajectory.is_file():
            raise FileNotFoundError(f"Missing source episode files for task={cli.task} episode={episode}")
        with source_trajectory.open("rb") as stream:
            trajectory = pickle.load(stream)
        episode_args = dict(args)
        episode_args.update(
            {
                "now_ep_num": episode,
                "seed": seeds[episode],
                "left_joint_path": trajectory["left_joint_path"],
                "right_joint_path": trajectory["right_joint_path"],
            }
        )
        last_error = ""
        for attempt in range(1, cli.max_attempts + 1):
            env_open = False
            try:
                print(
                    f"start task={cli.task} episode={episode} seed={seeds[episode]} attempt={attempt}",
                    flush=True,
                )
                task_env.setup_demo(**episode_args)
                env_open = True
                task_env.set_path_lst(episode_args)
                task_env.play_once()
                success = bool(task_env.plan_success and task_env.check_success())
                cache_dir = Path(task_env.folder_path["cache"])
                task_env.close_env(
                    clear_cache=((episode + 1) % max(cli.clear_cache_every, 1) == 0)
                )
                env_open = False
                merge_cache_to_hdf5(cache_dir, replay_hdf5)
                task_env.remove_data_cache()
                metrics = compare_hdf5_images(
                    source_hdf5,
                    replay_hdf5,
                    original_key=cli.original_image_key,
                    replay_key=cli.replay_image_key,
                )
                record = {
                    "task": cli.task,
                    "episode": episode,
                    "seed": seeds[episode],
                    "status": "completed",
                    "success": success,
                    "failure": "" if success else "task_check_success_false",
                    "attempt": attempt,
                    "source_hdf5": str(source_hdf5),
                    "source_trajectory": str(source_trajectory),
                    "replay_hdf5": str(replay_hdf5),
                    "original_image_key": cli.original_image_key,
                    "replay_image_key": cli.replay_image_key,
                    "replay_mode": "clean_training_trajectory_closed_loop_physics",
                    "metrics": metrics,
                }
                atomic_json(record_dir / f"episode{episode}.json", record)
                print(json.dumps(record, sort_keys=True), flush=True)
                last_error = ""
                break
            except Exception:
                last_error = traceback.format_exc()
                print(last_error, flush=True)
                if env_open:
                    try:
                        task_env.close_env(clear_cache=True)
                    except Exception:
                        pass
                try:
                    sapien_clear_cache()
                except Exception:
                    pass
        if last_error:
            errors.append(episode)
            (error_dir / f"episode{episode}.log").write_text(last_error, encoding="utf-8")
        write_task_outputs(output_task_root, cli.task, cli.episodes)

    summary = write_task_outputs(output_task_root, cli.task, cli.episodes)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    completed = load_completed_records(record_dir, replay_data_dir)
    shard_missing = sorted(set(selected_episodes) - set(completed))
    if shard_missing:
        print(f"shard_missing={shard_missing}", flush=True)
    return 1 if errors or shard_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
