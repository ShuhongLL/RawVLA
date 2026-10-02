#!/usr/bin/env python3
"""Create paired clean-RGB/action and random-light RAW/state-copy RoboTwin2 data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback
import types
from typing import Any

import numpy as np


VIEWS = ("head_camera", "left_camera", "right_camera")


def atomic_npz(path: Path, arrays: dict[str, Any], *, compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        if compressed:
            np.savez_compressed(stream, **arrays)
        else:
            np.savez(stream, **arrays)
    os.replace(temporary, path)


def actor_names(scene: object) -> list[str]:
    return [str(actor.get_name()) for actor in scene.get_all_actors()]


def articulation_names(scene: object) -> list[str]:
    return [str(articulation.get_name()) for articulation in scene.get_all_articulations()]


def copy_physics_state(source: object, target: object) -> None:
    """Copy physical state between identically seeded RoboTwin scenes, without stepping target."""
    source_actors = source.scene.get_all_actors()
    target_actors = target.scene.get_all_actors()
    source_actor_names = actor_names(source.scene)
    target_actor_names = actor_names(target.scene)
    if source_actor_names != target_actor_names:
        raise RuntimeError(
            "Actor topology differs between clean and RAW scenes: "
            f"clean={source_actor_names}, raw={target_actor_names}"
        )
    for source_actor, target_actor in zip(source_actors, target_actors):
        target_actor.set_pose(source_actor.get_pose())
        source_dynamic = source_actor.find_component_by_type(
            __import__("sapien").physx.PhysxRigidDynamicComponent
        )
        target_dynamic = target_actor.find_component_by_type(
            __import__("sapien").physx.PhysxRigidDynamicComponent
        )
        if (source_dynamic is None) != (target_dynamic is None):
            raise RuntimeError(f"Rigid-body type mismatch for actor {source_actor.get_name()!r}")
        if source_dynamic is not None:
            target_dynamic.set_linear_velocity(source_dynamic.get_linear_velocity())
            target_dynamic.set_angular_velocity(source_dynamic.get_angular_velocity())

    source_articulations = source.scene.get_all_articulations()
    target_articulations = target.scene.get_all_articulations()
    if articulation_names(source.scene) != articulation_names(target.scene):
        raise RuntimeError(
            "Articulation topology differs between clean and RAW scenes: "
            f"clean={articulation_names(source.scene)}, raw={articulation_names(target.scene)}"
        )
    for source_articulation, target_articulation in zip(source_articulations, target_articulations):
        target_articulation.set_root_pose(source_articulation.get_root_pose())
        target_articulation.set_root_linear_velocity(source_articulation.get_root_linear_velocity())
        target_articulation.set_root_angular_velocity(source_articulation.get_root_angular_velocity())
        target_articulation.set_qpos(source_articulation.get_qpos())
        target_articulation.set_qvel(source_articulation.get_qvel())
    target.scene.update_render()


def stack_frames(frames: list[dict[str, Any]], key: str, dtype: Any) -> np.ndarray:
    return np.stack([np.asarray(frame[key]) for frame in frames], axis=0).astype(dtype, copy=False)


def build_arrays(
    frames: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    target_light_rgb: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not frames:
        raise RuntimeError("Clean replay produced no captured frames")
    common = {
        "actions": stack_frames(frames, "actions", np.float32),
        "robot_states": stack_frames(frames, "robot_states", np.float32),
        "left_endpose": stack_frames(frames, "left_endpose", np.float32),
        "right_endpose": stack_frames(frames, "right_endpose", np.float32),
        "dones": np.asarray([0] * (len(frames) - 1) + [1], dtype=np.uint8),
        "rewards": np.asarray([0] * (len(frames) - 1) + [1], dtype=np.uint8),
    }
    raw_arrays = dict(common)
    rgb_arrays = dict(common)
    for view in VIEWS:
        raw_arrays[f"{view}_raw_float32"] = stack_frames(frames, f"{view}_raw", np.float32)
        raw_arrays[f"{view}_clean_raw_float32"] = stack_frames(
            frames, f"{view}_clean_raw", np.float32
        )
        rgb_source = f"{view}_target_rgb" if target_light_rgb else f"{view}_rgb"
        rgb_arrays[f"{view}_rgb_uint8"] = stack_frames(frames, rgb_source, np.uint8)
    raw_metadata = {**metadata, "representation": "three_channel_linear_float32"}
    rgb_metadata = {
        **metadata,
        "benchmark": (
            "rawvla-bench-light-robotwin2-v1-target-light-clean-rgb"
            if target_light_rgb
            else "rawvla-bench-light-robotwin2-v1-train-clean-rgb"
        ),
        "representation": (
            "target_light_clean_render_rgb_uint8"
            if target_light_rgb
            else "clean_render_rgb_uint8"
        ),
    }
    raw_arrays["metadata_json"] = np.asarray(json.dumps(raw_metadata, sort_keys=True))
    rgb_arrays["metadata_json"] = np.asarray(json.dumps(rgb_metadata, sort_keys=True))
    return raw_arrays, rgb_arrays


def robot_state(observation: dict[str, Any]) -> np.ndarray:
    left = np.asarray(observation["endpose"]["left_endpose"], dtype=np.float32).reshape(-1)
    right = np.asarray(observation["endpose"]["right_endpose"], dtype=np.float32).reshape(-1)
    left_gripper = np.asarray([observation["endpose"]["left_gripper"]], dtype=np.float32)
    right_gripper = np.asarray([observation["endpose"]["right_gripper"]], dtype=np.float32)
    return np.concatenate((left, left_gripper, right, right_gripper)).astype(np.float32)


def replay_entry(
    entry: dict[str, Any],
    *, robotwin_root: Path,
    output_root: Path,
    compression: bool,
    target_light_rgb: bool = False,
    rgb_only: bool = False,
) -> dict[str, Any]:
    from replay_robotwin2_clean_training import build_robotwin_args
    from script.collect_data import class_decorator
    from envs.lighting import apply_lighting

    task_name = str(entry["task_name"])
    episode = int(entry["episode"])
    raw_path = output_root / "raw" / task_name / f"episode_{episode:06d}.npz"
    rgb_path = output_root / "rgb" / task_name / f"episode_{episode:06d}.npz"
    if rgb_path.is_file() and (rgb_only or raw_path.is_file()):
        return {"status": "skipped", "raw_npz": str(raw_path), "rgb_npz": str(rgb_path)}

    with Path(entry["source_trajectory"]).open("rb") as stream:
        trajectory = pickle.load(stream)
    task_root = output_root / "_runtime" / task_name
    base_args = build_robotwin_args(robotwin_root, task_root, task_name)
    base_args.update(
        {
            "now_ep_num": episode,
            "seed": int(entry["dataset_seed"]),
            "left_joint_path": trajectory["left_joint_path"],
            "right_joint_path": trajectory["right_joint_path"],
        }
    )
    clean_args = dict(base_args)
    clean_args.update({"save_data": True, "hdr_raw_white_level": 0.0, "raw_sensor_noise_enabled": False})
    raw_args = dict(base_args)
    raw_args.update(
        {
            "save_data": False,
            "hdr_raw_white_level": float(entry["raw_white_level"]),
            "raw_sensor_noise_enabled": bool(entry["raw_sensor_noise_enabled"]),
            "raw_sensor_noise_seed": int(entry["noise_seed"]),
            "raw_sensor_shot_noise": float(entry["raw_sensor_shot_noise"]),
            "raw_sensor_read_noise": float(entry["raw_sensor_read_noise"]),
        }
    )

    clean_env = class_decorator(task_name)
    raw_env = class_decorator(task_name)
    clean_open = False
    raw_open = False
    frames: list[dict[str, Any]] = []
    try:
        clean_env.setup_demo(**clean_args)
        clean_open = True
        raw_env.setup_demo(**raw_args)
        raw_open = True
        lighting = apply_lighting(
            raw_env,
            str(entry["lighting_domain"]),
            str(entry["lighting_strategy"]),
            sample_unit=float(entry["lighting_sample_unit"]),
        )

        def paired_take_picture(self: object, observation: dict[str, Any] | None = None) -> None:
            clean_observation = self.get_obs() if observation is None else observation
            copy_physics_state(clean_env, raw_env)
            raw_observation = raw_env.get_obs()
            noise_enabled = bool(raw_env.cameras.raw_sensor_noise_enabled)
            raw_env.cameras.raw_sensor_noise_enabled = False
            clean_raw_observation = raw_env.get_obs()
            raw_env.cameras.raw_sensor_noise_enabled = noise_enabled
            # The clean reread is the same physical frame and must not consume
            # the next noisy-frame seed index.
            raw_env.cameras.raw_sensor_noise_frame_index -= 1
            frame: dict[str, Any] = {
                "actions": np.asarray(clean_observation["joint_action"]["vector"], dtype=np.float32),
                "robot_states": robot_state(clean_observation),
                "left_endpose": np.asarray(clean_observation["endpose"]["left_endpose"], dtype=np.float32),
                "right_endpose": np.asarray(clean_observation["endpose"]["right_endpose"], dtype=np.float32),
            }
            for view in VIEWS:
                frame[f"{view}_rgb"] = np.asarray(
                    clean_observation["observation"][view]["render_rgb"], dtype=np.uint8
                ).copy()
                frame[f"{view}_target_rgb"] = np.asarray(
                    raw_observation["observation"][view]["render_rgb"], dtype=np.uint8
                ).copy()
                frame[f"{view}_raw"] = np.asarray(
                    raw_observation["observation"][view]["raw_linear"], dtype=np.float32
                ).copy()
                frame[f"{view}_clean_raw"] = np.asarray(
                    clean_raw_observation["observation"][view]["raw_linear"], dtype=np.float32
                ).copy()
            frames.append(frame)

        clean_env._take_picture = types.MethodType(paired_take_picture, clean_env)
        clean_env.set_path_lst(clean_args)
        clean_env.play_once()
        success = bool(clean_env.plan_success and clean_env.check_success())
        if not success:
            raise RuntimeError("Clean RGB action replay no longer passes task check_success")

        metadata = {
            **entry,
            "benchmark": "rawvla-bench-light-robotwin2-v1-train-state-copy",
            "num_frames": len(frames),
            "views": list(VIEWS),
            "clean_protocol": "original_lighting_joint_action_replay",
            "raw_protocol": "copy_clean_physics_state_each_frame_then_render_without_action",
            "lighting_applied": lighting,
            "raw_npz": str(raw_path),
            "rgb_npz": str(rgb_path),
        }
        raw_arrays, rgb_arrays = build_arrays(
            frames, metadata, target_light_rgb=target_light_rgb
        )
        if not rgb_only:
            atomic_npz(raw_path, raw_arrays, compressed=compression)
        atomic_npz(rgb_path, rgb_arrays, compressed=compression)
        return {
            "status": "completed",
            "task_name": task_name,
            "episode": episode,
            "num_frames": len(frames),
            "lighting_domain": entry["lighting_domain"],
            "lighting_ev": entry["lighting_ev"],
            "raw_npz": str(raw_path),
            "rgb_npz": str(rgb_path),
        }
    finally:
        if clean_open:
            clean_env.close_env(clear_cache=False)
        if raw_open:
            raw_env.close_env(clear_cache=False)
        try:
            from sapien.render import clear_cache as sapien_clear_cache

            sapien_clear_cache()
        except Exception:
            pass


def worker_main(args: argparse.Namespace, manifest: dict[str, Any]) -> int:
    entries = [
        entry
        for entry in manifest["entries"]
        if int(entry["entry_index"]) % args.num_workers == args.worker_index
    ]
    record_dir = args.output_root / "_manifests"
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / f"worker_{args.worker_index:02d}.jsonl"
    failures = 0
    consecutive_failures = 0
    started = time.time()
    with record_path.open("a", encoding="utf-8") as records:
        for entry in entries:
            try:
                result = replay_entry(
                    entry,
                    robotwin_root=args.robotwin_root,
                    output_root=args.output_root,
                    compression=args.compression == "deflated",
                    target_light_rgb=args.target_light_rgb,
                    rgb_only=args.rgb_only,
                )
            except Exception:
                failures += 1
                consecutive_failures += 1
                result = {
                    "status": "failed",
                    "task_name": entry["task_name"],
                    "episode": entry["episode"],
                    "error": traceback.format_exc(),
                }
            else:
                consecutive_failures = 0
            result.update({"entry_index": entry["entry_index"], "worker_index": args.worker_index})
            records.write(json.dumps(result, sort_keys=True) + "\n")
            records.flush()
            print(json.dumps(result, sort_keys=True), flush=True)
            if consecutive_failures >= 3:
                print("aborting worker after three consecutive replay failures", flush=True)
                break
    print(
        json.dumps(
            {
                "worker_index": args.worker_index,
                "entries": len(entries),
                "failures": failures,
                "seconds": time.time() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 1 if failures else 0


def finalize(args: argparse.Namespace, manifest: dict[str, Any]) -> int:
    completed = []
    missing = []
    for entry in manifest["entries"]:
        task = str(entry["task_name"])
        episode = int(entry["episode"])
        raw_path = args.output_root / "raw" / task / f"episode_{episode:06d}.npz"
        rgb_path = args.output_root / "rgb" / task / f"episode_{episode:06d}.npz"
        if raw_path.is_file() and rgb_path.is_file():
            completed.append({**entry, "raw_npz": str(raw_path), "rgb_npz": str(rgb_path)})
        else:
            missing.append({"task_name": task, "episode": episode})
    output = {
        **{key: value for key, value in manifest.items() if key != "entries"},
        "num_completed": len(completed),
        "num_missing": len(missing),
        "missing": missing,
        "entries": completed,
    }
    path = args.output_root / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"completed": len(completed), "missing": len(missing), "manifest": str(path)}, indent=2))
    return 1 if missing else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--compression", choices=("stored", "deflated"), default="stored")
    parser.add_argument(
        "--target-light-rgb",
        action="store_true",
        help="Save the noise-free RGB render from the target-light RAW scene instead of original-light RGB.",
    )
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help="Write only the paired RGB NPZ, without duplicating the RAW NPZ.",
    )
    parser.add_argument(
        "--selection-json",
        type=Path,
        help="Optional burst-ablation selection; restrict replay to selected RoboTwin task/episode pairs.",
    )
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.robotwin_root = args.robotwin_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.num_workers < 1 or not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker index must satisfy 0 <= worker-index < num-workers")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.selection_json is not None:
        selected_rows = json.loads(args.selection_json.read_text(encoding="utf-8"))
        selected = {
            (str(row["task"]), int(Path(row["raw_path"]).stem.split("_")[-1]))
            for row in selected_rows
            if row["benchmark"] == "robotwin"
        }
        manifest["entries"] = [
            entry for entry in manifest["entries"]
            if (str(entry["task_name"]), int(entry["episode"])) in selected
        ]
        if len(manifest["entries"]) != len(selected):
            raise RuntimeError(
                f"selection matched {len(manifest['entries'])}/{len(selected)} RoboTwin entries"
            )
    if args.finalize_only:
        return finalize(args, manifest)
    os.environ.pop("ROBOTWIN_LIGHTING_DOMAIN", None)
    os.environ.setdefault("ROBOTWIN_ISP_AXIS", "none")
    os.chdir(args.robotwin_root)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(args.robotwin_root))
    sys.path.insert(0, str(args.robotwin_root / "script"))
    return worker_main(args, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
