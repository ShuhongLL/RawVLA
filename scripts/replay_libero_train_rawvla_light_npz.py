#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAWVLA_BENCH_ROOT = ROOT / "benchmark" / "rawvla-bench"
STARVLA_ROOT = ROOT / "starVLA"
OPENVLA_OFT_ROOT = ROOT / "third_party" / "openvla-oft"
for path in (RAWVLA_BENCH_ROOT, STARVLA_ROOT, OPENVLA_OFT_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from libero.libero import benchmark, get_libero_path  # noqa: E402
try:  # Newer LIBERO releases download assets into a user cache.
    from libero.libero import get_assets_path  # type: ignore[attr-defined]  # noqa: E402
except ImportError:  # Older releases only expose the configured assets path.
    get_assets_path = None
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
import libero.libero.utils.utils as libero_utils  # noqa: E402
from rawvla_bench.libero import apply_entry_lighting  # noqa: E402
from rawvla_bench.lighting import DEFAULT_EV_RANGES, LIGHTING_DOMAINS, lighting_parameters_for_sample, rawvla_light_xml_from_string  # noqa: E402
from rawvla_bench.raw import make_rawvla_observation, stable_uint32, unprocess_metadata  # noqa: E402


LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
SUITE_TO_LEROBOT = {
    "libero_10": "libero_10_no_noops_1.0.0_lerobot",
    "libero_goal": "libero_goal_no_noops_1.0.0_lerobot",
    "libero_object": "libero_object_no_noops_1.0.0_lerobot",
    "libero_spatial": "libero_spatial_no_noops_1.0.0_lerobot",
}


@dataclass(frozen=True)
class EpisodeRef:
    suite: str
    lerobot_dataset_name: str
    task_id: int
    task_name: str
    task_description: str
    hdf5_path: Path
    demo_key: str
    demo_index: int
    trajectory_id: int


def _is_noop(action: np.ndarray, prev_action: np.ndarray | None = None, threshold: float = 1e-4) -> bool:
    if prev_action is None:
        return float(np.linalg.norm(action[:-1])) < threshold
    return float(np.linalg.norm(action[:-1])) < threshold and float(action[-1]) == float(prev_action[-1])


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = math.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def _state_from_obs(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            _quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
        )
    ).astype(np.float32)


def _sample_lighting(
    *,
    seed: int,
    domains: tuple[str, ...],
    suite: str,
    task_id: int,
    demo_index: int,
    trajectory_id: int,
    variant_index: int,
    variant_domain_mode: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(
        stable_uint32("train_lighting", seed, suite, int(task_id), int(demo_index), int(trajectory_id), int(variant_index))
    )
    if variant_domain_mode == "cycle_domains":
        domain = str(domains[int(variant_index) % len(domains)])
    elif variant_domain_mode == "random":
        domain = str(domains[int(rng.integers(0, len(domains)))])
    else:
        raise ValueError(f"Unknown variant_domain_mode={variant_domain_mode!r}")
    ev_low, ev_high = DEFAULT_EV_RANGES[domain]
    sample_unit = float(rng.random())
    lighting_ev = float(ev_low + sample_unit * (ev_high - ev_low))
    params = lighting_parameters_for_sample(domain, lighting_ev, sample_unit=sample_unit)
    noise_seed = stable_uint32(
        "train_raw_noise",
        seed,
        suite,
        int(task_id),
        int(demo_index),
        int(trajectory_id),
        int(variant_index),
        domain,
    )
    return {
        "benchmark": "rawvla-bench-light-libero-v1-train-rerender-npz",
        "lighting_domain": domain,
        "lighting_sample_unit": sample_unit,
        **params,
        "noise_seed": int(noise_seed),
    }


def _stats_uint8(images: np.ndarray) -> dict[str, float]:
    arr = images.astype(np.float32) / 255.0
    luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return {
        "mean_luma": float(luma.mean()),
        "median_luma": float(np.median(luma)),
        "p01_luma": float(np.percentile(luma, 1.0)),
        "p95_luma": float(np.percentile(luma, 95.0)),
        "dark_ratio": float((luma < 0.03).mean()),
        "clip_ratio": float((arr >= 250.0 / 255.0).any(axis=-1).mean()),
    }


def _write_npz(path: Path, arrays: dict[str, Any], compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        if compressed:
            np.savez_compressed(handle, **arrays)
        else:
            np.savez(handle, **arrays)
    tmp.replace(path)


def _make_env(task, seed: int):
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(int(seed))
    return env


def _postprocess_demo_model_xml(xml: str) -> str:
    xml = libero_utils.postprocess_model_xml(xml, {})
    assets_path = get_assets_path() if get_assets_path is not None else get_libero_path("assets")
    libero_assets = str(Path(assets_path))
    legacy_checkout = re.compile(
        r"/(?:Users|home)/[^/]+/workspace/libero-dev/chiliocosm"
    )
    xml = legacy_checkout.sub(str(Path(libero_assets).parent), xml)
    return xml


def _discover_episodes(hdf5_root: Path, suites: tuple[str, ...]) -> list[EpisodeRef]:
    benchmark_dict = benchmark.get_benchmark_dict()
    refs: list[EpisodeRef] = []
    for suite in suites:
        if suite not in SUITE_TO_LEROBOT:
            raise ValueError(f"Unsupported suite for train replay: {suite!r}")
        task_suite = benchmark_dict[suite]()
        trajectory_id = 0
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            hdf5_path = hdf5_root / suite / f"{task.name}_demo.hdf5"
            if not hdf5_path.exists():
                raise FileNotFoundError(f"Missing HDF5 demo file: {hdf5_path}")
            with h5py.File(hdf5_path, "r") as f:
                demo_keys = sorted(f["data"].keys(), key=lambda key: int(key.split("_", 1)[1]))
            for demo_key in demo_keys:
                demo_index = int(demo_key.split("_", 1)[1])
                refs.append(
                    EpisodeRef(
                        suite=suite,
                        lerobot_dataset_name=SUITE_TO_LEROBOT[suite],
                        task_id=task_id,
                        task_name=task.name,
                        task_description=task.language,
                        hdf5_path=hdf5_path,
                        demo_key=demo_key,
                        demo_index=demo_index,
                        trajectory_id=trajectory_id,
                    )
                )
                trajectory_id += 1
    return refs


def _select_shard(refs: list[EpisodeRef], num_shards: int, shard_index: int) -> list[EpisodeRef]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    return [ref for index, ref in enumerate(refs) if index % num_shards == shard_index]


def _replay_episode(
    ref: EpisodeRef,
    *,
    seed: int,
    lighting: dict[str, Any],
    representation: str,
    replay_mode: str,
) -> dict[str, Any] | None:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[ref.suite]()
    task = task_suite.get_task(ref.task_id)
    with h5py.File(ref.hdf5_path, "r") as f:
        demo = f["data"][ref.demo_key]
        orig_actions = np.asarray(demo["actions"][()], dtype=np.float32)
        orig_states = np.asarray(demo["states"][()])
        orig_dones = np.asarray(demo["dones"][()], dtype=np.uint8) if "dones" in demo else None
        orig_robot_states = np.asarray(demo["robot_states"][()], dtype=np.float32) if "robot_states" in demo else None
        model_xml = _postprocess_demo_model_xml(demo.attrs["model_file"])

    raw_env = _make_env(task, seed)
    rgb_env = _make_env(task, seed)
    try:
        raw_env.reset()
        rawvla_xml = rawvla_light_xml_from_string(model_xml)
        raw_env.reset_from_xml_string(rawvla_xml)
        raw_env.sim.reset()
        rgb_env.reset()
        rgb_env.reset_from_xml_string(model_xml)
        rgb_env.sim.reset()
        apply_entry_lighting(raw_env, lighting)

        actions: list[np.ndarray] = []
        states: list[np.ndarray] = []
        agentview_raw: list[np.ndarray] = []
        wrist_raw: list[np.ndarray] = []
        agentview_rgb: list[np.ndarray] = []
        wrist_rgb: list[np.ndarray] = []
        agentview_target_rgbs: list[np.ndarray] = []
        wrist_target_rgbs: list[np.ndarray] = []
        robot_states: list[np.ndarray] = []
        noops = 0
        done = False

        if replay_mode == "actions":
            rgb_obs = rgb_env.set_init_state(orig_states[0])
            for _ in range(10):
                rgb_obs, _, rgb_done, _ = rgb_env.step(LIBERO_DUMMY_ACTION)
                done = bool(rgb_done)
        elif replay_mode != "states":
            raise ValueError(f"Unknown replay_mode={replay_mode!r}")

        for action_index, action in enumerate(orig_actions):
            prev_action = actions[-1] if actions else None
            if _is_noop(action, prev_action):
                noops += 1
                continue

            if replay_mode == "states":
                raw_obs = raw_env.set_init_state(orig_states[action_index])
                rgb_obs = rgb_env.set_init_state(orig_states[action_index])
                current_state = np.asarray(orig_states[action_index])
            else:
                current_state = rgb_env.sim.get_state().flatten()
                raw_obs = raw_env.set_init_state(current_state)

            frame_index = len(actions)
            agent_target_rgb = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1])
            wrist_target_rgb = np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1])
            agent_baseline_rgb = np.ascontiguousarray(rgb_obs["agentview_image"][::-1, ::-1])
            wrist_baseline_rgb = np.ascontiguousarray(rgb_obs["robot0_eye_in_hand_image"][::-1, ::-1])
            agentview_rgb.append(agent_baseline_rgb)
            wrist_rgb.append(wrist_baseline_rgb)
            agentview_target_rgbs.append(agent_target_rgb)
            wrist_target_rgbs.append(wrist_target_rgb)
            agentview_raw.append(
                make_rawvla_observation(
                    agent_target_rgb,
                    noise_seed=int(lighting["noise_seed"]),
                    frame_index=frame_index,
                    view_index=0,
                    representation=representation,
                    sensor_saturation_ev=float(lighting.get("sensor_saturation_ev", 0.0)),
                )
            )
            wrist_raw.append(
                make_rawvla_observation(
                    wrist_target_rgb,
                    noise_seed=int(lighting["noise_seed"]),
                    frame_index=frame_index,
                    view_index=1,
                    representation=representation,
                    sensor_saturation_ev=float(lighting.get("sensor_saturation_ev", 0.0)),
                )
            )
            if replay_mode == "states":
                states.append(current_state)
                if orig_robot_states is not None:
                    robot_states.append(np.asarray(orig_robot_states[action_index], dtype=np.float32))
                else:
                    robot_states.append(_state_from_obs(rgb_obs))
            else:
                states.append(current_state)
                robot_states.append(_state_from_obs(rgb_obs))
            actions.append(np.asarray(action, dtype=np.float32))
            if replay_mode == "states":
                if orig_dones is None:
                    done = True
                elif bool(orig_dones[action_index]):
                    done = True
                    break
            else:
                rgb_obs, _, rgb_done, _ = rgb_env.step(action.tolist())
                done = bool(rgb_done)
                if done:
                    break

        if replay_mode == "states" and orig_dones is not None and bool(orig_dones.any()):
            done = True

        if not actions or not done:
            return None

        agent_arr = np.stack(agentview_raw, axis=0).astype(np.uint8, copy=False)
        wrist_arr = np.stack(wrist_raw, axis=0).astype(np.uint8, copy=False)
        agent_rgb_arr = np.stack(agentview_rgb, axis=0).astype(np.uint8, copy=False)
        wrist_rgb_arr = np.stack(wrist_rgb, axis=0).astype(np.uint8, copy=False)
        agent_target_rgb_arr = np.stack(agentview_target_rgbs, axis=0).astype(np.uint8, copy=False)
        wrist_target_rgb_arr = np.stack(wrist_target_rgbs, axis=0).astype(np.uint8, copy=False)
        action_arr = np.stack(actions, axis=0).astype(np.float32, copy=False)
        state_arr = np.stack(states, axis=0)
        robot_state_arr = np.stack(robot_states, axis=0).astype(np.float32, copy=False)
        rewards = np.zeros(len(actions), dtype=np.uint8)
        dones = np.zeros(len(actions), dtype=np.uint8)
        rewards[-1] = 1
        dones[-1] = 1
        return {
            "agentview_raw_uint8": agent_arr,
            "wrist_raw_uint8": wrist_arr,
            "agentview_rgb_uint8": agent_rgb_arr,
            "wrist_rgb_uint8": wrist_rgb_arr,
            "agentview_target_rgb_uint8": agent_target_rgb_arr,
            "wrist_target_rgb_uint8": wrist_target_rgb_arr,
            "actions": action_arr,
            "states": state_arr,
            "robot_states": robot_state_arr,
            "rewards": rewards,
            "dones": dones,
            "noops_filtered": noops,
            "done": bool(done),
        }
    finally:
        raw_env.close()
        rgb_env.close()


def render_target_light_rgb_for_states(
    ref: EpisodeRef,
    *,
    seed: int,
    lighting: dict[str, Any],
    states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Render noise-free target-light RGB at the exact persisted simulator states."""
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[ref.suite]()
    task = task_suite.get_task(ref.task_id)
    with h5py.File(ref.hdf5_path, "r") as stream:
        model_xml = _postprocess_demo_model_xml(stream["data"][ref.demo_key].attrs["model_file"])
    env = _make_env(task, seed)
    try:
        env.reset()
        env.reset_from_xml_string(rawvla_light_xml_from_string(model_xml))
        env.sim.reset()
        apply_entry_lighting(env, lighting)
        agent = []
        wrist = []
        for state in np.asarray(states):
            observation = env.set_init_state(state)
            agent.append(np.ascontiguousarray(observation["agentview_image"][::-1, ::-1]))
            wrist.append(np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1]))
        return (
            np.stack(agent).astype(np.uint8, copy=False),
            np.stack(wrist).astype(np.uint8, copy=False),
        )
    finally:
        env.close()


def _process_refs(refs: list[EpisodeRef], cfg: dict[str, Any], worker_id: int) -> str:
    out_root = Path(cfg["out_root"])
    rgb_out_root = Path(cfg["rgb_out_root"]) if cfg.get("rgb_out_root") else None
    manifest_stem = f"shard_{int(cfg.get('shard_index', 0)):04d}_worker_{worker_id:04d}"
    manifest_path = out_root / "_manifests" / f"{manifest_stem}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_manifest = None
    if rgb_out_root is not None:
        rgb_manifest_path = rgb_out_root / "_manifests" / f"{manifest_stem}.jsonl"
        rgb_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        rgb_manifest = rgb_manifest_path.open("w", encoding="utf-8")
    written = 0
    skipped = 0
    failed = 0
    started = time.time()
    domains = tuple(cfg["domains"])

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for ref in refs:
            for variant_index in range(int(cfg["k_per_episode"])):
                lighting = _sample_lighting(
                    seed=int(cfg["seed"]),
                    domains=domains,
                    suite=ref.suite,
                    task_id=ref.task_id,
                    demo_index=ref.demo_index,
                    trajectory_id=ref.trajectory_id,
                    variant_index=variant_index,
                    variant_domain_mode=str(cfg["variant_domain_mode"]),
                )
                rel_path = (
                    Path(ref.lerobot_dataset_name)
                    / f"episode_{ref.trajectory_id:06d}"
                    / f"variant_{variant_index:02d}.npz"
                )
                out_path = out_root / rel_path
                rgb_out_path = rgb_out_root / rel_path if rgb_out_root is not None else None
                outputs_exist = out_path.exists() and (rgb_out_path is None or rgb_out_path.exists())
                if outputs_exist and not bool(cfg["overwrite"]):
                    skipped += 1
                    skipped_entry = {"npz_relpath": str(rel_path), "skipped": True, **lighting}
                    manifest.write(json.dumps(skipped_entry) + "\n")
                    if rgb_manifest is not None:
                        rgb_manifest.write(json.dumps(skipped_entry) + "\n")
                    manifest.flush()
                    if rgb_manifest is not None:
                        rgb_manifest.flush()
                    continue

                error_message = ""
                try:
                    episode = _replay_episode(
                        ref,
                        seed=int(cfg["env_seed"]),
                        lighting=lighting,
                        representation=str(cfg["representation"]),
                        replay_mode=str(cfg["replay_mode"]),
                    )
                except Exception as exc:
                    episode = None
                    error_message = f"{type(exc).__name__}: {exc}"
                if episode is None:
                    failed += 1
                    manifest.write(
                        json.dumps(
                            {
                                "npz_relpath": str(rel_path),
                                "failed": True,
                                "suite": ref.suite,
                                "task_id": ref.task_id,
                                "demo_key": ref.demo_key,
                                "trajectory_id": ref.trajectory_id,
                                "error": error_message,
                                **lighting,
                            },
                            sort_keys=True,
                    )
                    + "\n"
                    )
                    if rgb_manifest is not None:
                        rgb_manifest.write(
                            json.dumps(
                                {
                                    "npz_relpath": str(rel_path),
                                    "failed": True,
                                    "suite": ref.suite,
                                    "task_id": ref.task_id,
                                    "demo_key": ref.demo_key,
                                    "trajectory_id": ref.trajectory_id,
                                    "error": error_message,
                                    **lighting,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                    manifest.flush()
                    if rgb_manifest is not None:
                        rgb_manifest.flush()
                    continue

                metadata = {
                    **lighting,
                    "suite": ref.suite,
                    "dataset_name": ref.lerobot_dataset_name,
                    "trajectory_id": ref.trajectory_id,
                    "variant_index": int(variant_index),
                    "task_id": ref.task_id,
                    "task_name": ref.task_name,
                    "task_description": ref.task_description,
                    "hdf5_path": str(ref.hdf5_path),
                    "demo_key": ref.demo_key,
                    "demo_index": ref.demo_index,
                    "num_frames": int(episode["agentview_raw_uint8"].shape[0]),
                    "source_shape": [LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3],
                    "raw_shape": list(episode["agentview_raw_uint8"].shape[1:]),
                    "representation": cfg["representation"],
                    "noops_filtered": int(episode["noops_filtered"]),
                    "render_order": "env_lighting_then_rgb256_then_fixed_unprocess_noise_then_policy_resize",
                    "paired_rgb_render_order": "default_lighting_rgb256_then_rotate_180",
                    "replay_mode": cfg["replay_mode"],
                    "npz_relpath": str(rel_path),
                    "rgb_npz_relpath": str(rel_path) if rgb_out_root is not None else "",
                    "agentview_stats": _stats_uint8(episode["agentview_raw_uint8"]),
                    "wrist_stats": _stats_uint8(episode["wrist_raw_uint8"]),
                }
                arrays = {
                    "agentview_raw_uint8": episode["agentview_raw_uint8"],
                    "wrist_raw_uint8": episode["wrist_raw_uint8"],
                    "actions": episode["actions"],
                    "states": episode["states"],
                    "robot_states": episode["robot_states"],
                    "rewards": episode["rewards"],
                    "dones": episode["dones"],
                    "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
                }
                _write_npz(out_path, arrays, compressed=bool(cfg["compressed"]))
                if rgb_out_path is not None:
                    rgb_metadata = {
                        **metadata,
                        "benchmark": "rawvla-bench-light-libero-v1-baseline-rgb-replay-npz",
                        "representation": "rgb_uint8",
                        "render_order": "default_lighting_rgb256_then_rotate_180",
                        "raw_npz_relpath": str(rel_path),
                        "agentview_stats": _stats_uint8(episode["agentview_rgb_uint8"]),
                        "wrist_stats": _stats_uint8(episode["wrist_rgb_uint8"]),
                    }
                    rgb_arrays = {
                        "agentview_rgb_uint8": episode["agentview_rgb_uint8"],
                        "wrist_rgb_uint8": episode["wrist_rgb_uint8"],
                        "actions": episode["actions"],
                        "states": episode["states"],
                        "robot_states": episode["robot_states"],
                        "rewards": episode["rewards"],
                        "dones": episode["dones"],
                        "metadata_json": np.asarray(json.dumps(rgb_metadata, sort_keys=True)),
                    }
                    _write_npz(rgb_out_path, rgb_arrays, compressed=bool(cfg["compressed"]))
                manifest.write(json.dumps({"npz_relpath": str(rel_path), **metadata}, sort_keys=True) + "\n")
                if rgb_manifest is not None:
                    rgb_manifest.write(json.dumps({"npz_relpath": str(rel_path), **rgb_metadata}, sort_keys=True) + "\n")
                manifest.flush()
                if rgb_manifest is not None:
                    rgb_manifest.flush()
                written += 1

    if rgb_manifest is not None:
        rgb_manifest.close()
    return json.dumps(
        {
            "worker_id": worker_id,
            "written": written,
            "skipped": skipped,
            "failed": failed,
            "seconds": round(time.time() - started, 3),
        }
    )


def _combine_manifests(out_root: Path, cfg: dict[str, Any], worker_results: list[str], benchmark_name: str) -> None:
    entries: list[dict[str, Any]] = []
    for path in sorted((out_root / "_manifests").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
    entries.sort(key=lambda item: (item.get("dataset_name", ""), int(item.get("trajectory_id", -1)), int(item.get("variant_index", -1))))
    manifest = {
        "benchmark": benchmark_name,
        "version": 1,
        "created_at_unix": time.time(),
        "config": cfg,
        "ev_ranges": {key: list(value) for key, value in DEFAULT_EV_RANGES.items()},
        "lighting_domains": list(LIGHTING_DOMAINS),
        "unprocess": unprocess_metadata(),
        "worker_results": [json.loads(item) for item in worker_results],
        "num_entries": len(entries),
        "entries": entries,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay LIBERO HDF5 demos under RAWVLA-Bench-Light settings into NPZ.")
    parser.add_argument(
        "--hdf5-root",
        type=Path,
        default=ROOT / "benchmark_data" / "libero" / "LIBERO-datasets",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "benchmark_data" / "libero" / "rawvla_light_train_rerender_npz",
        help="Output root for target-light RAW NPZ files.",
    )
    parser.add_argument(
        "--rgb-out-root",
        default=str(ROOT / "benchmark_data" / "libero" / "rawvla_light_train_rgb_replay_npz"),
        help="Output root for paired baseline RGB NPZ files. Pass an empty string to disable.",
    )
    parser.add_argument("--suites", default="libero_spatial,libero_object,libero_goal,libero_10")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--k-per-episode", type=int, default=1)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 8, 32))
    parser.add_argument("--max-episodes", type=int, default=-1, help="Global cap across selected suites for smoke tests.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--domains", default=",".join(LIGHTING_DOMAINS))
    parser.add_argument("--variant-domain-mode", choices=["random", "cycle_domains"], default="random")
    parser.add_argument("--representation", choices=["raw", "default_isp"], default="raw")
    parser.add_argument("--replay-mode", choices=["states", "actions"], default="states")
    parser.add_argument("--compression", choices=["stored", "deflated"], default="stored")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    suites = tuple(part.strip() for part in args.suites.split(",") if part.strip())
    domains = tuple(part.strip() for part in args.domains.split(",") if part.strip())
    for domain in domains:
        if domain not in LIGHTING_DOMAINS:
            raise ValueError(f"Unknown lighting domain {domain!r}; valid domains are {LIGHTING_DOMAINS}")
    if args.k_per_episode < 1:
        raise ValueError("--k-per-episode must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    refs = _discover_episodes(args.hdf5_root, suites)
    refs = _select_shard(refs, args.num_shards, args.shard_index)
    if args.max_episodes >= 0:
        refs = refs[: args.max_episodes]
    if not refs:
        raise FileNotFoundError(
            f"No LIBERO episodes found under {args.hdf5_root} for suites={suites}"
        )
    args.out_root.mkdir(parents=True, exist_ok=True)
    rgb_out_root = Path(args.rgb_out_root) if str(args.rgb_out_root) else None
    if rgb_out_root is not None:
        rgb_out_root.mkdir(parents=True, exist_ok=True)
    cfg = {
        "hdf5_root": str(args.hdf5_root),
        "out_root": str(args.out_root),
        "rgb_out_root": str(rgb_out_root) if rgb_out_root is not None else "",
        "suites": list(suites),
        "seed": args.seed,
        "env_seed": args.env_seed,
        "k_per_episode": args.k_per_episode,
        "domains": list(domains),
        "variant_domain_mode": args.variant_domain_mode,
        "representation": args.representation,
        "replay_mode": args.replay_mode,
        "compressed": args.compression == "deflated",
        "max_episodes": args.max_episodes,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "overwrite": args.overwrite,
        "source": "libero_hdf5_simulator_rerender",
    }
    (args.out_root / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    worker_count = min(int(args.workers), max(len(refs), 1))
    buckets = [[] for _ in range(worker_count)]
    for index, ref in enumerate(refs):
        buckets[index % worker_count].append(ref)

    if worker_count == 1:
        results = [_process_refs(buckets[0], cfg, 0)]
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(worker_count) as pool:
            results = pool.starmap(_process_refs, [(bucket, cfg, idx) for idx, bucket in enumerate(buckets)])

    _combine_manifests(args.out_root, cfg, results, "rawvla-bench-light-libero-v1-train-rerender-raw-npz")
    if rgb_out_root is not None:
        _combine_manifests(rgb_out_root, cfg, results, "rawvla-bench-light-libero-v1-baseline-rgb-replay-npz")
    print(
        json.dumps(
            {
                "out_root": str(args.out_root),
                "rgb_out_root": str(rgb_out_root) if rgb_out_root is not None else "",
                "worker_results": [json.loads(item) for item in results],
            },
            indent=2,
        )
    )
    failed = sum(int(json.loads(item)["failed"]) for item in results)
    if failed:
        raise SystemExit(f"LIBERO replay failed for {failed} variant(s); inspect _manifests/*.jsonl")


if __name__ == "__main__":
    main()
