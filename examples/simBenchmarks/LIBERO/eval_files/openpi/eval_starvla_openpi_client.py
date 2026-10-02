#!/usr/bin/env python3
"""LIBERO client eval for StarVLA PI0/PI05 served over websocket."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import tqdm

from deployment.model_server.tools import image_tools
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.simBenchmarks.LIBERO.eval_files.raw_rgb10_unprocessing import (
    episode_view_seed,
    params_dict,
    quantize_image_to_float_bits,
    sample_raw_rgb10_params,
    unprocess_rgb8_to_raw_rgb10_view8,
    make_ev_model_input,
    make_chromatic_model_input,
    make_noise_level_model_input,
    make_tonal_model_input,
)
from pi_libero_common import (
    DEFAULT_LIBERO_HOME,
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    canonicalize_model_name,
    canonicalize_model_source,
    get_max_steps,
    normalize_openpi_value,
    OPENPI_MODEL_SOURCE,
    postprocess_openpi_actions,
    preprocess_env_obs,
    resolve_norm_stats_source,
    STARVLA_MODEL_SOURCE,
)

log = logging.getLogger("eval_starvla_openpi_client")

RAWVLA_BENCH_ROOT = Path(__file__).resolve().parents[6] / "benchmark" / "rawvla-bench"
if RAWVLA_BENCH_ROOT.exists() and str(RAWVLA_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RAWVLA_BENCH_ROOT))


class StarVLAOpenPIClient:
    def __init__(
        self,
        host: str,
        port: int,
        model_name: str,
        norm_stats: dict[str, dict[str, np.ndarray]] | None = None,
        noise_seed: int | None = None,
        assets_checkpoint: Path | None = None,
        model_source: str = OPENPI_MODEL_SOURCE,
    ):
        self.policy = WebsocketClientPolicy(host=host, port=port)
        self.model_name = canonicalize_model_name(model_name)
        self.metadata = self.policy.get_server_metadata()
        checkpoint = self.metadata.get("checkpoint")
        server_source = self.metadata.get("model_source", OPENPI_MODEL_SOURCE)
        self.model_source = canonicalize_model_source(model_source or server_source)
        self.norm_stats = norm_stats or resolve_norm_stats_source(
            self.model_name,
            checkpoint=Path(checkpoint) if checkpoint else None,
            assets_checkpoint=assets_checkpoint,
        )
        self.use_quantile_norm = self.model_name != "PI0"
        self.last_action_stats: dict[str, Any] | None = None
        self.action_horizon = int(self.metadata.get("action_horizon", 10 if self.model_name == "PI05" else 50))
        self.action_dim = int(self.metadata.get("action_dim", 32))
        self.noise_rng = None if noise_seed is None else np.random.default_rng(noise_seed)
        self.last_noise: np.ndarray | None = None
        self.raw_frontend_name = str(self.metadata.get("raw_frontend", "none"))
        self.raw_frontend_rnn_frames = int(self.metadata.get("raw_frontend_rnn_frames", 1))
        self._raw_recurrent_history: list[collections.deque[np.ndarray]] = []
        log.info("server metadata: %s", self.metadata)
        log.info(
            "using %s norm for %s with model_source=%s",
            "quantile" if self.use_quantile_norm else "zscore",
            self.model_name,
            self.model_source,
        )

    def reset(self) -> None:
        self._raw_recurrent_history = []

    def _with_raw_frontend_history(self, example: dict[str, Any]) -> dict[str, Any]:
        bursts = example.get("raw_burst_float32")
        if self.raw_frontend_name not in {"rawvla", "raw-vla"} or bursts is None:
            return example
        if not self._raw_recurrent_history:
            self._raw_recurrent_history = [
                collections.deque(maxlen=self.raw_frontend_rnn_frames) for _ in bursts
            ]
        packed = []
        for history, burst in zip(self._raw_recurrent_history, bursts, strict=True):
            history.append(np.asarray(burst, dtype=np.float32))
            frames = list(history)
            frames = [frames[0]] * (self.raw_frontend_rnn_frames - len(frames)) + frames
            packed.append(np.stack(frames, axis=0))
        return {**example, "raw_burst_float32": packed}

    def predict_env_actions(self, example: dict[str, Any]) -> np.ndarray:
        example = self._with_raw_frontend_history(example)
        raw_state = example["raw_state"]
        model_input = {
            "image": example["image"],
            "lang": example["lang"],
            "state": normalize_openpi_value(
                raw_state,
                self.norm_stats["state"],
                self.use_quantile_norm,
            ).astype(np.float32)[None],
        }
        if "raw_burst_float32" in example:
            model_input["raw_burst_float32"] = example["raw_burst_float32"]
        payload = {"examples": [model_input]}
        self.last_noise = None
        if self.noise_rng is not None:
            self.last_noise = self.noise_rng.standard_normal(
                (self.action_horizon, self.action_dim),
                dtype=np.float32,
            )
            payload["noise"] = self.last_noise
        response = self.policy.predict_action(payload)
        if not response.get("ok", False):
            raise RuntimeError(f"policy server error: {response}")
        normalized_actions = np.asarray(response["data"]["normalized_actions"][0], dtype=np.float32)
        actions = postprocess_openpi_actions(
            normalized_actions=normalized_actions,
            raw_state=raw_state,
            norm_stats=self.norm_stats,
            model_name=self.model_name,
            model_source=self.model_source,
        )
        self.last_action_stats = {
            "normalized_actions": normalized_actions,
            "normalized_min": float(normalized_actions.min()),
            "normalized_max": float(normalized_actions.max()),
            "normalized_mean": float(normalized_actions.mean()),
            "normalized_std": float(normalized_actions.std()),
            "env_min": float(actions.min()),
            "env_max": float(actions.max()),
            "env_mean": float(actions.mean()),
            "env_std": float(actions.std()),
            "first_env_action": actions[0].tolist(),
        }
        return actions


@dataclasses.dataclass
class EvalArgs:
    model: str
    host: str = "127.0.0.1"
    port: int = 18000
    assets_checkpoint: Path | None = None
    libero_home: Path = DEFAULT_LIBERO_HOME
    task_suite: str = "libero_spatial"
    num_trials: int = 1
    max_tasks: int = 1
    task_start: int = 0
    task_end: int = -1
    num_steps_wait: int = 10
    replan_steps: int = 5
    resize_size: int = 224
    seed: int = 7
    result_json: Path | None = None
    log_action_stats: bool = True
    debug_dump_dir: Path | None = None
    noise_seed: int | None = None
    model_source: str = OPENPI_MODEL_SOURCE
    image_mode: str = "rgb"
    exposure_ev: float = 0.0
    ev_representation: str = "raw_direct"
    image_quantize_bits: int = 0
    chromatic_family: str = "none"
    chromatic_setting: str = ""
    chromatic_theta_deg: int = 0
    noise_capture_ev: float = -2.0
    tonal_mode: str = "raw_reprocess_isp_tone"
    tonal_setting: str = ""
    tonal_c: float = 1.0
    tonal_pivot: float = 0.18
    rawvla_bench_manifest: Path | None = None
    rawvla_bench_representation: str = "raw"
    raw_sensor_noise_seed: int = 1695213855
    resume_path: Path | None = None


def _load_episode_ledger(path: Path | None, suite: str) -> dict[tuple[int, int], bool]:
    completed = {}
    if path is None or not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            if record.get("task_suite") == suite:
                completed[(int(record["task_id"]), int(record["episode_idx"]))] = bool(record["success"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return completed


def _append_episode(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(args: EvalArgs) -> dict[str, Any]:
    np.random.seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(args.libero_home / "libero"))
    if str(args.libero_home) not in sys.path:
        sys.path.insert(0, str(args.libero_home))

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    rawvla_apply_entry_lighting = None
    rawvla_load_episode_plan = None
    rawvla_make_observation = None
    rawvla_prepare_entry_lighting = None
    if args.image_mode in {"rawvla_bench", "ev_noisy_rgb_direct"}:
        from rawvla_bench.raw import make_rawvla_observation

        rawvla_make_observation = make_rawvla_observation
    if args.image_mode == "rawvla_bench":
        if args.rawvla_bench_manifest is None:
            raise ValueError("image_mode='rawvla_bench' requires --rawvla-bench-manifest")
        if args.rawvla_bench_representation not in {"raw", "default_isp"}:
            raise ValueError(
                "rawvla_bench_representation must be one of {'raw', 'default_isp'}, "
                f"got {args.rawvla_bench_representation!r}"
            )
        from rawvla_bench.libero import apply_entry_lighting, load_episode_plan
        from rawvla_bench.lighting import rawvla_light_xml_from_env

        rawvla_apply_entry_lighting = apply_entry_lighting
        rawvla_load_episode_plan = load_episode_plan
        rawvla_prepare_entry_lighting = rawvla_light_xml_from_env

    model_name = canonicalize_model_name(args.model)
    client = StarVLAOpenPIClient(
        args.host,
        args.port,
        model_name,
        noise_seed=args.noise_seed,
        assets_checkpoint=args.assets_checkpoint,
        model_source=args.model_source,
    )

    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    num_tasks = task_suite.n_tasks
    task_end = num_tasks if args.task_end <= 0 else min(args.task_end, num_tasks)
    task_start = max(0, args.task_start)
    if args.max_tasks > 0:
        task_end = min(task_end, task_start + args.max_tasks)
    max_steps = get_max_steps(args.task_suite)
    replan_steps = int(args.replan_steps)

    total_episodes = 0
    total_successes = 0
    per_task: dict[str, dict[str, int]] = {}
    dumped_first_action = False
    completed = _load_episode_ledger(args.resume_path, args.task_suite)
    rawvla_plans: dict[int, list[dict[str, Any]]] = {}
    if args.image_mode == "rawvla_bench":
        assert rawvla_load_episode_plan is not None
        for task_id in range(task_start, task_end):
            rawvla_plans[task_id] = rawvla_load_episode_plan(
                str(args.rawvla_bench_manifest),
                args.task_suite,
                task_id,
            )

    task_iter = tqdm.tqdm(
        range(task_start, task_end),
        desc=f"{model_name} {args.task_suite}",
        dynamic_ncols=True,
    )
    for task_id in task_iter:
        task = task_suite.get_task(task_id)
        task_description = task.language
        initial_states = task_suite.get_task_init_states(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)
        task_num_trials = len(rawvla_plans[task_id]) if args.image_mode == "rawvla_bench" else args.num_trials
        task_completed = {
            ep: completed[(task_id, ep)]
            for ep in range(task_num_trials)
            if (task_id, ep) in completed
        }
        task_episodes = len(task_completed)
        task_successes = sum(task_completed.values())
        total_episodes += task_episodes
        total_successes += task_successes
        log.info("[task %s/%s] %s", task_id + 1, num_tasks, task_description)

        ep_iter = tqdm.tqdm(
            range(task_num_trials),
            desc=f"task {task_id + 1}/{num_tasks}",
            leave=False,
            dynamic_ncols=True,
        )
        for ep_idx in ep_iter:
            if ep_idx in task_completed:
                continue
            rawvla_entry = rawvla_plans[task_id][ep_idx] if args.image_mode == "rawvla_bench" else None
            if rawvla_entry is not None:
                env.seed(int(rawvla_entry["base_seed"]))
                env.reset()
                assert rawvla_prepare_entry_lighting is not None
                rawvla_xml = rawvla_prepare_entry_lighting(env)
                env.reset_from_xml_string(rawvla_xml)
                env.sim.reset()
                if hasattr(env, "_rawvla_bench_lighting_baseline"):
                    delattr(env, "_rawvla_bench_lighting_baseline")
            else:
                env.reset()
            if rawvla_entry is not None:
                assert rawvla_apply_entry_lighting is not None
                lighting_scale = rawvla_apply_entry_lighting(env, rawvla_entry)
                log.info(
                    "RAWVLA-Bench task=%d episode=%d base_seed=%s init_state=%s domain=%s ev=%.4f scale=%.6f noise_seed=%s",
                    task_id,
                    ep_idx,
                    rawvla_entry["base_seed"],
                    rawvla_entry["init_state_index"],
                    rawvla_entry["lighting_domain"],
                    float(rawvla_entry["lighting_ev"]),
                    lighting_scale,
                    rawvla_entry["noise_seed"],
                )
            init_state_index = int(rawvla_entry["init_state_index"]) if rawvla_entry is not None else ep_idx
            obs = env.set_init_state(initial_states[init_state_index])
            raw_params = None
            if args.image_mode == "raw_rgb10":
                raw_params = [
                    sample_raw_rgb10_params(episode_view_seed(args.seed, task_id, ep_idx, view_idx))
                    for view_idx in range(2)
                ]
                log.info(
                    "RAW RGB10 params task=%d episode=%d primary=%s wrist=%s",
                    task_id,
                    ep_idx,
                    params_dict(raw_params[0]),
                    params_dict(raw_params[1]),
                )
            elif args.image_mode in {"ev_float32", "ev_noisy_rgb_direct", "chromatic", "noise_level", "tonal"}:
                raw_params = [sample_raw_rgb10_params(0), sample_raw_rgb10_params(0)]
            elif args.image_mode == "rawvla_bench":
                raw_params = [sample_raw_rgb10_params(0), sample_raw_rgb10_params(0)]
            elif args.image_mode not in {"rgb", "black"}:
                raise ValueError(f"Unknown image_mode={args.image_mode!r}")
            action_plan: collections.deque[np.ndarray] = collections.deque()
            client.reset()
            done = False
            t = 0
            logged_action_stats = False
            raw_frame_history = None

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                example, _ = preprocess_env_obs(obs, task_description, args.resize_size)
                if args.image_mode == "black":
                    example["image"] = [
                        np.zeros((args.resize_size, args.resize_size, 3), dtype=np.float32)
                        for _ in example["image"]
                    ]
                elif raw_params is not None:
                    if args.image_mode == "raw_rgb10":
                        example["image"] = [
                            unprocess_rgb8_to_raw_rgb10_view8(example["image"][view_idx], raw_params[view_idx])
                            for view_idx in range(2)
                        ]
                    elif args.image_mode == "ev_float32":
                        example["image"] = [
                            make_ev_model_input(
                                example["image"][view_idx],
                                raw_params[view_idx],
                                args.exposure_ev,
                                args.ev_representation,
                            )
                            for view_idx in range(2)
                        ]
                    elif args.image_mode == "ev_noisy_rgb_direct":
                        assert rawvla_make_observation is not None
                        example["image"] = [
                            rawvla_make_observation(
                                image_tools.convert_to_uint8(example["image"][view_idx]),
                                noise_seed=args.raw_sensor_noise_seed,
                                frame_index=t - args.num_steps_wait,
                                view_index=view_idx,
                                representation="default_isp",
                                exposure_ev=args.exposure_ev,
                            )
                            for view_idx in range(2)
                        ]
                    elif args.image_mode == "chromatic":
                        example["image"] = [
                            make_chromatic_model_input(
                                example["image"][view_idx],
                                raw_params[view_idx],
                                args.chromatic_family,
                                args.chromatic_setting,
                                args.chromatic_theta_deg,
                            )
                            for view_idx in range(2)
                        ]
                    elif args.image_mode == "noise_level":
                        example["image"] = [
                            make_noise_level_model_input(
                                image_tools.convert_to_uint8(example["image"][view_idx]),
                                raw_params[view_idx],
                                args.noise_capture_ev,
                                noise_seed=episode_view_seed(args.raw_sensor_noise_seed, task_id, ep_idx, view_idx),
                                frame_index=t - args.num_steps_wait,
                                view_index=view_idx,
                            )
                            for view_idx in range(2)
                        ]
                    elif args.image_mode == "tonal":
                        example["image"] = [
                            make_tonal_model_input(
                                example["image"][view_idx],
                                raw_params[view_idx],
                                args.tonal_mode,
                                args.tonal_setting,
                                args.tonal_c,
                                args.tonal_pivot,
                            )
                            for view_idx in range(2)
                        ]
                    elif args.image_mode == "rawvla_bench":
                        assert rawvla_entry is not None
                        assert rawvla_make_observation is not None
                        raw_sources = [
                            np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
                            np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
                        ]
                        burst_frames = (
                            int(client.metadata.get("raw_frontend_burst_frames", 1))
                            if client.raw_frontend_name in {"rawvla", "raw-vla"}
                            else 1
                        )
                        if raw_frame_history is None:
                            raw_frame_history = [
                                collections.deque(maxlen=burst_frames) for _ in raw_sources
                            ]
                        raw_images = []
                        raw_burst_images = []
                        for view_idx, source in enumerate(raw_sources):
                            policy_step = t - args.num_steps_wait
                            current_raw = rawvla_make_observation(
                                source,
                                noise_seed=int(rawvla_entry["noise_seed"]),
                                frame_index=policy_step,
                                view_index=view_idx,
                                representation=args.rawvla_bench_representation,
                                sensor_saturation_ev=float(rawvla_entry.get("sensor_saturation_ev", 0.0)),
                            )
                            current_raw = image_tools.convert_to_uint8(
                                image_tools.resize_image(current_raw, args.resize_size, args.resize_size)
                            )
                            history = raw_frame_history[view_idx]
                            history.append(current_raw)
                            burst = list(history)
                            burst = [burst[0]] * (burst_frames - len(burst)) + burst
                            raw_burst_images.append(
                                np.stack(burst, axis=0).astype(np.float32) / 255.0
                            )
                            raw_images.append(burst[-1])
                        example["image"] = raw_images
                        if client.raw_frontend_name in {"rawvla", "raw-vla"}:
                            example["raw_burst_float32"] = raw_burst_images
                    else:
                        raise ValueError(f"Unknown image_mode={args.image_mode!r}")
                if args.image_quantize_bits > 0:
                    example["image"] = [
                        quantize_image_to_float_bits(image, args.image_quantize_bits)
                        for image in example["image"]
                    ]
                for image_idx, image in enumerate(example["image"]):
                    image_tools.audit_image_once(f"openpi_client_model_input_{image_idx}", image)
                if not action_plan:
                    actions = client.predict_env_actions(example)
                    if args.debug_dump_dir is not None and not dumped_first_action:
                        args.debug_dump_dir.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            args.debug_dump_dir / "first_policy_call.npz",
                            base_image=np.asarray(example["image"][0]),
                            wrist_image=np.asarray(example["image"][1]),
                            raw_state=np.asarray(example["raw_state"], dtype=np.float32),
                            normalized_state=normalize_openpi_value(
                                example["raw_state"],
                                client.norm_stats["state"],
                                client.use_quantile_norm,
                            ).astype(np.float32),
                            normalized_actions=np.asarray(
                                client.last_action_stats.get("normalized_actions", [])
                                if client.last_action_stats is not None
                                else [],
                                dtype=np.float32,
                            ),
                            noise=np.asarray([] if client.last_noise is None else client.last_noise, dtype=np.float32),
                            env_actions=actions,
                            first_env_action=actions[0],
                            task_description=np.asarray(str(task_description)),
                            model=np.asarray(model_name),
                            task_suite=np.asarray(args.task_suite),
                            task_id=np.asarray(task_id),
                            episode_idx=np.asarray(ep_idx),
                        )
                        log.info("  wrote debug dump: %s", args.debug_dump_dir / "first_policy_call.npz")
                        dumped_first_action = True
                    if args.log_action_stats and not logged_action_stats and client.last_action_stats is not None:
                        stats = client.last_action_stats
                        log.info(
                            "  ep=%s action_stats norm[min=%.3f max=%.3f mean=%.3f std=%.3f] "
                            "env[min=%.3f max=%.3f mean=%.3f std=%.3f] first=%s",
                            ep_idx,
                            stats["normalized_min"],
                            stats["normalized_max"],
                            stats["normalized_mean"],
                            stats["normalized_std"],
                            stats["env_min"],
                            stats["env_max"],
                            stats["env_mean"],
                            stats["env_std"],
                            np.array2string(np.asarray(stats["first_env_action"]), precision=4, suppress_small=True),
                        )
                        logged_action_stats = True
                    if len(actions) < replan_steps:
                        raise ValueError(f"replan_steps={replan_steps} but server returned {len(actions)} actions")
                    action_plan.extend(actions[:replan_steps])

                obs, _, done, _ = env.step(action_plan.popleft().tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

            task_episodes += 1
            total_episodes += 1
            _append_episode(
                args.resume_path,
                {
                    "task_suite": args.task_suite,
                    "task_id": task_id,
                    "task_description": task_description,
                    "episode_idx": ep_idx,
                    "success": bool(done),
                    "seed": args.seed,
                    "image_mode": args.image_mode,
                    "ev_representation": args.ev_representation,
                    "image_quantize_bits": args.image_quantize_bits,
                    "chromatic_family": args.chromatic_family,
                    "chromatic_setting": args.chromatic_setting,
                    "chromatic_theta_deg": args.chromatic_theta_deg,
                    "noise_capture_ev": args.noise_capture_ev,
                    "tonal_mode": args.tonal_mode,
                    "tonal_setting": args.tonal_setting,
                    "tonal_c": args.tonal_c,
                    "tonal_pivot": args.tonal_pivot,
                    "rawvla_bench_manifest": str(args.rawvla_bench_manifest) if args.rawvla_bench_manifest else None,
                    "rawvla_bench_representation": args.rawvla_bench_representation,
                    "raw_sensor_noise_seed": args.raw_sensor_noise_seed,
                    "rawvla_bench_entry": rawvla_entry,
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            task_sr = task_successes / max(task_episodes, 1)
            total_sr = total_successes / max(total_episodes, 1)
            ep_iter.set_postfix(
                task_sr=f"{task_successes}/{task_episodes} ({100.0 * task_sr:.1f}%)",
                total_sr=f"{total_successes}/{total_episodes} ({100.0 * total_sr:.1f}%)",
            )
            task_iter.set_postfix(total_sr=f"{total_successes}/{total_episodes} ({100.0 * total_sr:.1f}%)")
            log.info(
                "  ep=%s %s task_sr=%s/%s %.1f%% total_sr=%s/%s %.1f%%",
                ep_idx,
                "SUCCESS" if done else "fail",
                task_successes,
                task_episodes,
                100.0 * task_sr,
                total_successes,
                total_episodes,
                100.0 * total_sr,
            )

        per_task[task_description] = {"success": task_successes, "total": task_episodes}
        log.info(
            "[task %s/%s done] task_sr=%s/%s %.1f%% total_sr=%s/%s %.1f%%",
            task_id + 1,
            num_tasks,
            task_successes,
            task_episodes,
            100.0 * task_successes / max(task_episodes, 1),
            total_successes,
            total_episodes,
            100.0 * total_successes / max(total_episodes, 1),
        )
        env.close()

    summary = {
        "model": model_name,
        "host": args.host,
        "port": args.port,
        "task_suite": args.task_suite,
        "task_start": task_start,
        "task_end": task_end,
        "image_mode": args.image_mode,
        "exposure_ev": args.exposure_ev,
        "ev_representation": args.ev_representation,
        "image_quantize_bits": args.image_quantize_bits,
        "chromatic_family": args.chromatic_family,
        "chromatic_setting": args.chromatic_setting,
        "chromatic_theta_deg": args.chromatic_theta_deg,
        "noise_capture_ev": args.noise_capture_ev,
        "tonal_mode": args.tonal_mode,
        "tonal_setting": args.tonal_setting,
        "tonal_c": args.tonal_c,
        "tonal_pivot": args.tonal_pivot,
        "rawvla_bench_manifest": str(args.rawvla_bench_manifest) if args.rawvla_bench_manifest else None,
        "rawvla_bench_representation": args.rawvla_bench_representation,
        "raw_sensor_noise_seed": args.raw_sensor_noise_seed,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "success_rate": float(total_successes / max(total_episodes, 1)),
        "per_task": per_task,
    }
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("FINAL SR: %s/%s %.2f%%", total_successes, total_episodes, 100.0 * summary["success_rate"])
    return summary


def parse_args() -> EvalArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["PI0", "PI05", "pi0", "pi05"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--assets-checkpoint", type=Path, default=None)
    parser.add_argument("--libero-home", type=Path, default=DEFAULT_LIBERO_HOME)
    parser.add_argument("--task-suite", default="libero_spatial", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-end", type=int, default=-1)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument("--log-action-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-dump-dir", type=Path, default=None)
    parser.add_argument("--noise-seed", type=int, default=None)
    parser.add_argument("--model-source", default=OPENPI_MODEL_SOURCE, choices=[OPENPI_MODEL_SOURCE, STARVLA_MODEL_SOURCE])
    parser.add_argument(
        "--image-mode",
        default="rgb",
        choices=[
            "rgb",
            "raw_rgb10",
            "ev_float32",
            "ev_noisy_rgb_direct",
            "chromatic",
            "noise_level",
            "tonal",
            "rawvla_bench",
            "black",
        ],
    )
    parser.add_argument("--exposure-ev", type=float, default=0.0)
    parser.add_argument(
        "--ev-representation",
        default="raw_direct",
        choices=["raw_direct", "raw_recovered", "rgb_direct", "rgb_recovered"],
    )
    parser.add_argument("--image-quantize-bits", type=int, default=0)
    parser.add_argument("--chromatic-family", default="none", choices=["none", "uv_whitepoint", "color_relation"])
    parser.add_argument("--chromatic-setting", default="")
    parser.add_argument("--chromatic-theta-deg", type=int, default=0)
    parser.add_argument("--noise-capture-ev", type=float, default=-2.0)
    parser.add_argument("--tonal-mode", default="raw_reprocess_isp_tone", choices=["raw_reprocess_isp_tone"])
    parser.add_argument("--tonal-setting", default="")
    parser.add_argument("--tonal-c", type=float, default=1.0)
    parser.add_argument("--tonal-pivot", type=float, default=0.18)
    parser.add_argument("--rawvla-bench-manifest", type=Path, default=None)
    parser.add_argument("--rawvla-bench-representation", default="raw", choices=["raw", "default_isp"])
    parser.add_argument("--raw-sensor-noise-seed", type=int, default=1695213855)
    parser.add_argument("--resume-path", type=Path, default=None)
    ns = parser.parse_args()
    return EvalArgs(**vars(ns))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
