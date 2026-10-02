from __future__ import annotations

import dataclasses
import collections
import json
import logging
import math
import os
import pathlib
import sys
import time
from typing import Any

import cv2 as cv
import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model2libero_interface import ModelClient  # noqa: E402

OPENPI_DIR = pathlib.Path(__file__).resolve().parents[2] / "LIBERO" / "eval_files" / "openpi"
sys.path.insert(0, str(OPENPI_DIR))
from eval_starvla_openpi_client import StarVLAOpenPIClient  # noqa: E402
from pi_libero_common import canonicalize_model_name  # noqa: E402
from deployment.model_server.tools import image_tools  # noqa: E402

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size: tuple[int, int] = (224, 224)

    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    max_tasks: int = -1
    task_start: int = 0
    task_end: int = -1
    task_category: str = "Light Conditions"

    video_out_path: str = "experiments/libero_plus/logs"
    resume_path: str | None = None
    task_log_dir: str | None = None
    result_json: str | None = None

    seed: int = 7
    pretrained_path: str = ""
    model: str = ""
    model_source: str = "starvla"
    assets_checkpoint: str | None = None
    unnorm_key: str | None = None
    post_process_action: bool = True
    replan_steps: int = 5
    job_name: str = "libero_plus"

    phase: str = "light"
    image_effect: str = "auto"
    blur_kernel: int = 9
    blur_sigma: float = 2.0
    haze_strength: float = 0.35
    haze_airlight: float = 235.0
    image_quantize_bits: int = 0


def eval_libero_plus(args: Args) -> None:
    logging.info("Arguments: %s", json.dumps(dataclasses.asdict(args), indent=4))
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    max_steps = _max_steps(args.task_suite_name)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    task_meta = _load_task_meta(args.task_suite_name)
    selected_task_ids = _select_task_ids(args, num_tasks_in_suite, task_meta)
    if not selected_task_ids:
        logging.warning("No tasks selected for suite=%s category=%r", args.task_suite_name, args.task_category)
        _write_result_json(args, 0, 0, len(selected_task_ids), {})
        return

    effect = _resolve_effect(args)
    logging.info(
        "Task suite=%s selected=%d/%d category=%r phase=%s effect=%s",
        args.task_suite_name,
        len(selected_task_ids),
        num_tasks_in_suite,
        args.task_category,
        args.phase,
        effect,
    )

    use_openpi = args.model_source.lower() == "openpi"
    if use_openpi:
        client_model: Any = StarVLAOpenPIClient(
            args.host,
            args.port,
            canonicalize_model_name(args.model),
            assets_checkpoint=pathlib.Path(args.assets_checkpoint) if args.assets_checkpoint else None,
            model_source=args.model_source,
        )
    else:
        client_model = ModelClient(
            policy_ckpt_path=args.pretrained_path,
            unnorm_key=args.unnorm_key,
            host=args.host,
            port=args.port,
            image_size=list(args.resize_size),
        )

    completed = _load_episode_ledger(args.resume_path, args.task_suite_name, args.phase)
    target_keys = {
        (task_id, episode_idx)
        for task_id in selected_task_ids
        for episode_idx in range(args.num_trials_per_task)
    }
    completed = {key: value for key, value in completed.items() if key in target_keys}
    total_episodes = len(completed)
    total_successes = sum(completed.values())
    per_category = _initial_category_counts(task_meta, selected_task_ids)
    _apply_completed_to_categories(per_category, completed, task_meta)
    logging.info(
        "Resume ledger contains %d/%d completed episodes (%d successes)",
        total_episodes,
        len(target_keys),
        total_successes,
    )

    for task_id in tqdm.tqdm(selected_task_ids):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_completed = {
            episode_idx: completed[(task_id, episode_idx)]
            for episode_idx in range(args.num_trials_per_task)
            if (task_id, episode_idx) in completed
        }
        if len(task_completed) == args.num_trials_per_task:
            logging.info("Skipping completed task_id=%d", task_id)
            continue

        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        task_handler = _add_task_log_handler(args.task_log_dir, task_id)
        task_episodes = len(task_completed)
        task_successes = sum(task_completed.values())

        try:
            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                if episode_idx in task_completed:
                    continue
                logging.info("Task: %s", task_description)
                if not use_openpi:
                    client_model.reset(task_description=task_description)
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

                t = 0
                step = 0
                replay_images = []
                full_actions = []
                action_plan: collections.deque[np.ndarray] = collections.deque()
                done = False
                logging.info("Starting task_id=%d episode=%d", task_id, episode_idx)

                while t < max_steps + args.num_steps_wait:
                    if t < args.num_steps_wait:
                        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = _apply_image_effect(img, effect, args)
                    wrist_img = _apply_image_effect(wrist_img, effect, args)
                    if args.image_quantize_bits > 0:
                        img = _quantize_image(img, args.image_quantize_bits)
                        wrist_img = _quantize_image(wrist_img, args.image_quantize_bits)
                    replay_images.append(img)

                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )
                    if use_openpi:
                        if not action_plan:
                            example_dict = {
                                "image": [_resize_uint8_image(img, args.resize_size[0]), _resize_uint8_image(wrist_img, args.resize_size[0])],
                                "lang": str(task_description),
                                "raw_state": np.asarray(state, dtype=np.float32),
                            }
                            actions = client_model.predict_env_actions(example_dict)
                            if len(actions) < args.replan_steps:
                                raise ValueError(
                                    f"replan_steps={args.replan_steps} but server returned {len(actions)} actions"
                                )
                            action_plan.extend(actions[: args.replan_steps])
                        delta_action = np.asarray(action_plan.popleft(), dtype=np.float32)[:7]
                    else:
                        example_dict = {"image": [img, wrist_img], "lang": str(task_description)}
                        response = client_model.step(example=example_dict, step=step)
                        raw_action = response["raw_action"]
                        world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                        rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                        open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                        gripper = _binarize_gripper_open(open_gripper)

                        if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                            raise ValueError(
                                "Invalid action sizes: "
                                f"world_vector={world_vector_delta.shape}, "
                                f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                            )
                        delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)
                    full_actions.append(delta_action)

                    obs, _, done, _ = env.step(delta_action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        per_category[_task_category(task_meta, task_id)]["success_count"] += 1
                        break
                    t += 1
                    step += 1

                task_episodes += 1
                total_episodes += 1
                per_category[_task_category(task_meta, task_id)]["episode_count"] += 1
                _save_video(args, task_meta, task_id, episode_idx, done, replay_images)
                if full_actions:
                    np.stack(full_actions)

                record = {
                    "task_suite": args.task_suite_name,
                    "task_id": task_id,
                    "libero_plus_id": task_id + 1,
                    "task_name": _task_name(task_meta, task_id),
                    "task_category": _task_category(task_meta, task_id),
                    "task_description": task_description,
                    "episode_idx": episode_idx,
                    "success": bool(done),
                    "seed": args.seed,
                    "phase": args.phase,
                    "model_source": args.model_source,
                    "model": args.model,
                    "image_effect": effect,
                    "blur_kernel": args.blur_kernel,
                    "blur_sigma": args.blur_sigma,
                    "haze_strength": args.haze_strength,
                    "haze_airlight": args.haze_airlight,
                    "image_quantize_bits": args.image_quantize_bits,
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                _append_episode_ledger(args.resume_path, record)
                logging.info("Success: %s", done)
                logging.info(
                    "Completed episodes=%d successes=%d success_rate=%.4f",
                    total_episodes,
                    total_successes,
                    total_successes / max(total_episodes, 1),
                )
                _write_result_json(args, total_episodes, total_successes, len(target_keys), per_category)
        finally:
            env.close()
            if task_handler is not None:
                logging.getLogger().removeHandler(task_handler)
                task_handler.close()

        logging.info(
            "Current task success rate: %.4f",
            float(task_successes) / float(max(task_episodes, 1)),
        )

    _write_result_json(args, total_episodes, total_successes, len(target_keys), per_category)
    logging.info("Total success rate: %.4f", float(total_successes) / float(max(total_episodes, 1)))
    logging.info("Total episodes: %d", total_episodes)


def _resolve_effect(args: Args) -> str:
    if args.image_effect != "auto":
        return args.image_effect
    if args.phase in {"light", "none"}:
        return "none"
    if args.phase in {"blur", "haze"}:
        return args.phase
    raise ValueError(f"Unsupported phase={args.phase!r}")


def _apply_image_effect(image: np.ndarray, effect: str, args: Args) -> np.ndarray:
    if effect == "none":
        return image
    if effect == "blur":
        kernel = max(3, int(args.blur_kernel))
        if kernel % 2 == 0:
            kernel += 1
        return cv.GaussianBlur(image, (kernel, kernel), sigmaX=args.blur_sigma)
    if effect == "haze":
        strength = float(np.clip(args.haze_strength, 0.0, 1.0))
        air = float(np.clip(args.haze_airlight, 0.0, 255.0))
        out = image.astype(np.float32) * (1.0 - strength) + air * strength
        return np.clip(out, 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported image_effect={effect!r}")


def _quantize_image(image: np.ndarray, bits: int) -> np.ndarray:
    levels = 2**bits - 1
    if levels <= 0:
        return image
    arr = image.astype(np.float32) / 255.0
    return np.clip(np.round(arr * levels) / levels * 255.0, 0, 255).astype(np.uint8)


def _resize_uint8_image(image: np.ndarray, size: int) -> np.ndarray:
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image[None], size, size)[0])


def _load_task_meta(task_suite_name: str) -> dict[int, dict]:
    libero_home = os.environ.get("LIBERO_HOME")
    if not libero_home:
        raise RuntimeError("LIBERO_HOME must point to the LIBERO-plus checkout.")
    path = pathlib.Path(libero_home) / "libero/libero/benchmark/task_classification.json"
    with open(path, encoding="utf-8") as f:
        task_mapping = json.load(f)[task_suite_name]
    return {int(item["id"]) - 1: item for item in task_mapping}


def _select_task_ids(args: Args, num_tasks_in_suite: int, task_meta: dict[int, dict]) -> list[int]:
    start = max(0, args.task_start)
    end = num_tasks_in_suite if args.task_end <= 0 else min(args.task_end, num_tasks_in_suite)
    task_ids = list(range(start, end))
    if args.task_category:
        task_ids = [task_id for task_id in task_ids if _task_category(task_meta, task_id) == args.task_category]
    if args.max_tasks > 0:
        task_ids = task_ids[: args.max_tasks]
    return task_ids


def _initial_category_counts(task_meta: dict[int, dict], selected_task_ids: list[int]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for task_id in selected_task_ids:
        category = _task_category(task_meta, task_id)
        counts.setdefault(category, {"task_count": 0, "episode_count": 0, "success_count": 0})
        counts[category]["task_count"] += 1
    return counts


def _apply_completed_to_categories(
    per_category: dict[str, dict[str, int]], completed: dict[tuple[int, int], bool], task_meta: dict[int, dict]
) -> None:
    for (task_id, _), success in completed.items():
        category = _task_category(task_meta, task_id)
        per_category.setdefault(category, {"task_count": 0, "episode_count": 0, "success_count": 0})
        per_category[category]["episode_count"] += 1
        per_category[category]["success_count"] += int(success)


def _task_category(task_meta: dict[int, dict], task_id: int) -> str:
    return str(task_meta.get(task_id, {}).get("category", "Unknown"))


def _task_name(task_meta: dict[int, dict], task_id: int) -> str:
    return str(task_meta.get(task_id, {}).get("name", f"task_{task_id:04d}"))


def _write_result_json(
    args: Args, total_episodes: int, total_successes: int, target_episodes: int, per_category: dict[str, dict[str, int]]
) -> None:
    if not args.result_json:
        return
    result_path = pathlib.Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "task_suite": args.task_suite_name,
        "task_start": args.task_start,
        "task_end": args.task_end,
        "task_category": args.task_category,
        "phase": args.phase,
        "image_effect": _resolve_effect(args),
        "total_episodes": total_episodes,
        "target_episodes": target_episodes,
        "total_successes": int(total_successes),
        "success_rate": float(total_successes) / float(max(total_episodes, 1)),
        "complete": total_episodes >= target_episodes,
        "per_category": per_category,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp_path = result_path.with_suffix(result_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(result_path)


def _load_episode_ledger(path: str | None, task_suite_name: str, phase: str) -> dict[tuple[int, int], bool]:
    completed: dict[tuple[int, int], bool] = {}
    if not path or not pathlib.Path(path).exists():
        return completed
    with open(path, encoding="utf-8") as ledger:
        for line_number, line in enumerate(ledger, 1):
            try:
                record = json.loads(line)
                if record.get("task_suite") != task_suite_name or record.get("phase") != phase:
                    continue
                key = (int(record["task_id"]), int(record["episode_idx"]))
                completed[key] = bool(record["success"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                logging.warning("Ignoring invalid resume record %s:%d: %s", path, line_number, exc)
    return completed


def _append_episode_ledger(path: str | None, record: dict) -> None:
    if not path:
        return
    ledger_path = pathlib.Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(record, sort_keys=True) + "\n")
        ledger.flush()
        os.fsync(ledger.fileno())


def _add_task_log_handler(log_dir: str | None, task_id: int) -> logging.Handler | None:
    if not log_dir:
        return None
    path = pathlib.Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path / f"task_{task_id:04d}.log", mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s | %(message)s", "%m/%d [%H:%M:%S]"))
    logging.getLogger().addHandler(handler)
    return handler


def _save_video(args: Args, task_meta: dict[int, dict], task_id: int, episode_idx: int, done: bool, images: list) -> None:
    if os.environ.get("LIBERO_SKIP_VIDEO", "0") == "1":
        return
    suffix = "success" if done else "failure"
    task_segment = _task_name(task_meta, task_id).replace(" ", "_")
    video_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4"
    try:
        imageio.mimwrite(video_path, [np.asarray(x) for x in images], fps=25)
    except Exception as exc:
        logging.warning("Failed to write replay video %s: %s", video_path, exc)


def _max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    eval_libero_plus(tyro.cli(Args))
