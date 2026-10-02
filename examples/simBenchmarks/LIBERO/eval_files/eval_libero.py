import dataclasses
from collections import deque
import json
import logging
import math
import os
import pathlib
import sys
import time

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.simBenchmarks.LIBERO.eval_files.model2libero_interface import ModelClient
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

RAWVLA_BENCH_ROOT = pathlib.Path(__file__).resolve().parents[5] / "benchmark" / "rawvla-bench"
if RAWVLA_BENCH_ROOT.exists() and str(RAWVLA_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RAWVLA_BENCH_ROOT))

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def _as_float32_image_channel(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.float32:
        return np.ascontiguousarray(arr, dtype=np.float32)
    return np.ascontiguousarray(arr.astype(np.float32) / 255.0, dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int = -1  # If > 0, limit the number of tasks evaluated (smoke / quick check). -1 = run all.
    task_start: int = 0  # Inclusive task index for chunked/resumed evaluation.
    task_end: int = -1  # Exclusive task index; -1 means the end of the suite.
    resume_path: str | None = None  # JSONL episode ledger; completed episodes are skipped on restart.
    task_log_dir: str | None = None  # Optional directory for one append-only log per task.
    max_new_episodes_per_run: int = -1  # If > 0, return cleanly after this many new ledger entries.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    # Dataset key for un-normalization. None = auto (only if model trained on a single dataset).
    unnorm_key: str | None = None

    post_process_action: bool = True

    job_name: str = "test"

    # Input image domain. raw_rgb10 is three-channel linear pseudo-RAW:
    # no Bayer mosaic/CCM/noise, RGB10 quantization, then linear uint8 mapping.
    image_mode: str = "rgb"
    exposure_ev: float = 0.0
    ev_representation: str = "raw_direct"
    image_quantize_bits: int = 0
    chromatic_family: str = "none"
    chromatic_setting: str = ""
    chromatic_theta_deg: int = 0
    noise_capture_ev: float = -2.0
    raw_sensor_noise_seed: int = 1695213855
    tonal_mode: str = "raw_reprocess_isp_tone"
    tonal_setting: str = ""
    tonal_c: float = 1.0
    tonal_pivot: float = 0.18

    # RAWVLA-Bench uses simulator-side lighting from a frozen manifest, then
    # fixed unprocess + fixed shot/read RAW noise.
    rawvla_bench_manifest: str | None = None
    rawvla_bench_representation: str = "raw"
    rawvla_frontend_export_dir: str | None = None
    rawvla_frontend_export_seed: int = 20260823


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    rawvla_apply_entry_lighting = None
    rawvla_load_episode_plan = None
    rawvla_make_observation = None
    rawvla_prepare_entry_lighting = None
    if args.rawvla_bench_manifest:
        from rawvla_bench.libero import load_episode_plan

        rawvla_load_episode_plan = load_episode_plan
    if args.image_mode == "rawvla_bench":
        if not args.rawvla_bench_manifest:
            raise ValueError("image_mode='rawvla_bench' requires --args.rawvla-bench-manifest")
        if args.rawvla_bench_representation not in {"raw", "default_isp"}:
            raise ValueError(
                "rawvla_bench_representation must be one of {'raw', 'default_isp'}, "
                f"got {args.rawvla_bench_representation!r}"
            )
        from rawvla_bench.libero import apply_entry_lighting
        from rawvla_bench.lighting import rawvla_light_xml_from_env
        from rawvla_bench.raw import make_rawvla_observation

        rawvla_apply_entry_lighting = apply_entry_lighting
        rawvla_make_observation = make_rawvla_observation
        rawvla_prepare_entry_lighting = rawvla_light_xml_from_env

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
    )

    task_start = max(0, args.task_start)
    task_end = num_tasks_in_suite if args.task_end <= 0 else min(args.task_end, num_tasks_in_suite)
    if args.max_tasks > 0:
        task_end = min(task_end, task_start + args.max_tasks)
    if task_start >= task_end:
        logging.info(
            "No tasks selected for suite %s: task_start=%d task_end=%d max_tasks=%d",
            args.task_suite_name,
            task_start,
            task_end,
            args.max_tasks,
        )
        return
    logging.info(
        "Evaluating tasks [%d, %d) of %d (max_tasks=%d)",
        task_start,
        task_end,
        num_tasks_in_suite,
        args.max_tasks,
    )

    rawvla_plans: dict[int, list[dict]] = {}
    if args.rawvla_bench_manifest:
        assert rawvla_load_episode_plan is not None
        for task_id in range(task_start, task_end):
            rawvla_plans[task_id] = rawvla_load_episode_plan(
                args.rawvla_bench_manifest,
                args.task_suite_name,
                task_id,
            )

    completed = _load_episode_ledger(args.resume_path, args.task_suite_name)
    target_keys = {
        (task_id, episode_idx)
        for task_id in range(task_start, task_end)
        for episode_idx in range(len(rawvla_plans[task_id]) if args.rawvla_bench_manifest else args.num_trials_per_task)
    }
    completed = {key: value for key, value in completed.items() if key in target_keys}
    total_episodes = len(completed)
    total_successes = sum(completed.values())
    new_episodes_this_run = 0
    logging.info(
        "Resume ledger contains %d/%d completed episodes (%d successes)",
        total_episodes,
        len(target_keys),
        total_successes,
    )

    # Start evaluation
    for task_id in tqdm.tqdm(range(task_start, task_end)):
        # Get task
        task = task_suite.get_task(task_id)

        task_completed = {
            episode_idx: completed[(task_id, episode_idx)]
            for episode_idx in range(len(rawvla_plans[task_id]) if args.rawvla_bench_manifest else args.num_trials_per_task)
            if (task_id, episode_idx) in completed
        }
        task_episodes = len(task_completed)
        task_successes = sum(task_completed.values())
        task_num_trials = len(rawvla_plans[task_id]) if args.rawvla_bench_manifest else args.num_trials_per_task
        if task_episodes == task_num_trials:
            logging.info(
                "Skipping completed task %d (%d/%d successes)",
                task_id,
                task_successes,
                task_episodes,
            )
            continue

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        task_handler = _add_task_log_handler(args.task_log_dir, task_id)

        # Start episodes
        for episode_idx in tqdm.tqdm(range(task_num_trials)):
            if episode_idx in task_completed:
                continue
            rawvla_entry = rawvla_plans[task_id][episode_idx] if args.rawvla_bench_manifest else None
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            client_model.reset(task_description=task_description)  # Reset the client connection
            if rawvla_entry is not None and args.image_mode == "rawvla_bench":
                env.seed(int(rawvla_entry["base_seed"]))
                env.reset()
                assert rawvla_prepare_entry_lighting is not None
                rawvla_xml = rawvla_prepare_entry_lighting(env)
                env.reset_from_xml_string(rawvla_xml)
                env.sim.reset()
                if hasattr(env, "_rawvla_bench_lighting_baseline"):
                    delattr(env, "_rawvla_bench_lighting_baseline")
            else:
                if rawvla_entry is not None:
                    env.seed(int(rawvla_entry["base_seed"]))
                env.reset()
            if rawvla_entry is not None and args.image_mode == "rawvla_bench":
                assert rawvla_apply_entry_lighting is not None
                lighting_scale = rawvla_apply_entry_lighting(env, rawvla_entry)
                logging.info(
                    "RAWVLA-Bench task=%d episode=%d base_seed=%s init_state=%s domain=%s ev=%.4f scale=%.6f noise_seed=%s",
                    task_id,
                    episode_idx,
                    rawvla_entry["base_seed"],
                    rawvla_entry["init_state_index"],
                    rawvla_entry["lighting_domain"],
                    float(rawvla_entry["lighting_ev"]),
                    lighting_scale,
                    rawvla_entry["noise_seed"],
                )
            elif rawvla_entry is not None:
                logging.info(
                    "Manifest-plan RGB task=%d episode=%d base_seed=%s init_state=%s",
                    task_id,
                    episode_idx,
                    rawvla_entry["base_seed"],
                    rawvla_entry["init_state_index"],
                )

            # Set initial states
            init_state_index = int(rawvla_entry["init_state_index"]) if rawvla_entry is not None else episode_idx
            obs = env.set_init_state(initial_states[init_state_index])

            raw_params = None
            if args.image_mode == "raw_rgb10":
                raw_params = [
                    sample_raw_rgb10_params(episode_view_seed(args.seed, task_id, episode_idx, view_idx))
                    for view_idx in range(2)
                ]
                logging.info(
                    "RAW RGB10 params task=%d episode=%d primary=%s wrist=%s",
                    task_id,
                    episode_idx,
                    params_dict(raw_params[0]),
                    params_dict(raw_params[1]),
                )
            elif args.image_mode in {"ev_float32", "chromatic", "noise_level", "tonal"}:
                raw_params = [sample_raw_rgb10_params(0), sample_raw_rgb10_params(0)]
            elif args.image_mode == "rawvla_bench":
                raw_params = [sample_raw_rgb10_params(0), sample_raw_rgb10_params(0)]
            elif args.image_mode != "rgb":
                raise ValueError(f"Unknown image_mode={args.image_mode!r}")

            # Setup
            t = 0
            replay_images = []
            full_actions = []
            raw_frame_history = None

            logging.info(f"Starting task {task_id} episode {episode_idx + 1}...")
            step = 0
            rawvla_export_step = None
            if args.rawvla_frontend_export_dir:
                inference_steps = np.arange(0, max_steps, client_model.action_chunk_size)
                export_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [args.rawvla_frontend_export_seed, task_id, episode_idx]
                    )
                )
                rawvla_export_step = int(export_rng.choice(inference_steps))
                logging.info(
                    "RAWVLA frontend export task=%d episode=%d init_state=%s step=%d dir=%s",
                    task_id,
                    episode_idx,
                    rawvla_entry.get("init_state_index") if rawvla_entry else None,
                    rawvla_export_step,
                    args.rawvla_frontend_export_dir,
                )

            # full_actions = np.load("./debug/action.npy")

            while t < max_steps + args.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                raw_burst_images = None

                if raw_params is not None:
                    if args.image_mode == "raw_rgb10":
                        img = np.asarray(Image.fromarray(img).resize((224, 224), Image.BILINEAR))
                        wrist_img = np.asarray(Image.fromarray(wrist_img).resize((224, 224), Image.BILINEAR))
                        img = unprocess_rgb8_to_raw_rgb10_view8(img, raw_params[0])
                        wrist_img = unprocess_rgb8_to_raw_rgb10_view8(wrist_img, raw_params[1])
                    elif args.image_mode == "ev_float32":
                        img = np.asarray(Image.fromarray(img).resize((224, 224), Image.BILINEAR))
                        wrist_img = np.asarray(Image.fromarray(wrist_img).resize((224, 224), Image.BILINEAR))
                        img = make_ev_model_input(img, raw_params[0], args.exposure_ev, args.ev_representation)
                        wrist_img = make_ev_model_input(
                            wrist_img, raw_params[1], args.exposure_ev, args.ev_representation
                        )
                    elif args.image_mode == "chromatic":
                        img = np.asarray(Image.fromarray(img).resize((224, 224), Image.BILINEAR))
                        wrist_img = np.asarray(Image.fromarray(wrist_img).resize((224, 224), Image.BILINEAR))
                        img = make_chromatic_model_input(
                            img,
                            raw_params[0],
                            args.chromatic_family,
                            args.chromatic_setting,
                            args.chromatic_theta_deg,
                        )
                        wrist_img = make_chromatic_model_input(
                            wrist_img,
                            raw_params[1],
                            args.chromatic_family,
                            args.chromatic_setting,
                            args.chromatic_theta_deg,
                        )
                    elif args.image_mode == "noise_level":
                        img = np.asarray(Image.fromarray(img).resize((224, 224), Image.BILINEAR))
                        wrist_img = np.asarray(Image.fromarray(wrist_img).resize((224, 224), Image.BILINEAR))
                        img = make_noise_level_model_input(
                            img,
                            raw_params[0],
                            args.noise_capture_ev,
                            noise_seed=episode_view_seed(args.raw_sensor_noise_seed, task_id, episode_idx, 0),
                            frame_index=step,
                            view_index=0,
                        )
                        wrist_img = make_noise_level_model_input(
                            wrist_img,
                            raw_params[1],
                            args.noise_capture_ev,
                            noise_seed=episode_view_seed(args.raw_sensor_noise_seed, task_id, episode_idx, 1),
                            frame_index=step,
                            view_index=1,
                        )
                    elif args.image_mode == "tonal":
                        img = np.asarray(Image.fromarray(img).resize((224, 224), Image.BILINEAR))
                        wrist_img = np.asarray(Image.fromarray(wrist_img).resize((224, 224), Image.BILINEAR))
                        img = make_tonal_model_input(
                            img,
                            raw_params[0],
                            args.tonal_mode,
                            args.tonal_setting,
                            args.tonal_c,
                            args.tonal_pivot,
                        )
                        wrist_img = make_tonal_model_input(
                            wrist_img,
                            raw_params[1],
                            args.tonal_mode,
                            args.tonal_setting,
                            args.tonal_c,
                            args.tonal_pivot,
                        )
                    elif args.image_mode == "rawvla_bench":
                        assert rawvla_entry is not None
                        assert rawvla_make_observation is not None
                        # Correct deployment order: simulator/render resolution
                        # -> RAW + frame noise -> downstream policy resize.
                        raw_sources = [img, wrist_img]
                        burst_frames = (
                            int(client_model._server_metadata.get("raw_frontend_burst_frames", 1))
                            if client_model.raw_frontend_name in {"rawvla", "raw-vla"}
                            else 1
                        )
                        if raw_frame_history is None:
                            raw_frame_history = [deque(maxlen=burst_frames) for _ in raw_sources]
                        raw_burst_images = []
                        for view_index, source in enumerate(raw_sources):
                            current_raw = rawvla_make_observation(
                                source,
                                noise_seed=int(rawvla_entry["noise_seed"]),
                                frame_index=step,
                                view_index=view_index,
                                representation=args.rawvla_bench_representation,
                                sensor_saturation_ev=float(
                                    rawvla_entry.get("sensor_saturation_ev", 0.0)
                                ),
                            )
                            history = raw_frame_history[view_index]
                            history.append(current_raw)
                            burst = list(history)
                            burst = [burst[0]] * (burst_frames - len(burst)) + burst
                            raw_burst_images.append(burst)
                        # RAWVLA defines the final K-axis element as the current frame.
                        img, wrist_img = raw_burst_images[0][-1], raw_burst_images[1][-1]
                    else:
                        raise ValueError(f"Unknown image_mode={args.image_mode!r}")
                if args.image_quantize_bits > 0:
                    img = quantize_image_to_float_bits(img, args.image_quantize_bits)
                    wrist_img = quantize_image_to_float_bits(wrist_img, args.image_quantize_bits)

                # Save preprocessed image for replay video
                replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                observation = {  #
                    "observation.primary": np.expand_dims(img, axis=0),  # (H, W, C), dtype=unit8, range(0-255)
                    "observation.wrist_image": np.expand_dims(wrist_img, axis=0),  # (H, W, C)
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                # align key with model API --> two images provided here --> check training
                example_dict = {
                    "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "image_float32": [
                        _as_float32_image_channel(observation["observation.primary"][0]),
                        _as_float32_image_channel(observation["observation.wrist_image"][0]),
                    ],
                    "lang": observation["instruction"][0],
                }
                if raw_burst_images is not None and client_model.raw_frontend_name in {"rawvla", "raw-vla"}:
                    example_dict["raw_burst_float32"] = [
                        np.stack([_as_float32_image_channel(frame) for frame in burst], axis=0)
                        for burst in raw_burst_images
                    ]
                    if step == rawvla_export_step:
                        example_dict["_rawvla_frontend_export"] = {
                            "output_dir": args.rawvla_frontend_export_dir,
                            "task_id": int(task_id),
                            "episode_idx": int(episode_idx),
                            "init_state_index": int(rawvla_entry["init_state_index"]),
                            "base_seed": int(rawvla_entry["base_seed"]),
                            "lighting_domain": str(rawvla_entry["lighting_domain"]),
                            "lighting_ev": float(rawvla_entry["lighting_ev"]),
                            "noise_seed": int(rawvla_entry["noise_seed"]),
                            "env_step": int(step),
                            "action_chunk_size": int(client_model.action_chunk_size),
                            "export_seed": int(args.rawvla_frontend_export_seed),
                        }

                start_time = time.time()

                response = client_model.step(example=example_dict, step=step)

                end_time = time.time()
                # print(f"time: {end_time - start_time}")

                # #
                raw_action = response["raw_action"]

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(
                        f"Unexpected action sizes: "
                        f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                        f"Falling back to LIBERO_DUMMY_ACTION."
                    )
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)

                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4"
            if os.environ.get("LIBERO_SKIP_VIDEO", "0") != "1":
                try:
                    imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)
                except Exception as exc:
                    logging.warning(f"Failed to write replay video {video_path}: {exc}")

            full_actions = np.stack(full_actions)
            # np.save(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.npy", full_actions)

            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            _append_episode_ledger(
                args.resume_path,
                {
                    "task_suite": args.task_suite_name,
                    "task_id": task_id,
                    "task_description": task_description,
                    "episode_idx": episode_idx,
                    "success": bool(done),
                    "seed": args.seed,
                    "image_mode": args.image_mode,
                    "ev_representation": args.ev_representation,
                    "image_quantize_bits": args.image_quantize_bits,
                    "chromatic_family": args.chromatic_family,
                    "chromatic_setting": args.chromatic_setting,
                    "chromatic_theta_deg": args.chromatic_theta_deg,
                    "noise_capture_ev": args.noise_capture_ev,
                    "raw_sensor_noise_seed": args.raw_sensor_noise_seed,
                    "tonal_mode": args.tonal_mode,
                    "tonal_setting": args.tonal_setting,
                    "tonal_c": args.tonal_c,
                    "tonal_pivot": args.tonal_pivot,
                    "rawvla_bench_manifest": args.rawvla_bench_manifest,
                    "rawvla_bench_representation": args.rawvla_bench_representation,
                    "rawvla_bench_entry": rawvla_entry,
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            new_episodes_this_run += 1
            if 0 < args.max_new_episodes_per_run <= new_episodes_this_run:
                logging.info(
                    "Reached max_new_episodes_per_run=%d; exiting cleanly for server restart.",
                    args.max_new_episodes_per_run,
                )
                env.close()
                if task_handler is not None:
                    logging.getLogger().removeHandler(task_handler)
                    task_handler.close()
                return

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        env.close()
        if task_handler is not None:
            logging.getLogger().removeHandler(task_handler)
            task_handler.close()

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _load_episode_ledger(path: str | None, task_suite_name: str) -> dict[tuple[int, int], bool]:
    completed: dict[tuple[int, int], bool] = {}
    if not path or not pathlib.Path(path).exists():
        return completed
    with open(path, encoding="utf-8") as ledger:
        for line_number, line in enumerate(ledger, 1):
            try:
                record = json.loads(line)
                if record.get("task_suite") != task_suite_name:
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
    handler = logging.FileHandler(path / f"task_{task_id:02d}.log", mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s | %(message)s", "%m/%d [%H:%M:%S]"))
    logging.getLogger().addHandler(handler)
    return handler


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
