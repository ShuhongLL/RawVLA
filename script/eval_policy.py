import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import pickle
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = int(os.environ.get("ROBOTWIN_EVAL_START_SEED", 100000 * (1 + seed)))
    suc_nums = []
    test_num = int(os.environ.get("ROBOTWIN_TEST_NUM", usr_args.get("test_num", 100)))
    topk = 1

    model = get_model(usr_args)
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    os.environ["ROBOTWIN_ACTIVE_TASK"] = task_name
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0
    record_episode_data = os.environ.get("ROBOTWIN_POLICY_RECORD_EPISODE_DATA", "0") == "1"
    trajectory_root = os.environ.get("ROBOTWIN_POLICY_TRAJECTORY_ROOT", "").strip()
    if trajectory_root:
        trajectory_root = Path(trajectory_root)
        trajectory_root.mkdir(parents=True, exist_ok=True)

    use_dataset_instructions = os.environ.get("ROBOTWIN_USE_DATASET_INSTRUCTIONS", "0") == "1"
    dataset_root = os.environ.get("ROBOTWIN_POLICY_AUDIT_DATA_ROOT", "").strip()
    dataset_episode_start = int(os.environ.get("START_EPISODE", "0"))
    if use_dataset_instructions and not dataset_root:
        raise ValueError("ROBOTWIN_USE_DATASET_INSTRUCTIONS=1 requires ROBOTWIN_POLICY_AUDIT_DATA_ROOT")

    # Seed-matched clean evaluation already has per-episode language in the
    # dataset.  In that mode an expert rollout is unnecessary and can hang on
    # a planner-unstable seed even though policy evaluation itself is valid.
    expert_check = not use_dataset_instructions

    if record_episode_data:
        record_task_root = trajectory_root / task_name if trajectory_root else Path(TASK_ENV.save_dir)
        record_task_root.mkdir(parents=True, exist_ok=True)
        args["save_path"] = str(record_task_root / "rollout")
        args["save_freq"] = 1

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True
    eval_seeds_text = os.environ.get("ROBOTWIN_EVAL_SEEDS", "").strip()
    eval_seeds = [int(value) for value in eval_seeds_text.split(",") if value.strip()] if eval_seeds_text else []
    if eval_seeds and len(eval_seeds) < test_num:
        raise ValueError(f"ROBOTWIN_EVAL_SEEDS has {len(eval_seeds)} seeds but test_num={test_num}")

    def save_policy_episode(episode_id, dataset_episode_id, eval_seed, instruction, success):
        """Save native RoboTwin observations plus the actions predicted by the VLA.

        RoboTwin's save_traj_data() serializes expert motion-planner paths.  A
        policy rollout must not call it because those paths are not the action
        source.  Instead, retain the native observation HDF5 produced from the
        environment cache and attach the exact 14-D model actions.
        """
        actions = np.asarray(TASK_ENV.policy_predicted_actions, dtype=np.float32)
        if actions.size == 0:
            actions = actions.reshape(0, 14)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"Expected predicted actions with shape (T, 14), got {actions.shape}")

        TASK_ENV.merge_pkl_to_hdf5_video()
        hdf5_path = Path(TASK_ENV.save_dir) / "data" / f"episode{TASK_ENV.ep_num}.hdf5"
        if not hdf5_path.is_file():
            raise FileNotFoundError(f"RoboTwin episode HDF5 was not generated: {hdf5_path}")

        import h5py
        with h5py.File(hdf5_path, "a") as h5:
            observation_count = len(h5["joint_action/vector"])
            if observation_count != len(actions):
                raise ValueError(
                    "Policy replay is not time-aligned: "
                    f"{observation_count} observations versus {len(actions)} actions"
                )
            if "policy_action" in h5:
                del h5["policy_action"]
            action_group = h5.create_group("policy_action")
            action_group.create_dataset("vector", data=actions, compression="gzip")
            action_group.attrs["action_source"] = "model_prediction"
            action_group.attrs["language"] = str(instruction)
            action_group.attrs["eval_seed"] = int(eval_seed)
            action_group.attrs["success"] = bool(success)

        policy_dir = Path(TASK_ENV.save_dir) / "_policy_data"
        policy_dir.mkdir(parents=True, exist_ok=True)
        policy_path = policy_dir / f"episode{TASK_ENV.ep_num}.pkl"
        policy_record = {
            "task": task_name,
            "episode": int(episode_id),
            "dataset_episode": int(dataset_episode_id),
            "eval_seed": int(eval_seed),
            "language": str(instruction),
            "success": bool(success),
            "num_steps": int(TASK_ENV.take_action_cnt),
            "action_source": "model_prediction",
            "image_input_key": os.environ.get("ROBOTWIN_IMAGE_OBS_KEY", "render_rgb"),
            "camera_views": ["head_camera", "left_camera", "right_camera"],
            "predicted_actions": actions,
            "hdf5_path": str(hdf5_path),
        }
        with policy_path.open("wb") as policy_file:
            pickle.dump(policy_record, policy_file)
        return hdf5_path, policy_path

    while succ_seed < test_num:
        if eval_seeds:
            now_seed = eval_seeds[now_id]
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        # Never record the expert-check trajectory as policy rollout data.
        args["save_data"] = False

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"Error while expert-checking task={args['task_name']} seed={now_seed}:")
                print(stack_trace)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        force_eval_seed = os.environ.get("ROBOTWIN_FORCE_EVAL_SEED", "0") == "1"
        if force_eval_seed or (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        # These values must be set before setup_demo so the very first policy
        # episode (episode 0) receives the correct save_dir/save_data values.
        if record_episode_data:
            args["save_data"] = True
            args["save_freq"] = 1
        else:
            args["save_data"] = False
            args["save_freq"] = None
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        if use_dataset_instructions:
            instruction_path = (
                Path(dataset_root)
                / task_name
                / "aloha-agilex_clean_50"
                / "instructions"
                / f"episode{dataset_episode_start + now_id}.json"
            )
            with instruction_path.open("r", encoding="utf-8") as instruction_file:
                instruction_data = __import__("json").load(instruction_file)
            candidates = instruction_data.get(instruction_type, [])
            if not candidates:
                raise ValueError(f"No {instruction_type!r} instructions in {instruction_path}")
            instruction = np.random.choice(candidates)
        else:
            episode_info_list = [episode_info["info"]]
            results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
            instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        TASK_ENV.policy_predicted_actions = []
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            # Persist exactly the pre-action observation consumed by the
            # policy, yielding one observation for every predicted action.
            if record_episode_data:
                TASK_ENV._take_picture(observation)
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        audit_dir = os.environ.get("ROBOTWIN_POLICY_AUDIT_DIR")
        if audit_dir:
            Path(audit_dir).mkdir(parents=True, exist_ok=True)
            audit_start = int(os.environ.get("START_EPISODE", "0"))
            with open(Path(audit_dir) / "episode_results.jsonl", "a", encoding="utf-8") as audit_file:
                audit_file.write(__import__("json").dumps({
                    "task": task_name, "episode": audit_start + now_id,
                    "eval_seed": now_seed, "success": bool(succ),
                    "num_steps": int(TASK_ENV.take_action_cnt), "language": str(instruction),
                    "action_source": "model_prediction", "policy_name": args["policy_name"],
                    "task_config": args["task_config"],
                    "checkpoint": os.environ.get("ROBOTWIN_POLICY_CHECKPOINT", ""),
                    "original_image_key": os.environ.get("ROBOTWIN_POLICY_AUDIT_ORIGINAL_KEY", "rgb"),
                    "image_input_key": os.environ.get("ROBOTWIN_IMAGE_OBS_KEY", "render_rgb"),
                    "result_dir": str(audit_dir),
                }) + "\n")

        if record_episode_data:
            try:
                audit_start = int(os.environ.get("START_EPISODE", "0"))
                hdf5_path, policy_path = save_policy_episode(
                    episode_id=now_id,
                    dataset_episode_id=audit_start + now_id,
                    eval_seed=now_seed,
                    instruction=instruction,
                    success=succ,
                )
                print(f"Saved model-policy episode: {hdf5_path} and {policy_path}")
            except Exception as exc:
                print(f"\033[93mWarning: Episode {now_id} trajectory save failed: {exc}\033[0m")
                if trajectory_root:
                    error_dir = trajectory_root / "errors"
                    error_dir.mkdir(parents=True, exist_ok=True)
                    with (error_dir / "trajectory_save_errors.log").open("a", encoding="utf-8") as f:
                        f.write(f"Episode {now_id} save failed: {type(exc).__name__}: {exc}\n")
                raise

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--policy_ckpt_path", type=str, default=None)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    if args.policy_ckpt_path is not None:
        config["policy_ckpt_path"] = args.policy_ckpt_path

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
