from collections import deque
import csv
import json
import os
from pathlib import Path
import sys
from typing import Dict, Optional

import cv2 as cv
import numpy as np

from deployment.model_server.tools import image_tools
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

try:
    from examples.simBenchmarks.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        action_ensemble=False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="127.0.0.1",
        port=5694,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
    ) -> None:

        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode} ***"
        )
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.normalization_mode = normalization_mode

        # Action mode: "abs", "delta", or "rel"
        self.action_mode = action_mode
        # State tracking for delta/rel modes
        self.initial_state = None  # s_0 for rel mode
        self.prev_action = None  # last absolute action for delta mode

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_chunk_size = None
        self.state_norm_stats = None
        self.raw_actions = None
        self.state_order = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13])
        self._load_state_norm_stats(policy_ckpt_path, unnorm_key)

        server_meta = self.client.get_server_metadata()
        self.action_chunk_size = server_meta["action_chunk_size"]
        self.raw_frontend_name = str(server_meta.get("raw_frontend", "none") or "none").lower()
        self.raw_frontend_source_key = str(server_meta.get("raw_frontend_source_key", "image"))
        self.raw_frontend_burst_frames = int(server_meta.get("raw_frontend_burst_frames", 1))
        self.raw_frontend_rnn_frames = int(server_meta.get("raw_frontend_rnn_frames", 1))
        raw_history_len = self.raw_frontend_burst_frames + self.raw_frontend_rnn_frames - 1
        self.raw_image_histories = [deque(maxlen=raw_history_len) for _ in range(3)]
        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode}, "
            f"server_meta: {server_meta} ***"
        )

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        for history in self.raw_image_histories:
            history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None
        # Reset state tracking for delta/rel modes
        self.initial_state = None
        self.prev_action = None

    def step(
        self,
        example: dict,
        step: int = 0,
    ) -> np.ndarray:
        state = example.get("state", None)
        if state is not None:
            state = self.normalize_state(state)
            example["state"] = state.reshape(1, -1)

        # Store initial state for delta/rel modes
        if self.action_mode in ["delta", "rel"] and self.initial_state is None:
            if state is None:
                raise ValueError(f"action_mode='{self.action_mode}' requires state to be provided in example")
            self.initial_state = np.array(state).copy()

        task_description = example.get("lang", None)
        images = example["image"]

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)
                # Re-store initial state after reset if in delta/rel mode
                if self.action_mode in ["delta", "rel"] and state is not None:
                    self.initial_state = np.array(state).copy()

        if self.raw_frontend_name not in {"", "none"}:
            if len(images) != len(self.raw_image_histories):
                raise ValueError(
                    f"RAWVLA RoboTwin expects {len(self.raw_image_histories)} camera views, got {len(images)}"
                )
            raw_images = [self._validate_raw_image(image) for image in images]
            for history, image in zip(self.raw_image_histories, raw_images):
                history.append(image.copy())
            example[self.raw_frontend_source_key] = [
                self._pack_causal_raw_history(history) for history in self.raw_image_histories
            ]

        images = [self._resize_image(image) for image in images]
        example["image"] = images
        example_copy = example.copy()
        vla_input = {
            "examples": [example_copy],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        vla_input["unnorm_key"] = self.unnorm_key

        action_chunk_size = self.action_chunk_size

        if step % action_chunk_size == 0 or self.raw_actions is None:
            # === TRAIN/TEST CONSISTENCY: keep the observation below aligned with training ===
            # Embodied policies degrade SILENTLY (no error) when the eval-time observation
            # differs from what the model saw during TRAINING. Verify these match the
            # training config used for this checkpoint:
            #   - state       : whether proprioceptive state is included (and its dim/order/normalization))
            #   - image size  : resize / crop resolution (e.g. 224x224)
            #   - image count : how many camera views are fed
            #   - image order : the ordering of those camera views
            #   - action normalization: unnorm_key must match the training dataset stats
            # ==============================================================================
            response = self.client.predict_action(vla_input)
            # server already un-normalized via training-time transform
            raw_actions = np.array(response["data"]["actions"][0])  # (chunk, D)

            # Convert delta/rel to absolute actions
            if self.action_mode == "delta":
                self.raw_actions = self._delta_to_absolute(raw_actions, state)
            elif self.action_mode == "rel":
                self.raw_actions = self._rel_to_absolute(raw_actions)
            else:
                self.raw_actions = raw_actions

        action_idx = step % action_chunk_size
        if action_idx >= len(self.raw_actions):
            pass

        current_action = self.raw_actions[action_idx]

        # Update prev_action for delta mode (for cross-chunk continuity)
        if self.action_mode == "delta":
            self.prev_action = current_action.copy()

        current_action = current_action[[0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]
        return current_action

    def _delta_to_absolute(self, delta_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """Convert delta actions to absolute actions."""
        abs_actions = np.zeros_like(delta_actions)
        base = self.prev_action if self.prev_action is not None else self.initial_state
        for i in range(len(delta_actions)):
            abs_actions[i] = delta_actions[i] + base
            base = abs_actions[i]
        return abs_actions

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        """Convert relative actions to absolute actions."""
        return rel_actions + self.initial_state

    def _load_state_norm_stats(self, policy_ckpt_path: str, unnorm_key: Optional[str]) -> None:
        ckpt_path = Path(policy_ckpt_path)
        stats_path = ckpt_path.parents[1] / "dataset_statistics.json"
        if not stats_path.exists():
            print(f"[WARN] dataset_statistics.json not found for state normalization: {stats_path}")
            return
        with stats_path.open("r", encoding="utf-8") as f:
            all_stats = json.load(f)
        key = unnorm_key
        if key is None:
            if len(all_stats) != 1:
                raise ValueError(
                    f"Multiple unnorm keys in {stats_path}; pass unnorm_key explicitly: {list(all_stats)}"
                )
            key = next(iter(all_stats))
        state_stats = all_stats[key].get("state", {})
        self.state_norm_stats = {
            "min": np.asarray(state_stats["min"], dtype=np.float32),
            "max": np.asarray(state_stats["max"], dtype=np.float32),
        }

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Convert RoboTwin env state to the normalized state layout used at training."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != 14:
            raise ValueError(f"Expected 14-D RoboTwin state, got shape={state.shape}")
        state = state[self.state_order]
        if self.state_norm_stats is None:
            return state

        min_v = self.state_norm_stats["min"]
        max_v = self.state_norm_stats["max"]
        denom = max_v - min_v
        normalized = np.zeros_like(state, dtype=np.float32)
        mask = denom != 0
        normalized[mask] = 2.0 * (state[mask] - min_v[mask]) / denom[mask] - 1.0
        normalized[~mask] = 0.0
        normalized[-2:] = (state[-2:] > 0.49).astype(np.float32)
        return normalized

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = image_tools.validate_image_array(image)
        require_float = os.getenv("STARVLA_REQUIRE_FLOAT_IMAGE", "0").strip().lower() in {
            "1", "true", "yes", "on",
        }
        if require_float and image.dtype != np.float32:
            raise TypeError(
                "STARVLA_REQUIRE_FLOAT_IMAGE=1 requires the RoboTwin observation "
                f"to already be float32 [0, 1], got {image.dtype}"
            )
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / np.float32(255.0)
        else:
            image = image.astype(np.float32, copy=False)
        resized = cv.resize(
            image,
            tuple(self.image_size),
            interpolation=cv.INTER_AREA,
        )
        resized = np.clip(resized, 0.0, 1.0).astype(np.float32, copy=False)
        return image_tools.validate_image_array(resized)

    @staticmethod
    def _validate_raw_image(image: np.ndarray) -> np.ndarray:
        image = image_tools.validate_image_array(image)
        if image.dtype != np.float32:
            raise TypeError(f"RAWVLA requires float32 RAW input, got {image.dtype}")
        if not np.isfinite(image).all():
            raise ValueError("RAWVLA input contains non-finite values")
        if float(image.min()) < 0.0 or float(image.max()) > 1.0:
            raise ValueError(
                f"RAWVLA input must be in [0, 1], got [{float(image.min())}, {float(image.max())}]"
            )
        return image

    def _pack_causal_raw_history(self, history: deque) -> np.ndarray:
        """Pack one view as [rnn, burst, H, W, C], padding only at episode start."""
        frames = list(history)
        if not frames:
            raise RuntimeError("Cannot pack an empty RAW history")
        packed = []
        newest = len(frames) - 1
        for rnn_index in range(self.raw_frontend_rnn_frames):
            outer_end = newest - (self.raw_frontend_rnn_frames - 1 - rnn_index)
            outer_end = max(outer_end, 0)
            burst = []
            for burst_index in range(self.raw_frontend_burst_frames):
                frame_index = outer_end - (self.raw_frontend_burst_frames - 1 - burst_index)
                burst.append(frames[max(frame_index, 0)])
            packed.append(np.stack(burst, axis=0))
        return np.stack(packed, axis=0).astype(np.float32, copy=False)


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 5694)
    unnorm_key = usr_args.get("unnorm_key", None)
    action_mode = usr_args.get("action_mode", "abs")
    normalization_mode = usr_args.get(
        "action_normalization_mode",
        usr_args.get("normalization_mode", "min_max"),
    )

    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=host,
        port=port,
        unnorm_key=unnorm_key,
        action_mode=action_mode,
        normalization_mode=normalization_mode,
    )


def reset_model(model):
    model.reset(task_description="")
    _audit_reset()


_AUDIT_EPISODE = -1
_AUDIT_FRAME = 0


def _audit_reset():
    global _AUDIT_EPISODE, _AUDIT_FRAME
    _AUDIT_EPISODE += 1
    _AUDIT_FRAME = 0


def _decode_hdf_image(value):
    if isinstance(value, np.ndarray) and value.ndim == 3:
        return value
    raw = value.tobytes() if isinstance(value, np.ndarray) else bytes(value)
    image = cv.imdecode(np.frombuffer(raw, dtype=np.uint8), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode reference image")
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def _audit_observation(observation, action):
    global _AUDIT_FRAME
    root = os.environ.get("ROBOTWIN_POLICY_AUDIT_DIR")
    if not root:
        return
    # The queue image has an older glibc than the h5py wheel bundled in the
    # RoboTwin environment.  Allow the audit reader to use a compatible h5py
    # without changing the Python environment used by simulation or policy.
    h5py_site = os.environ.get("ROBOTWIN_POLICY_AUDIT_H5PY_SITE", "")
    if h5py_site and h5py_site not in sys.path:
        sys.path.insert(0, h5py_site)
    import h5py
    task = getattr(observation, "task_name", None) or os.environ.get("ROBOTWIN_POLICY_AUDIT_TASK", "")
    if not task:
        task = os.environ.get("ROBOTWIN_TASK_NAME", "")
    # eval.sh does not otherwise expose the task to the adapter; the wrapper sets it
    # in eval_policy.py before entering the policy loop.
    task = os.environ.get("ROBOTWIN_ACTIVE_TASK", task)
    start = int(os.environ.get("ROBOTWIN_POLICY_AUDIT_START_EPISODE", os.environ.get("START_EPISODE", "0")))
    ref_episode = start + _AUDIT_EPISODE
    data_root = os.environ["ROBOTWIN_POLICY_AUDIT_DATA_ROOT"]
    ref_path = Path(data_root) / task / "aloha-agilex_clean_50" / "data" / f"episode{ref_episode}.hdf5"
    out = Path(root) / task / f"episode_{ref_episode:03d}"
    out.mkdir(parents=True, exist_ok=True)
    policy_key = os.environ.get("ROBOTWIN_POLICY_AUDIT_POLICY_KEY", "render_rgb")
    original_key = os.environ.get("ROBOTWIN_POLICY_AUDIT_ORIGINAL_KEY", "rgb")
    save_every = int(os.environ.get("ROBOTWIN_POLICY_AUDIT_SAVE_EVERY", "5"))
    rows=[]
    with h5py.File(ref_path, "r") as h5:
        for view in ("head_camera", "left_camera", "right_camera"):
            rollout_transport = image_tools.validate_image_array(
                observation["observation"][view][policy_key]
            )
            # PSNR files and PNGs are an explicit display-only boundary. The
            # policy transport above remains float32 [0, 1].
            rollout = image_tools.convert_to_uint8(rollout_transport)
            ds = h5[f"observation/{view}/{original_key}"]
            # Do not repeatedly compare a long policy rollout with the final
            # expert frame after the reference trajectory has ended.
            if _AUDIT_FRAME >= len(ds):
                continue
            idx = _AUDIT_FRAME
            original = _decode_hdf_image(ds[idx])
            if original.shape != rollout.shape:
                raise ValueError(f"Image shape mismatch {view}: {original.shape} vs {rollout.shape}")
            mse = float(np.mean((original.astype(np.float32)-rollout.astype(np.float32))**2))
            psnr = float("inf") if mse == 0 else float(20*np.log10(255.0)-10*np.log10(mse))
            rows.append([task,ref_episode,_AUDIT_FRAME,idx,view,psnr,original_key,policy_key])
            if save_every > 0 and _AUDIT_FRAME % save_every == 0:
                view_dir=out/view; view_dir.mkdir(exist_ok=True)
                cv.imwrite(str(view_dir/f"frame_{_AUDIT_FRAME:04d}_original.png"),cv.cvtColor(original,cv.COLOR_RGB2BGR))
                cv.imwrite(str(view_dir/f"frame_{_AUDIT_FRAME:04d}_rollout.png"),cv.cvtColor(rollout,cv.COLOR_RGB2BGR))
                comp=np.concatenate([original,rollout,cv.absdiff(original,rollout)],axis=1)
                cv.imwrite(str(view_dir/f"frame_{_AUDIT_FRAME:04d}_comparison.png"),cv.cvtColor(comp,cv.COLOR_RGB2BGR))

    csv_path=out/"frame_psnr.csv"; new=not csv_path.exists()
    with csv_path.open("a",newline="") as f:
        w=csv.writer(f)
        if new: w.writerow(["task","episode","rollout_frame","original_frame","view","psnr","original_image_key","policy_image_key"])
        w.writerows(rows)
    with (out/"predicted_actions.jsonl").open("a") as f:
        f.write(json.dumps({"task":task,"episode":ref_episode,"step":_AUDIT_FRAME,
                            "action_source":"model_prediction","action":np.asarray(action).tolist()})+"\n")
    _AUDIT_FRAME += 1


def eval(TASK_ENV, model, observation):
    # Get instruction
    instruction = TASK_ENV.get_instruction()

    # Prepare images
    image_obs_key = os.environ.get("ROBOTWIN_IMAGE_OBS_KEY", "rgb")
    head_img = observation["observation"]["head_camera"][image_obs_key]
    left_img = observation["observation"]["left_camera"][image_obs_key]
    right_img = observation["observation"]["right_camera"][image_obs_key]

    # Order: [head, left, right] to match training order
    images = [head_img, left_img, right_img]

    state = observation["joint_action"]["vector"]
    example = {
        "lang": str(instruction),
        "image": images,
        "state": state,  # Required for delta/rel action modes
    }

    action = model.step(example, step=TASK_ENV.take_action_cnt)

    # Keep the exact action sent to the simulator.  eval_policy.py initializes
    # this list for each episode and persists it as model-policy trajectory
    # data; expert left/right_joint_path is deliberately not involved.
    if hasattr(TASK_ENV, "policy_predicted_actions"):
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.shape != (14,):
            raise ValueError(f"StarVLA must produce one 14-D RoboTwin action, got {action_array.shape}")
        TASK_ENV.policy_predicted_actions.append(action_array.copy())

    _audit_observation(observation, action)

    # Execute action
    TASK_ENV.take_action(action)
