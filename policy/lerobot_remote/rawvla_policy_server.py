#!/usr/bin/env python3
"""Serve LeRobot PI0.5 composed with a streaming RAW-VLA frontend."""

from __future__ import annotations

import argparse
import json
import pickle
import socket
import struct
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import torch

from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_pre_post_processors
from lerobot.processor import ProcessorStepRegistry, RelativeActionsProcessorStep
from baselines.rawvla import RAWVLA


RAWVLA_CONFIG = {
    "name": "rawvla",
    "checkpoint": None,
    "strict_load": True,
    "trainable": False,
    "burst_frames": 6,
    "rnn_frames": 6,
    "spatial_width": 32,
    "hist_bins": 64,
    "state_dim": 128,
    "luma_state_dim": 64,
    "fft_patch_size": 32,
    "fft_stride": 16,
    "use_local_color": False,
    "use_local_tone": False,
    "split_luma_chroma_condition": True,
    "gray_preserving_ccm": True,
    "luminance_weights": [1.0, 1.0, 1.0],
    "luma_spatial_input": "luma",
    "max_exposure_ev": 10.0,
    "fixed_update_alpha": 1.0,
}


try:
    ProcessorStepRegistry.get("delta_actions_processor")
except KeyError:
    ProcessorStepRegistry.register("delta_actions_processor")(RelativeActionsProcessorStep)


def recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks = []
    while size:
        chunk = conn.recv(size)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def load_policy(checkpoint: Path):
    from lerobot.policies.pi05 import PI05Config, PI05Policy

    policy_type = json.loads((checkpoint / "config.json").read_text())["type"]
    if policy_type != "pi05":
        raise ValueError(f"Expected a pi05 checkpoint, got {policy_type!r}")
    config = PI05Config.from_pretrained(checkpoint)
    config.device = "cuda"
    config.compile_model = False
    policy = PI05Policy.from_pretrained(checkpoint, config=config).to("cuda").eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    return policy, preprocessor, postprocessor


class StreamingRawVLA:
    VIEW_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __init__(self, config_path: Path, checkpoint: Path, observation_stride: int):
        # config_path is retained as a required, auditable deployment artifact.
        # Use an explicit dependency-free inference topology in this lightweight
        # policy environment; it is identical to the portable YAML.
        frontend_cfg = dict(RAWVLA_CONFIG)
        self.frontend = RAWVLA(
            spatial_width=frontend_cfg["spatial_width"],
            hist_bins=frontend_cfg["hist_bins"],
            state_dim=frontend_cfg["state_dim"],
            luma_state_dim=frontend_cfg["luma_state_dim"],
            fft_patch_size=frontend_cfg["fft_patch_size"],
            fft_stride=frontend_cfg["fft_stride"],
            use_local_color=frontend_cfg["use_local_color"],
            use_local_tone=frontend_cfg["use_local_tone"],
            split_luma_chroma_condition=frontend_cfg["split_luma_chroma_condition"],
            gray_preserving_ccm=frontend_cfg["gray_preserving_ccm"],
            luminance_weights=frontend_cfg["luminance_weights"],
            luma_spatial_input=frontend_cfg["luma_spatial_input"],
            max_exposure_ev=frontend_cfg["max_exposure_ev"],
            fixed_update_alpha=frontend_cfg["fixed_update_alpha"],
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = {key.removeprefix("model."): value for key, value in state.items()}
        self.frontend.load_state_dict(state, strict=True)
        self.frontend.to("cuda").eval()
        self.rnn_frames = int(frontend_cfg["rnn_frames"])
        self.burst_frames = int(frontend_cfg["burst_frames"])
        self.observation_stride = int(observation_stride)
        if (self.rnn_frames, self.burst_frames, self.observation_stride) != (6, 6, 5):
            raise ValueError(
                "Canonical PI0.5 RAW-VLA evaluation requires rnn=6, burst=6, stride=5; "
                f"got {(self.rnn_frames, self.burst_frames, self.observation_stride)}"
            )
        history_len = self.observation_stride * (self.rnn_frames - 1) + self.burst_frames
        self.histories = {name: deque(maxlen=history_len) for name in self.VIEW_NAMES}
        self.cached_images = None

    def reset(self):
        for history in self.histories.values():
            history.clear()
        self.cached_images = None

    @staticmethod
    def _validate(image, name: str) -> np.ndarray:
        array = np.asarray(image)
        if array.dtype != np.float32 or array.ndim != 3 or array.shape[-1] != 3:
            raise TypeError(f"{name} must be HWC float32 RAW, got {array.dtype} {array.shape}")
        if not np.isfinite(array).all() or float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise ValueError(f"{name} violates the float32 [0,1] RAW contract")
        return array

    def _pack(self, history: deque) -> np.ndarray:
        frames = list(history)
        newest = len(frames) - 1
        observations = [
            max(0, newest - self.observation_stride * (self.rnn_frames - 1 - index))
            for index in range(self.rnn_frames)
        ]
        bursts = []
        for endpoint in observations:
            bursts.append(
                np.stack(
                    [frames[max(0, endpoint - (self.burst_frames - 1 - index))] for index in range(self.burst_frames)],
                    axis=0,
                )
            )
        return np.stack(bursts, axis=0)

    def update(self, raw_images: dict[str, np.ndarray], *, render: bool) -> dict[str, np.ndarray]:
        if set(raw_images) != set(self.VIEW_NAMES):
            raise ValueError(f"Expected RAW views {self.VIEW_NAMES}, got {sorted(raw_images)}")
        for name in self.VIEW_NAMES:
            self.histories[name].append(self._validate(raw_images[name], name).copy())
        if not render:
            if self.cached_images is None:
                raise RuntimeError("RAW-VLA output is unavailable before the first policy inference")
            return self.cached_images

        packed = np.stack([self._pack(self.histories[name]) for name in self.VIEW_NAMES], axis=0)
        tensor = torch.from_numpy(packed).permute(0, 1, 2, 5, 3, 4).contiguous().to("cuda")
        with torch.inference_mode():
            state = theta = result = None
            for observation_index in range(self.rnn_frames):
                result = self.frontend(
                    tensor[:, observation_index], state=state, theta_prev=theta
                )
                state, theta = result.state, result.theta
            output = result.rgb.float().cpu()
        if output.shape[:2] != (3, 3) or not torch.isfinite(output).all():
            raise RuntimeError(f"Invalid RAW-VLA output: shape={tuple(output.shape)}")
        images = {}
        for index, name in enumerate(self.VIEW_NAMES):
            array = output[index].clamp(0, 1).permute(1, 2, 0).numpy()
            images[name] = np.rint(array * np.float32(255.0)).astype(np.uint8)
        self.cached_images = images
        return images


def infer(policy, preprocessor, postprocessor, rawvla: StreamingRawVLA, request):
    needs_observation = len(policy._action_queue) == 0
    images = rawvla.update(request["raw_images"], render=needs_observation)
    raw = {
        "pixels": images,
        "agent_pos": np.asarray(request["state"], dtype=np.float32),
    }
    observation = preprocess_observation(raw)
    observation["task"] = [str(request["instruction"])]
    observation = preprocessor(observation)
    with torch.inference_mode():
        action = policy.select_action(observation)
    action = postprocessor(action)
    return action[0].detach().cpu().float().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--raw-frontend-checkpoint", required=True, type=Path)
    parser.add_argument("--raw-frontend-config", required=True, type=Path)
    parser.add_argument("--observation-stride", default=5, type=int)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()

    for path in (args.checkpoint, args.raw_frontend_checkpoint, args.raw_frontend_config):
        if not path.exists():
            raise FileNotFoundError(path)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", args.port))
    server.listen(1)
    policy, preprocessor, postprocessor = load_policy(args.checkpoint)
    rawvla = StreamingRawVLA(args.raw_frontend_config, args.raw_frontend_checkpoint, args.observation_stride)
    print(
        f"READY checkpoint={args.checkpoint} raw_frontend={args.raw_frontend_checkpoint} "
        f"stride={args.observation_stride} port={args.port}",
        flush=True,
    )

    while True:
        conn, _ = server.accept()
        with conn:
            while True:
                try:
                    size = struct.unpack("!Q", recv_exact(conn, 8))[0]
                    request = pickle.loads(recv_exact(conn, size))
                except EOFError:
                    break
                try:
                    if request["command"] == "reset":
                        policy.reset()
                        rawvla.reset()
                        result = None
                    elif request["command"] == "infer":
                        result = infer(policy, preprocessor, postprocessor, rawvla, request)
                    else:
                        raise ValueError(f"Unknown command: {request['command']}")
                    response = {"ok": True, "result": result}
                except Exception:
                    response = {"ok": False, "error": traceback.format_exc()}
                body = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
                conn.sendall(struct.pack("!Q", len(body)) + body)


if __name__ == "__main__":
    main()
