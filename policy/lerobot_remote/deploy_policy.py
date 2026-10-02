"""RoboTwin adapter for a LeRobot policy served by ``policy_server.py``."""

from __future__ import annotations

import os
import json
import pickle
import socket
import struct
import time
from pathlib import Path

import numpy as np


def _policy_uint8_image(image):
    """Convert RoboTwin RGB products to policy uint8 without crushing float [0,1]."""
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all() or float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise ValueError(
            f"Expected uint8 or float RGB in [0,1], got dtype={array.dtype} "
            f"range=[{float(array.min())}, {float(array.max())}]"
        )
    return np.rint(array * np.float32(255.0)).astype(np.uint8)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("LeRobot policy server closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class RemoteLeRobotPolicy:
    def __init__(self, host: str, port: int, timeout: float = 600.0):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._input_audit_done = False
        self.sock.settimeout(timeout)
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.sock.connect((host, port))
                break
            except (ConnectionRefusedError, OSError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)

    def request(self, payload):
        body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        self.sock.sendall(struct.pack("!Q", len(body)) + body)
        response_size = struct.unpack("!Q", _recv_exact(self.sock, 8))[0]
        response = pickle.loads(_recv_exact(self.sock, response_size))
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "unknown policy-server error"))
        return response.get("result")

    def reset(self):
        self.request({"command": "reset"})
        self._input_audit_done = False

    def infer(self, observation, instruction: str):
        cameras = observation["observation"]
        image_key = os.environ.get("ROBOTWIN_IMAGE_OBS_KEY", "render_rgb")
        source_images = {
            "cam_high": np.asarray(cameras["head_camera"][image_key]),
            "cam_left_wrist": np.asarray(cameras["left_camera"][image_key]),
            "cam_right_wrist": np.asarray(cameras["right_camera"][image_key]),
        }
        policy_images = {name: _policy_uint8_image(image) for name, image in source_images.items()}
        audit_dir = os.environ.get("ROBOTWIN_POLICY_INPUT_AUDIT_DIR", "").strip()
        if audit_dir and not self._input_audit_done:
            output = Path(audit_dir)
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output / "first_policy_input.npz",
                **{f"source_{name}": image for name, image in source_images.items()},
                **{f"uint8_{name}": image for name, image in policy_images.items()},
            )
            stats = {}
            for group, images in (("source", source_images), ("uint8", policy_images)):
                for name, image in images.items():
                    stats[f"{group}_{name}"] = {
                        "dtype": str(image.dtype), "shape": list(image.shape),
                        "min": float(image.min()), "max": float(image.max()),
                        "mean": float(image.mean()), "std": float(image.std()),
                    }
            (output / "first_policy_input_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
            self._input_audit_done = True
        return self.request(
            {
                "command": "infer",
                "instruction": instruction,
                "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
                "images": policy_images,
            }
        )


def get_model(usr_args):
    host = usr_args.get("lerobot_server_host", os.environ.get("LEROBOT_SERVER_HOST", "127.0.0.1"))
    port = int(usr_args.get("lerobot_server_port", os.environ["LEROBOT_SERVER_PORT"]))
    return RemoteLeRobotPolicy(host, port)


def eval(TASK_ENV, model, observation):
    action = np.asarray(model.infer(observation, TASK_ENV.get_instruction()), dtype=np.float32)
    if action.shape != (14,):
        raise ValueError(f"Expected one 14-D action, got {action.shape}")
    TASK_ENV.take_action(action)


def reset_model(model):
    model.reset()
