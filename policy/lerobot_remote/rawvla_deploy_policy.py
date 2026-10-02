"""RoboTwin adapter for the remote PI0.5 + RAW-VLA policy server."""

from __future__ import annotations

import os
import pickle
import socket
import struct
import time

import numpy as np


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    while size:
        chunk = sock.recv(size)
        if not chunk:
            raise ConnectionError("PI0.5 RAW-VLA policy server closed the connection")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


class RemoteRawVLAPolicy:
    def __init__(self, host: str, port: int, timeout: float = 900.0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
        size = struct.unpack("!Q", _recv_exact(self.sock, 8))[0]
        response = pickle.loads(_recv_exact(self.sock, size))
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "unknown policy-server error"))
        return response.get("result")

    def reset(self):
        self.request({"command": "reset"})

    def infer(self, observation, instruction: str):
        cameras = observation["observation"]
        raw_images = {
            "cam_high": np.asarray(cameras["head_camera"]["raw_linear"]),
            "cam_left_wrist": np.asarray(cameras["left_camera"]["raw_linear"]),
            "cam_right_wrist": np.asarray(cameras["right_camera"]["raw_linear"]),
        }
        for name, image in raw_images.items():
            if image.dtype != np.float32 or image.ndim != 3 or image.shape[-1] != 3:
                raise TypeError(f"{name} must be HWC float32 RAW, got {image.dtype} {image.shape}")
            if not np.isfinite(image).all() or float(image.min()) < 0.0 or float(image.max()) > 1.0:
                raise ValueError(f"{name} violates the float32 [0,1] RAW contract")
        return self.request(
            {
                "command": "infer",
                "instruction": instruction,
                "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
                "raw_images": raw_images,
            }
        )


def get_model(usr_args):
    host = usr_args.get("lerobot_server_host", os.environ.get("LEROBOT_SERVER_HOST", "127.0.0.1"))
    port = int(usr_args.get("lerobot_server_port", os.environ["LEROBOT_SERVER_PORT"]))
    return RemoteRawVLAPolicy(host, port)


def eval(TASK_ENV, model, observation):
    action = np.asarray(model.infer(observation, TASK_ENV.get_instruction()), dtype=np.float32)
    if action.shape != (14,):
        raise ValueError(f"Expected one 14-D action, got {action.shape}")
    TASK_ENV.take_action(action)


def reset_model(model):
    model.reset()
