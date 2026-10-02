#!/usr/bin/env python3
"""Serve one LeRobot PI policy over a small localhost-only pickle protocol."""

from __future__ import annotations

import argparse
import json
import pickle
import socket
import struct
import traceback

import numpy as np
import torch

from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_pre_post_processors
from lerobot.processor import ProcessorStepRegistry, RelativeActionsProcessorStep


# ``sumitagrawal/pi0-robotwin-phaseB2-30k`` was serialized before LeRobot
# renamed this no-op-compatible processor. Register the old name as an alias
# while leaving the checkpoint files untouched.
try:
    ProcessorStepRegistry.get("delta_actions_processor")
except KeyError:
    ProcessorStepRegistry.register("delta_actions_processor")(RelativeActionsProcessorStep)


def recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def load_policy(checkpoint: str):
    policy_type = json.loads((__import__("pathlib").Path(checkpoint) / "config.json").read_text())["type"]
    if policy_type == "pi0":
        from lerobot.policies.pi0 import PI0Config, PI0Policy

        config_cls, policy_cls = PI0Config, PI0Policy
    elif policy_type == "pi05":
        from lerobot.policies.pi05 import PI05Config, PI05Policy

        config_cls, policy_cls = PI05Config, PI05Policy
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}")
    config = config_cls.from_pretrained(checkpoint)
    config.device = "cuda"
    config.compile_model = False
    policy = policy_cls.from_pretrained(checkpoint, config=config).to("cuda").eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    return policy, preprocessor, postprocessor


def infer(policy, preprocessor, postprocessor, request):
    raw = {
        "pixels": request["images"],
        "agent_pos": np.asarray(request["state"], dtype=np.float32),
    }
    observation = preprocess_observation(raw)
    observation["task"] = [str(request["instruction"])]
    observation = preprocessor(observation)
    with torch.inference_mode():
        action = policy.select_action(observation)
    action = postprocessor(action)
    # A plain list keeps the wire format independent of the NumPy major
    # versions used by the Python 3.12 server and Python 3.10 simulator.
    return action[0].detach().cpu().float().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", args.port))
    server.listen(1)
    policy, preprocessor, postprocessor = load_policy(args.checkpoint)
    print(f"READY checkpoint={args.checkpoint} port={args.port}", flush=True)

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
                        result = None
                    elif request["command"] == "infer":
                        result = infer(policy, preprocessor, postprocessor, request)
                    else:
                        raise ValueError(f"Unknown command: {request['command']}")
                    response = {"ok": True, "result": result}
                except Exception:
                    response = {"ok": False, "error": traceback.format_exc()}
                body = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
                conn.sendall(struct.pack("!Q", len(body)) + body)


if __name__ == "__main__":
    main()
