#!/usr/bin/env python3
"""Serve an official OpenPI/JAX RoboTwin checkpoint over localhost."""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import traceback

import numpy as np

from openpi.policies import policy_config
from openpi.training import config as training_config


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks = []
    while size:
        chunk = conn.recv(size)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def load_policy(checkpoint: str, config_name: str, asset_id: str):
    return policy_config.create_trained_policy(
        training_config.get_config(config_name),
        checkpoint,
        robotwin_repo_id=asset_id,
    )


def infer(policy, request):
    # OpenPI's ALOHA transform expects channel-first uint8 images.
    images = {
        key: np.asarray(value, dtype=np.uint8).transpose(2, 0, 1)
        for key, value in request["images"].items()
    }
    observation = {
        "state": np.asarray(request["state"], dtype=np.float32),
        "images": images,
        "prompt": str(request["instruction"]),
    }
    actions = np.asarray(policy.infer(observation)["actions"], dtype=np.float32)
    # Lists avoid NumPy pickle incompatibilities between policy and simulator envs.
    return actions.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="pi0_base_aloha_robotwin_full")
    parser.add_argument("--asset-id", default="robotwin_50_clean")
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", args.port))
    server.listen(1)
    policy = load_policy(args.checkpoint, args.config, args.asset_id)
    print(
        f"READY checkpoint={args.checkpoint} config={args.config} "
        f"asset_id={args.asset_id} port={args.port}",
        flush=True,
    )

    while True:
        conn, _ = server.accept()
        with conn:
            while True:
                try:
                    size = struct.unpack("!Q", _recv_exact(conn, 8))[0]
                    request = pickle.loads(_recv_exact(conn, size))
                except EOFError:
                    break
                try:
                    if request["command"] == "reset":
                        result = None
                    elif request["command"] == "infer":
                        result = infer(policy, request)
                    else:
                        raise ValueError(f"Unknown command: {request['command']}")
                    response = {"ok": True, "result": result}
                except Exception:
                    response = {"ok": False, "error": traceback.format_exc()}
                body = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
                conn.sendall(struct.pack("!Q", len(body)) + body)


if __name__ == "__main__":
    main()
