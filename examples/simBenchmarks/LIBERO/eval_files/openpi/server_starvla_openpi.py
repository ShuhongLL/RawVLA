#!/usr/bin/env python3
"""StarVLA PI0/PI05 websocket policy server for LIBERO eval."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from deployment.model_server.tools.image_tools import audit_image_once, validate_image_array

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from pi_libero_common import (
    DEFAULT_TOKENIZER,
    OPENPI_MODEL_SOURCE,
    STARVLA_MODEL_SOURCE,
    configure_torch,
    default_converted_checkpoint,
    build_model,
    canonicalize_model_name,
    canonicalize_model_source,
    normalize_openpi_value,
    postprocess_openpi_actions,
    resolve_norm_stats_source,
)


class StarVLAOpenPIPolicy:
    def __init__(
        self,
        model_name: str,
        checkpoint: Path,
        tokenizer: Path,
        precision: str,
        device: str,
        model_source: str,
        config_overrides: list[str] | None = None,
    ):
        configure_torch()
        self.model_name = canonicalize_model_name(model_name)
        self.device = torch.device(device)
        self.checkpoint = str(checkpoint)
        self.precision = precision
        self.model_source = canonicalize_model_source(model_source)
        self.config_overrides = list(config_overrides or [])
        self.model = build_model(
            checkpoint,
            self.device,
            precision,
            tokenizer=tokenizer,
            config_overrides=self.config_overrides,
        )
        self.action_horizon = 10 if self.model_name == "PI05" else 50
        self.action_dim = 32
        self.norm_stats = resolve_norm_stats_source(self.model_name, checkpoint=checkpoint)
        self.use_quantile_norm = self.model_name != "PI0"

    def _override(self, key: str, default: str) -> str:
        prefix = f"{key}="
        for item in reversed(self.config_overrides):
            if item.startswith(prefix):
                return item[len(prefix) :]
        return default

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "env": "starvla_openpi_server",
            "model": self.model_name,
            "checkpoint": self.checkpoint,
            "precision": self.precision,
            "model_source": self.model_source,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "norm": "quantile" if self.use_quantile_norm else "zscore",
            "config_override_keys": [item.split("=", 1)[0] for item in self.config_overrides],
            "raw_frontend": self._override("framework.raw_frontend.name", "none"),
            "raw_frontend_source_key": self._override("framework.raw_frontend.source_key", "image"),
            "raw_frontend_burst_frames": int(self._override("framework.raw_frontend.burst_frames", "1")),
            "raw_frontend_rnn_frames": int(self._override("framework.raw_frontend.rnn_frames", "1")),
        }

    def predict_action(self, examples: list[dict] | None = None, **kwargs: Any) -> dict[str, Any]:
        if examples is None:
            return self.infer(kwargs)
        for example in examples:
            for image_idx, image in enumerate(example["image"]):
                validate_image_array(image)
                audit_image_once("policy_server_received", image)
                audit_image_once(f"policy_server_received_{image_idx}", image)
        with torch.inference_mode():
            out = self.model.predict_action(examples, **kwargs)
        return {"normalized_actions": out["normalized_actions"]}

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        raw_state = np.asarray(obs["observation/state"], dtype=np.float32)
        model_input = {
            "image": [obs["observation/image"], obs["observation/wrist_image"]],
            "lang": str(obs.get("prompt", "")),
            "state": normalize_openpi_value(
                raw_state,
                self.norm_stats["state"],
                self.use_quantile_norm,
            ).astype(np.float32)[None],
        }
        with torch.inference_mode():
            out = self.model.predict_action([model_input])
        normalized_actions = np.asarray(out["normalized_actions"][0], dtype=np.float32)
        actions = postprocess_openpi_actions(
            normalized_actions=normalized_actions,
            raw_state=raw_state,
            norm_stats=self.norm_stats,
            model_name=self.model_name,
            model_source=self.model_source,
        )
        return {"actions": actions}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["PI0", "PI05", "pi0", "pi05"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--precision", default="float32", choices=["bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--idle-timeout", type=int, default=-1)
    parser.add_argument("--model-source", default=OPENPI_MODEL_SOURCE, choices=[OPENPI_MODEL_SOURCE, STARVLA_MODEL_SOURCE])
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable OmegaConf dotlist override applied before model construction.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    model_name = canonicalize_model_name(args.model)
    checkpoint = args.checkpoint or default_converted_checkpoint(model_name, args.precision)
    policy = StarVLAOpenPIPolicy(
        model_name,
        checkpoint,
        args.tokenizer,
        args.precision,
        args.device,
        args.model_source,
        config_overrides=args.config_override,
    )
    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=policy.metadata,
    )
    logging.info("server running on port %s metadata=%s", args.port, policy.metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
