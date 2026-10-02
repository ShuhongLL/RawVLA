#!/usr/bin/env python3
"""Smoke-test downloaded StarVLA LIBERO checkpoints.

Default mode validates that checkpoint/config/statistics files are present and
readable. Use --load to instantiate each model, which requires the base VLMs and
enough GPU/CPU memory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("STARVLA_ROOT", REPO_ROOT / "starVLA")).expanduser().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from starVLA.model.framework.share_tools import read_mode_config
CHECKPOINTS = [
    ROOT / "playground/Pretrained_models/StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt",
    ROOT / "playground/Pretrained_models/StarVLA/Qwen2.5-VL-FAST-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt",
    ROOT / "playground/Pretrained_models/StarVLA/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt",
    ROOT / "playground/Pretrained_models/StarVLA/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt",
    ROOT / "playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt",
]


def validate_files(ckpt: Path) -> None:
    print(f"\n== {ckpt} ==")
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")
    cfg, stats = read_mode_config(str(ckpt))
    proc = PolicyNormProcessor(str(ckpt))
    print("framework:", cfg["framework"]["name"])
    print("base_vlm:", cfg["framework"]["qwenvl"]["base_vlm"])
    print("stats keys:", list(stats.keys()))
    print("action keys:", proc.action_keys)
    print("state keys:", proc.state_keys)


def load_model(ckpt: Path) -> None:
    from starVLA.model.framework.base_framework import baseframework

    model = baseframework.from_pretrained(
        str(ckpt),
        config_overrides=["framework.qwenvl.attn_implementation=sdpa"],
    )
    print("loaded:", model.__class__.__name__)
    del model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="instantiate each model after metadata checks")
    args = parser.parse_args()

    for ckpt in CHECKPOINTS:
        validate_files(ckpt)
        if args.load:
            load_model(ckpt)


if __name__ == "__main__":
    main()
