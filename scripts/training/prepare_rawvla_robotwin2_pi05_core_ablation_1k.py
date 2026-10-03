#!/usr/bin/env python3
"""Build the controlled 1k RoboTwin2 π0.5 RAWVLA core-ablation matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml


VARIANTS = (
    "full",
    "no_burst",
    "no_spatial",
    "no_luma_desc",
    "no_chroma_desc",
    "no_hidden",
    "no_theta_prev",
    "shared_gru",
    "luma_state_only",
    "chroma_state_only",
)

FLAGS = {
    "full": (False, False, False, False, False, False, "split"),
    "no_burst": (True, False, False, False, False, False, "split"),
    "no_spatial": (False, False, False, False, False, True, "split"),
    "no_luma_desc": (False, False, True, False, False, False, "split"),
    "no_chroma_desc": (False, True, False, False, False, False, "split"),
    "no_hidden": (False, False, False, True, False, False, "split"),
    "no_theta_prev": (False, False, False, False, True, False, "split"),
    "shared_gru": (False, False, False, False, False, False, "shared"),
    "luma_state_only": (False, False, False, False, False, False, "luma_only"),
    "chroma_state_only": (False, False, False, False, False, False, "chroma_only"),
}

FLAG_KEYS = (
    "disable_burst_denoise",
    "disable_chroma_descriptor",
    "disable_luma_descriptor",
    "disable_previous_hidden",
    "disable_previous_theta",
    "disable_spatial_feature",
    "recurrent_mode",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(path: Path) -> Path:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def assert_stage1_contract(config: dict) -> None:
    frontend = config["framework"]["raw_frontend"]
    data = config["datasets"]["vla_data"]
    trainer = config["trainer"]
    expected = {
        "seed": 20260906,
        "steps": 1000,
        "warmup": 50,
        "batch": 1,
        "accumulation": 8,
        "action_weight": 0.1,
        "exposure_weight": 0.03,
        "chroma_weight": 0.3,
        "learning_rate": 5.0e-5,
        "max_exposure_ev": 10.0,
        "burst_frames": 6,
        "rnn_frames": 6,
        "observation_stride": 5,
        "action_horizon": 50,
        "action_offset": 1,
    }
    observed = {
        "seed": int(config["seed"]),
        "steps": int(trainer["max_train_steps"]),
        "warmup": int(trainer["num_warmup_steps"]),
        "batch": int(data["per_device_batch_size"]),
        "accumulation": int(trainer["gradient_accumulation_steps"]),
        "action_weight": float(trainer["action_loss_weight"]),
        "exposure_weight": float(frontend["exposure_loss_weight"]),
        "chroma_weight": float(frontend["chroma_supervision_weight"]),
        "learning_rate": float(trainer["learning_rate"]["raw_frontend"]),
        "max_exposure_ev": float(frontend["max_exposure_ev"]),
        "burst_frames": int(frontend["burst_frames"]),
        "rnn_frames": int(frontend["rnn_frames"]),
        "observation_stride": int(data["observation_stride"]),
        "action_horizon": int(data["action_horizon"]),
        "action_offset": int(data["action_offset"]),
    }
    if observed != expected:
        raise ValueError(f"Stage-1 contract changed: expected={expected}, observed={observed}")
    if frontend.get("checkpoint"):
        raise ValueError("The controlled 1k matrix must initialize RAWVLA from scratch")
    if data.get("normalization") != "zscore" or data.get("delta_action_indices") != []:
        raise ValueError("π0.5 must use zscore normalization without delta actions")
    if trainer.get("freeze_modules") != "vlm,action_head":
        raise ValueError("π0.5 VLM/action head must remain frozen")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    asset = args.asset_root.resolve()
    model = args.model_root.resolve()
    template_path = require(
        root / "docs/repro_configs/rawvla/pi05_stage1_strong_rgb_chroma_1k.full.yaml"
    )
    raw_cache = require(
        asset / "benchmark_data/robotwin2/rawvla_light_train_statecopy_633/raw"
    )
    rgb_cache = require(
        asset / "benchmark_data/robotwin2/rawvla_light_train_statecopy_633/rgb"
    )
    backbone = require(model / "model.safetensors")
    stats = require(model / "policy_norm_stats.json")
    tokenizer = require(asset / "openpi_converted_protocol/paligemma_tokenizer.model")

    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    assert_stage1_contract(template)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.result_root.mkdir(parents=True, exist_ok=True)

    matrix = []
    for index, variant in enumerate(VARIANTS):
        config = copy.deepcopy(template)
        run_id = f"pi05_robotwin2_1k__{variant}__seed20260906"
        config["run_id"] = run_id
        config["run_root_dir"] = str(args.result_root)
        config["wandb_project"] = "rawvla_pi05_robotwin2_core_ablation_1k"
        config["framework"]["tokenizer"]["model_path"] = str(tokenizer)
        frontend = config["framework"]["raw_frontend"]
        frontend.pop("checkpoint", None)
        frontend.pop("strict_load", None)
        for key, value in zip(FLAG_KEYS, FLAGS[variant], strict=True):
            frontend[key] = value
        data = config["datasets"]["vla_data"]
        data["cache_root"] = str(raw_cache)
        data["rgb_cache_root"] = str(rgb_cache)
        data["stats_path"] = str(stats)
        config["trainer"]["pretrained_checkpoint"] = str(backbone)
        config["config_yaml"] = "generated-at-job-start"
        config["output_dir"] = str(args.result_root / run_id)
        assert_stage1_contract(config)

        path = args.output_dir / f"{run_id}.yaml"
        path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        matrix.append(
            {
                "index": index,
                "preferred_gpu": index % 8,
                "variant": variant,
                "run_id": run_id,
                "config": str(path),
                "config_sha256": sha256(path),
                "flags": dict(zip(FLAG_KEYS, FLAGS[variant], strict=True)),
            }
        )

    payload = {
        "protocol": "pi05_robotwin2_stage1_controlled_core_ablation_1k",
        "template": str(template_path),
        "template_sha256": sha256(template_path),
        "seeded_before_model_build": True,
        "variants": matrix,
    }
    (args.output_dir / "matrix.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(matrix)} configs to {args.output_dir}")


if __name__ == "__main__":
    main()
