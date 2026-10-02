#!/usr/bin/env python3
"""Build a benchmark-independent RAW-VLA policy sweep from a JSON spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf


def require_file(path: str) -> str:
    resolved = str(Path(path).resolve())
    if not Path(resolved).is_file():
        raise FileNotFoundError(resolved)
    return resolved


def require_dir(path: str) -> str:
    resolved = str(Path(path).resolve())
    if not Path(resolved).is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--max-train-steps", type=int, default=1000)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    experiments = spec["experiments"]
    if len(experiments) != 8:
        raise ValueError(f"An eight-GPU sweep requires exactly 8 experiments, got {len(experiments)}")
    data = dict(spec["dataset"])
    data["cache_root"] = require_dir(data["cache_root"])
    tokenizer = require_file(spec["tokenizer_path"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.result_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    experiment_defaults = dict(spec.get("experiment_defaults", {}))
    for gpu, experiment in enumerate(experiments):
        experiment = {**experiment_defaults, **experiment}
        policy = dict(spec["policies"][experiment["policy"]])
        policy["checkpoint"] = require_file(policy["checkpoint"])
        policy["stats_path"] = require_file(policy["stats_path"])
        rnn_frames = int(experiment.get("rnn_frames", 6))
        learning_rate = float(experiment.get("learning_rate", 1.0e-4))
        raw_frontend = {
            "name": "rawvla",
            "checkpoint": experiment.get("raw_frontend_checkpoint", None),
            "strict_load": True,
            "source_key": "raw_burst_float32",
            "input_format": "rgb_raw",
            "output_contract": "rgb_chw_float32_0_1",
            "trainable": True,
            "trainable_parameter_prefixes": experiment.get(
                "trainable_parameter_prefixes", []
            ),
            "burst_frames": 6,
            "rnn_frames": rnn_frames,
            "spatial_width": 32,
            "hist_bins": 64,
            "state_dim": 128,
            "luma_state_dim": 64,
            "fft_patch_size": 32,
            "fft_stride": 16,
            "use_local_color": False,
            "use_local_tone": False,
            "split_luma_chroma_condition": True,
            "gray_preserving_ccm": bool(
                experiment.get("gray_preserving_ccm", False)
            ),
            "luminance_weights": [1.0, 1.0, 1.0],
            "luma_spatial_input": "luma",
            "exposure_target": float(experiment.get("exposure_target", 0.789)),
            "exposure_target_by_lighting_domain": experiment.get(
                "exposure_target_by_lighting_domain", {}
            ),
            "exposure_loss_weight": float(experiment.get("exposure_loss_weight", 0.01)),
            "exposure_loss_type": "two_sided_l1",
            "exposure_objective_clamp": "ste",
            "max_exposure_ev": float(experiment.get("max_exposure_ev", 10.0)),
            "fixed_update_alpha": experiment.get("fixed_update_alpha", 1.0),
            "extreme_low_exposure_multiplier": float(
                experiment.get("extreme_low_exposure_multiplier", 20.0)
            ),
            "exposure_loss_multiplier_by_lighting_domain": experiment.get(
                "exposure_loss_multiplier_by_lighting_domain", {}
            ),
            "chroma_prior_enabled": True,
            "chroma_prior_mode": "output_envelope",
            "chroma_prior_descriptor": "mean_log",
            "chroma_prior_weight": float(experiment.get("chroma_prior_weight", 0.001)),
            "chroma_prior_lower": [0.025407744850963355, -0.35791672915220263],
            "chroma_prior_upper": [0.25704274773597713, -0.0317848339676857],
            "chroma_prior_margin": 0.10,
            "chroma_supervision_weight": float(
                experiment.get("chroma_supervision_weight", 0.0)
            ),
            "chroma_supervision_beta": float(
                experiment.get("chroma_supervision_beta", 0.05)
            ),
            "chroma_supervision_min_luma": float(
                experiment.get("chroma_supervision_min_luma", 0.02)
            ),
            "chroma_supervision_max_luma": float(
                experiment.get("chroma_supervision_max_luma", 0.98)
            ),
            "chroma_supervision_highlight_luma_threshold": float(
                experiment.get("chroma_supervision_highlight_luma_threshold", 0.8)
            ),
            "chroma_supervision_highlight_multiplier_by_lighting_domain": experiment.get(
                "chroma_supervision_highlight_multiplier_by_lighting_domain", {}
            ),
            "chroma_supervision_multiplier_by_lighting_domain": experiment.get(
                "chroma_supervision_multiplier_by_lighting_domain", {}
            ),
        }
        dataset = {
            **data,
            "stats_path": policy["stats_path"],
            "normalization": policy["normalization"],
            "stats_key": policy.get("stats_key", ""),
            "action_stats_key": policy.get("action_stats_key", "actions"),
            "state_stats_key": policy.get("state_stats_key", "state"),
            "delta_action_indices": policy.get("delta_action_indices", []),
            "burst_frames": 6,
            "rnn_frames": rnn_frames,
            "observation_stride": int(policy.get("observation_stride", 5)),
            "action_horizon": int(policy.get("action_horizon", 50)),
            "action_offset": int(policy.get("action_offset", 0)),
            "state_source_key": policy.get("state_source_key", "robot_states"),
        }
        framework = {
            "name": policy["framework_name"],
            "precision": "bfloat16",
            "action_dim": int(policy.get("action_dim", 32)),
            "action_horizon": int(policy.get("action_horizon", 50)),
            "max_state_dim": int(policy.get("max_state_dim", 32)),
            "discrete_state_input": bool(policy.get("discrete_state_input", False)),
            "max_token_len": int(policy["max_token_len"]),
            "num_inference_steps": int(policy.get("num_inference_steps", 10)),
            "tokenizer": {"model_path": tokenizer, "pad_state_value": -2.0},
            "paligemma": {"model_name_or_path": ""},
            "raw_frontend": raw_frontend,
        }
        name = experiment["name"]
        cfg = OmegaConf.create(
            {
                "run_id": name,
                "run_root_dir": str(args.result_root.resolve()),
                "seed": int(spec.get("seed", 42)) + gpu,
                "wandb_entity": "your_wandb_entity",
                "wandb_project": spec.get("wandb_project", "rawvla_policy_sweep"),
                "is_debug": False,
                "framework": framework,
                "datasets": {"vla_data": dataset},
                "trainer": {
                    "pretrained_checkpoint": policy["checkpoint"],
                    "is_resume": False,
                    "max_train_steps": int(args.max_train_steps),
                    "num_warmup_steps": max(1, int(args.max_train_steps) // 20),
                    "save_interval": int(args.max_train_steps),
                    "eval_interval": 100000000,
                    "logging_frequency": 20,
                    "gradient_accumulation_steps": 8,
                    "gradient_clipping": 1.0,
                    "max_grad_norm": 1.0,
                    "freeze_modules": "vlm,action_head",
                    "frozen_modules_eval": True,
                    "require_only_trainable_module": "raw_frontend",
                    "action_loss_weight": float(experiment.get("action_loss_weight", 0.1)),
                    "loss_scale": {"vla": 1.0},
                    "weight_decay": 1.0e-10,
                    "learning_rate": {"base": learning_rate, "raw_frontend": learning_rate},
                    "lr_scheduler_type": "cosine_with_min_lr",
                    "scheduler_specific_kwargs": {
                        "min_lr": float(
                            experiment.get(
                                "min_learning_rate", min(1.0e-6, learning_rate)
                            )
                        )
                    },
                    "save_format": "pt",
                    "save_trainable_only": True,
                    "optimizer": {
                        "name": "AdamW",
                        "betas": [0.9, 0.95],
                        "eps": 1.0e-8,
                        "weight_decay": 1.0e-10,
                    },
                },
            }
        )
        config_path = args.output_dir / f"{name}.yaml"
        OmegaConf.save(cfg, config_path)
        manifest.append({"gpu": gpu, **experiment, "config": str(config_path)})
    (args.output_dir / "matrix.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"experiments": len(manifest), "steps": args.max_train_steps}, indent=2))


if __name__ == "__main__":
    main()
