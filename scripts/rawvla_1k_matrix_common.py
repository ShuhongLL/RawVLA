"""Shared configuration builder for controlled RAWVLA training matrices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf

from rawvla_train_data import RAWVLA_TRAIN_DATA_VERSION, rawvla_train_roots


def build_matrix(
    *,
    repo_root: Path,
    asset_root: Path,
    output_dir: Path,
    result_root: Path,
    project: str,
    experiments: Iterable[dict[str, Any]],
    max_train_steps: int = 1000,
) -> None:
    root = repo_root.resolve()
    asset = asset_root.resolve()
    star = root / "starVLA"
    pretrained = asset / "starVLA/playground/Pretrained_models"
    raw_cache, rgb_cache = rawvla_train_roots(asset)
    openpi = asset / "openpi_converted_protocol"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    backbones = {
        "qwen3_oft": {
            "source": pretrained / "StarVLA/Qwen3-VL-OFT-LIBERO-4in1/config.yaml",
            "checkpoint": pretrained / "StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt",
            "freeze": "qwen_vl_interface,action_model",
            "horizon": 8,
            "stride": 8,
            "normalization": "starvla_minmax",
            "stats": pretrained / "StarVLA/Qwen3-VL-OFT-LIBERO-4in1/dataset_statistics.json",
        },
        "qwen3_pi": {
            "source": pretrained / "StarVLA/Qwen3-VL-PI-LIBERO-4in1/config.yaml",
            "checkpoint": pretrained / "StarVLA/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt",
            "freeze": "qwen_vl_interface,action_model",
            "horizon": 8,
            "stride": 8,
            "normalization": "starvla_minmax",
            "stats": pretrained / "StarVLA/Qwen3-VL-PI-LIBERO-4in1/dataset_statistics.json",
        },
        "wm4a_cosmos": {
            "source": pretrained / "StarVLA/WM4A-CosmoPredict-GR00T-LIBERO-4in1/config.yaml",
            "checkpoint": pretrained / "StarVLA/WM4A-CosmoPredict-GR00T-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt",
            "freeze": "backbone,action_model",
            "horizon": 8,
            "stride": 8,
            "normalization": "starvla_minmax",
            "stats": pretrained / "StarVLA/WM4A-CosmoPredict-GR00T-LIBERO-4in1/dataset_statistics.json",
        },
        "wm4a_wan": {
            "source": pretrained / "StarVLA/WM4A-Wan2d2-OFT-LIBERO-4in1/config.yaml",
            "checkpoint": pretrained / "StarVLA/WM4A-Wan2d2-OFT-LIBERO-4in1/checkpoints/steps_60000_pytorch_model.pt",
            "freeze": "backbone,action_model,action_query_proj",
            "horizon": 8,
            "stride": 8,
            "normalization": "starvla_minmax",
            "stats": pretrained / "StarVLA/WM4A-Wan2d2-OFT-LIBERO-4in1/dataset_statistics.json",
        },
        "pi0": {
            "source": star / "examples/simBenchmarks/LIBERO/train_files/openpi/pi0_libero_8gpu.yaml",
            "checkpoint": openpi / "pi0_libero_starvla/bfloat16/model.safetensors",
            "freeze": "vlm,action_head",
            "horizon": 50,
            "stride": 5,
            "normalization": "pi0_zscore",
            "stats": openpi / "pi0_libero_starvla/bfloat16/assets/physical-intelligence/libero/norm_stats.json",
        },
        "pi05": {
            "source": star / "examples/simBenchmarks/LIBERO/train_files/openpi/pi05_libero_8gpu.yaml",
            "checkpoint": openpi / "pi05_libero_starvla/bfloat16/model.safetensors",
            "freeze": "vlm,action_head",
            "horizon": 10,
            "stride": 5,
            "normalization": "pi05_quantile",
            "stats": openpi / "pi05_libero_starvla/bfloat16/assets/physical-intelligence/libero/norm_stats.json",
        },
    }

    manifest = []
    for gpu, experiment in enumerate(experiments):
        spec = backbones[experiment["base"]]
        for required in (spec["source"], spec["checkpoint"], spec["stats"], raw_cache):
            if not Path(required).exists():
                raise FileNotFoundError(required)
        cfg = OmegaConf.load(spec["source"])
        cfg.run_id = experiment["name"]
        cfg.run_root_dir = str(result_root)
        cfg.is_debug = False
        cfg.wandb_entity = "your_wandb_entity"
        cfg.wandb_project = project

        rnn_frames = int(experiment.get("rnn_frames", 6))
        state_dim = int(experiment.get("state_dim", 128))
        luma_state_dim = int(experiment.get("luma_state_dim", state_dim // 2))
        cfg.framework.raw_frontend = {
            "name": "rawvla",
            "source_key": "raw_burst_float32",
            "input_format": "rgb_raw",
            "trainable": True,
            "burst_frames": 6,
            "rnn_frames": rnn_frames,
            "spatial_width": 32,
            "hist_bins": 64,
            "state_dim": state_dim,
            "luma_state_dim": luma_state_dim,
            "fft_patch_size": 32,
            "fft_stride": 16,
            "use_local_color": False,
            "use_local_tone": False,
            "split_luma_chroma_condition": True,
            "luminance_weights": [1.0, 1.0, 1.0],
            "luma_spatial_input": "luma",
            "exposure_target": 0.48,
            "exposure_loss_weight": 0.01,
            "exposure_loss_type": "two_sided_l1",
            "exposure_objective_clamp": experiment.get("exposure_objective_clamp", "ste"),
            "max_exposure_ev": 6.0,
            "fixed_update_alpha": experiment.get("fixed_update_alpha", 1.0),
            "wb_anchor_ev": experiment.get("wb_anchor_ev", [0.0, 0.0, 0.0]),
            "wb_residual_scale_ev": float(experiment.get("wb_residual_scale_ev", 1.0)),
            "gray_preserving_ccm": bool(experiment.get("gray_preserving_ccm", False)),
            "chroma_prior_enabled": bool(experiment.get("chroma_prior_enabled", True)),
            "chroma_prior_mode": experiment.get("chroma_prior_mode", "output_envelope"),
            "wb_anchor_loss_scale_ev": float(experiment.get("wb_anchor_loss_scale_ev", 0.5)),
            "chroma_prior_descriptor": experiment.get("chroma_prior_descriptor", "mean_log"),
            "chroma_prior_bias": experiment.get("chroma_prior_bias", [0.0, 0.0]),
            "chroma_prior_weight": float(experiment.get("chroma_prior_weight", 0.001)),
            "chroma_prior_lower": experiment.get(
                "chroma_prior_lower", [0.025407744850963355, -0.35791672915220263]
            ),
            "chroma_prior_upper": experiment.get(
                "chroma_prior_upper", [0.25704274773597713, -0.0317848339676857]
            ),
            "chroma_prior_margin": float(experiment.get("chroma_prior_margin", 0.10)),
        }
        if experiment["base"].startswith("qwen3"):
            cfg.framework.qwenvl.base_vlm = str(pretrained / "Qwen3-VL-4B-Instruct")
            cfg.framework.qwenvl.attn_implementation = "sdpa"
        if experiment["base"] == "qwen3_pi":
            cfg.framework.action_model.repeated_diffusion_steps = 8
        if experiment["base"] == "wm4a_cosmos":
            cfg.framework.world_model.base_wm = str(pretrained / "nvidia/Cosmos-Predict2-2B-Video2World")
            cfg.framework.world_model.vae_posterior = "mode"
        if experiment["base"] == "wm4a_wan":
            cfg.framework.world_model.base_wm = str(pretrained / "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            cfg.framework.world_model.vae_posterior = "mode"
        if experiment["base"] in {"pi0", "pi05"}:
            cfg.framework.tokenizer.model_path = str(openpi / "paligemma_tokenizer.model")

        cfg.datasets = OmegaConf.create(
            {
                "vla_data": {
                    "dataset_py": "rawvla_npz_datasets",
                    "data_mix": "rawvla_light_npz",
                    "dataset_version": RAWVLA_TRAIN_DATA_VERSION,
                    "cache_root": str(raw_cache),
                    "burst_frames": 6,
                    "rnn_frames": rnn_frames,
                    "observation_stride": int(spec["stride"]),
                    "action_horizon": int(spec["horizon"]),
                    "samples_per_episode": 32,
                    "virtual_length": 100000,
                    "normalization": spec["normalization"],
                    "stats_path": str(spec["stats"]),
                    "per_device_batch_size": 1,
                    "num_workers": 1,
                    "prefetch_factor": 1,
                    "persistent_workers": True,
                    "pin_memory": True,
                }
            }
        )
        cfg.trainer.pretrained_checkpoint = str(spec["checkpoint"])
        cfg.trainer.is_resume = False
        cfg.trainer.max_train_steps = int(max_train_steps)
        cfg.trainer.num_warmup_steps = max(50, int(max_train_steps) // 20)
        cfg.trainer.save_interval = 1000
        cfg.trainer.eval_interval = 100000000
        cfg.trainer.logging_frequency = 20
        cfg.trainer.gradient_accumulation_steps = 8
        cfg.trainer.gradient_clipping = 1.0
        cfg.trainer.max_grad_norm = 1.0
        cfg.trainer.freeze_modules = spec["freeze"]
        cfg.trainer.frozen_modules_eval = True
        cfg.trainer.action_loss_weight = float(experiment["action_loss_weight"])
        cfg.trainer.learning_rate = {"base": 1.0e-4, "raw_frontend": 1.0e-4}
        cfg.trainer.lr_scheduler_type = "cosine_with_min_lr"
        cfg.trainer.scheduler_specific_kwargs = {"min_lr": 1.0e-6}
        cfg.trainer.save_format = "pt"
        cfg.trainer.save_trainable_only = True
        cfg.trainer.rawvla_visualization = {
            "enabled": True,
            "cache_root": str(raw_cache),
            "rgb_cache_root": str(rgb_cache),
            "burst_frames": 6,
            "rnn_frames": rnn_frames,
            "observation_stride": int(spec["stride"]),
            "image_size": 224,
        }
        config_path = output_dir / f"{experiment['name']}.yaml"
        OmegaConf.save(cfg, config_path)
        manifest.append({"gpu": gpu, **experiment, "config": str(config_path)})

    if not 1 <= len(manifest) <= 8:
        raise ValueError(f"Expected between one and eight experiments, got {len(manifest)}")
    (output_dir / "matrix.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(manifest)} configs to {output_dir}")
