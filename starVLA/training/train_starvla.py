# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist

# NPU support: import torch_npu and enable automatic CUDA→NPU mapping.
# On GPU-only environments this is a no-op (ImportError is silently ignored).
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.config_loader import load_config
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self._frozen_eval_paths: tuple[str, ...] = ()

    @staticmethod
    def _resolve_module_path(model, path: str):
        module = model
        for attr in path.split("."):
            module = getattr(module, attr)
        return module

    def _enforce_frozen_modules_eval(self):
        """Disable frozen-backbone dropout without disabling input gradients."""
        if not self._frozen_eval_paths:
            return
        unwrapped = self.accelerator.unwrap_model(self.model)
        for path in self._frozen_eval_paths:
            self._resolve_module_path(unwrapped, path).eval()
        frontend = getattr(unwrapped, "raw_frontend", None)
        if frontend is not None and any(parameter.requires_grad for parameter in frontend.parameters()):
            frontend.train()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        required_trainable_module = str(
            self.config.trainer.get("require_only_trainable_module", "") or ""
        ).strip(".")
        if required_trainable_module:
            prefix = required_trainable_module + "."
            trainable_names = [
                name for name, parameter in self.model.named_parameters() if parameter.requires_grad
            ]
            unexpected = [
                name
                for name in trainable_names
                if name != required_trainable_module and not name.startswith(prefix)
            ]
            if not trainable_names or unexpected:
                raise RuntimeError(
                    "Trainable-parameter contract failed: expected only "
                    f"{required_trainable_module}, got unexpected={unexpected[:20]} "
                    f"trainable_count={len(trainable_names)}"
                )
            logger.info(
                "Trainable-parameter contract verified: module=%s tensors=%s",
                required_trainable_module,
                len(trainable_names),
            )
        if bool(self.config.trainer.get("frozen_modules_eval", False)):
            self._frozen_eval_paths = tuple(
                path.strip() for path in str(freeze_modules or "").split(",") if path.strip()
            )
            self._enforce_frozen_modules_eval()
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )
        self._enforce_frozen_modules_eval()

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases (best-effort; must not block training)."""
        self._wandb_enabled = False
        if os.environ.get("WANDB_MODE") == "disabled" or os.environ.get("WANDB_DISABLED", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            self.accelerator.wait_for_everyone()
            return
        if self.accelerator.is_main_process:
            try:
                wandb.init(
                    name=self.config.run_id,
                    dir=os.path.join(self.config.output_dir, "wandb"),
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    group="vla-train",
                )
                self._wandb_enabled = True
            except Exception as exc:
                logger.warning(f"W&B init failed; continuing without W&B: {exc}")
                self._wandb_enabled = False
        # Rendezvous after rank-0 W&B init. Otherwise a slow or failing init on
        # rank 0 lets the other ranks reach the first collective alone and
        # eventually hit an NCCL watchdog timeout.
        self.accelerator.wait_for_everyone()

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            save_trainable_only = bool(getattr(self.config.trainer, "save_trainable_only", False))
            if save_trainable_only:
                unwrapped = self.accelerator.unwrap_model(self.model)
                frontend = getattr(unwrapped, "raw_frontend", None)
                if frontend is None:
                    raise RuntimeError("save_trainable_only requires model.raw_frontend")
                state_dict = {key: value.detach().cpu() for key, value in frontend.state_dict().items()}
                checkpoint_path += "_raw_frontend"
            else:
                state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
            self._save_rawvla_visualization()

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _load_rawvla_visual_samples(self):
        """Load one deterministic mid-trajectory burst for each lighting domain."""
        if hasattr(self, "_rawvla_visual_samples"):
            return self._rawvla_visual_samples
        visual_cfg = getattr(self.config.trainer, "rawvla_visualization", None)
        if visual_cfg is None or not bool(visual_cfg.get("enabled", False)):
            self._rawvla_visual_samples = []
            return self._rawvla_visual_samples

        import glob

        cache_root = str(visual_cfg.get("cache_root"))
        rgb_cache_root = visual_cfg.get("rgb_cache_root", None)
        rgb_cache_root = None if rgb_cache_root is None else str(rgb_cache_root)
        burst_frames = int(visual_cfg.get("burst_frames", 6))
        rnn_frames = int(visual_cfg.get("rnn_frames", 6))
        observation_stride = int(visual_cfg.get("observation_stride", 1))
        image_size = int(visual_cfg.get("image_size", 224))
        wanted = ("ExtremeLow", "Low", "Normal", "Over", "ExtremeOver")
        selected = {}
        for path in sorted(glob.iglob(os.path.join(cache_root, "**", "*.npz"), recursive=True)):
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
                domain = metadata.get("lighting_domain")
                if domain not in wanted or domain in selected:
                    continue
                frames = []
                targets = []
                for key in ("agentview_raw_uint8", "wrist_raw_uint8"):
                    array = archive[key]
                    end = max(0, min(int(array.shape[0]) - 1, int(array.shape[0]) // 2))
                    observations = np.clip(
                        end - observation_stride * np.arange(rnn_frames - 1, -1, -1),
                        0,
                        array.shape[0] - 1,
                    )
                    burst_indices = np.clip(
                        observations[:, None] - np.arange(burst_frames - 1, -1, -1)[None, :],
                        0,
                        array.shape[0] - 1,
                    )
                    burst = np.asarray(array[burst_indices], dtype=np.float32) / 255.0
                    tensor = torch.from_numpy(burst).permute(0, 1, 4, 2, 3)
                    tensor = torch.nn.functional.interpolate(
                        tensor.flatten(0, 1),
                        size=(image_size, image_size),
                        mode="bilinear",
                        align_corners=False,
                    ).reshape(rnn_frames, burst_frames, 3, image_size, image_size)
                    frames.append(tensor)
                if rgb_cache_root is not None:
                    relative_path = os.path.relpath(path, cache_root)
                    rgb_path = os.path.join(rgb_cache_root, relative_path)
                    if not os.path.exists(rgb_path):
                        raise FileNotFoundError(
                            f"Missing paired default-lighting RGB visualization sample: {rgb_path}"
                        )
                    with np.load(rgb_path, allow_pickle=False) as rgb_archive:
                        for key in ("agentview_rgb_uint8", "wrist_rgb_uint8"):
                            array = np.asarray(rgb_archive[key][end], dtype=np.float32) / 255.0
                            tensor = torch.from_numpy(array).permute(2, 0, 1)[None]
                            tensor = torch.nn.functional.interpolate(
                                tensor,
                                size=(image_size, image_size),
                                mode="bilinear",
                                align_corners=False,
                            )[0]
                            targets.append(tensor)
                selected[domain] = {
                    "path": path,
                    "lighting_ev": float(metadata.get("lighting_ev", 0.0)),
                    "bursts": torch.stack(frames),
                    "targets": None if not targets else torch.stack(targets),
                }
            if len(selected) == len(wanted):
                break
        missing = [domain for domain in wanted if domain not in selected]
        if missing:
            raise RuntimeError(f"Missing RAWVLA visualization domains in {cache_root}: {missing}")
        self._rawvla_visual_samples = [(domain, selected[domain]) for domain in wanted]
        return self._rawvla_visual_samples

    @torch.no_grad()
    def _save_rawvla_visualization(self):
        """Save five-domain RAW/ISP/default-lighting-RGB comparisons.

        The paired RGB is evaluation-only: it is loaded inside this no-grad
        checkpoint hook and never enters the training objective.
        """
        samples = self._load_rawvla_visual_samples()
        if not samples:
            return
        from PIL import Image, ImageDraw

        unwrapped = self.accelerator.unwrap_model(self.model)
        frontend = getattr(unwrapped, "raw_frontend", None)
        if frontend is None:
            return
        was_training = frontend.training
        frontend.eval()
        rows = []
        records = []
        for domain, sample in samples:
            parameter = next(frontend.parameters())
            raw = sample["bursts"].to(device=parameter.device, dtype=parameter.dtype)
            rgb = frontend(raw, input_format="rgb_raw").float().cpu()
            current_raw = raw[:, -1, -1].float().cpu()
            targets = sample.get("targets", None)
            cells = []
            tensors = (
                (current_raw[0], rgb[0], targets[0], current_raw[1], rgb[1], targets[1])
                if targets is not None
                else (current_raw[0], rgb[0], current_raw[1], rgb[1])
            )
            for tensor in tensors:
                array = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
                cells.append(Image.fromarray(array, mode="RGB"))
            rows.append((domain, sample["lighting_ev"], cells))
            record = {
                "domain": domain,
                "lighting_ev": sample["lighting_ev"],
                "source": sample["path"],
                "agent_raw_mean": float(current_raw[0].mean()),
                "agent_isp_mean": float(rgb[0].mean()),
                "wrist_raw_mean": float(current_raw[1].mean()),
                "wrist_isp_mean": float(rgb[1].mean()),
            }
            if targets is not None:
                record.update(
                    {
                        "agent_target_mean": float(targets[0].mean()),
                        "agent_isp_target_mae": float((rgb[0] - targets[0]).abs().mean()),
                        "wrist_target_mean": float(targets[1].mean()),
                        "wrist_isp_target_mae": float((rgb[1] - targets[1]).abs().mean()),
                    }
                )
            records.append(record)
        if was_training:
            frontend.train()

        cell_w, cell_h = rows[0][2][0].size
        label_w, header_h = 150, 30
        column_count = len(rows[0][2])
        sheet = Image.new(
            "RGB", (label_w + column_count * cell_w, header_h + len(rows) * cell_h), "white"
        )
        draw = ImageDraw.Draw(sheet)
        titles = (
            ("agent RAW", "agent ISP", "agent target RGB", "wrist RAW", "wrist ISP", "wrist target RGB")
            if column_count == 6
            else ("agent RAW", "agent ISP", "wrist RAW", "wrist ISP")
        )
        for column, title in enumerate(titles):
            draw.text((label_w + column * cell_w + 8, 8), title, fill="black")
        for row_index, (domain, ev, cells) in enumerate(rows):
            y = header_h + row_index * cell_h
            draw.text((8, y + 8), f"{domain}\nEV {ev:+.2f}", fill="black")
            for column, cell in enumerate(cells):
                sheet.paste(cell, (label_w + column * cell_w, y))

        output_dir = Path(self.config.output_dir) / "isp_visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"step_{self.completed_steps:06d}"
        sheet.save(output_dir / f"{stem}.png")
        with open(output_dir / f"{stem}.json", "w") as handle:
            json.dump(records, handle, indent=2)

    def _log_metrics(self, metrics):
        """Record training metrics."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and rank == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            if getattr(self, "_wandb_enabled", False):
                try:
                    wandb.log(metrics, step=self.completed_steps)
                except Exception as exc:
                    self._wandb_enabled = False
                    logger.warning(f"W&B log failed; disabling W&B: {exc}")
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            did_optimizer_step = self.accelerator.sync_gradients
            if did_optimizer_step:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            # Checkpointing, evaluation and step-based logging are optimizer-
            # step events, not gradient-accumulation microstep events.
            if not did_optimizer_step:
                continue

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = self.accelerator.unwrap_model(self.model).predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        self._enforce_frozen_modules_eval()
        component_grad_norms = {}
        with self.accelerator.accumulate(self.model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                action_weight = float(self.config.trainer.get("action_loss_weight", 1.0))
                total_loss = action_weight * action_loss
                exposure_loss = output_dict.get("exposure_prior_loss", None)
                exposure_weight = 0.0
                raw_frontend_cfg = self.config.framework.get("raw_frontend", None)
                if exposure_loss is not None and raw_frontend_cfg is not None:
                    exposure_weight = float(raw_frontend_cfg.get("exposure_loss_weight", 0.01))
                    total_loss = total_loss + exposure_weight * exposure_loss
                chroma_loss = output_dict.get("chroma_prior_loss", None)
                chroma_weight = 0.0
                if chroma_loss is not None and raw_frontend_cfg is not None:
                    chroma_weight = float(raw_frontend_cfg.get("chroma_prior_weight", 0.0))
                    total_loss = total_loss + chroma_weight * chroma_loss
                chroma_supervision_loss = output_dict.get("chroma_supervision_loss", None)
                chroma_supervision_weight = 0.0
                if chroma_supervision_loss is not None and raw_frontend_cfg is not None:
                    chroma_supervision_weight = float(
                        raw_frontend_cfg.get("chroma_supervision_weight", 0.0)
                    )
                    total_loss = total_loss + chroma_supervision_weight * chroma_supervision_loss

            # Periodically measure how strongly each objective updates the ISP
            # itself. This catches an auxiliary loss that overwhelms (or never
            # reaches) the downstream action signal during short smoke runs.
            should_measure_grads = self.accelerator.sync_gradients and (
                (self.completed_steps + 1) % int(self.config.trainer.logging_frequency) == 0
            )
            if should_measure_grads:
                frontend = getattr(self.accelerator.unwrap_model(self.model), "raw_frontend", None)
                frontend_named_params = (
                    [(name, parameter) for name, parameter in frontend.named_parameters() if parameter.requires_grad]
                    if frontend is not None
                    else []
                )
                frontend_params = [parameter for _, parameter in frontend_named_params]

                def component_norm(loss):
                    gradients = torch.autograd.grad(
                        loss, frontend_params, retain_graph=True, allow_unused=True
                    )
                    squared = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
                    return torch.stack(squared).sum().sqrt() if squared else loss.new_zeros(())

                if frontend_params:
                    action_grads = torch.autograd.grad(
                        action_weight * action_loss,
                        frontend_params,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    action_raw_norm = component_norm(action_loss)
                    action_weighted_norm = torch.stack(
                        [gradient.detach().float().square().sum() for gradient in action_grads if gradient is not None]
                    ).sum().sqrt()
                    component_grad_norms["grad_norm/action_to_isp"] = action_raw_norm.item()
                    component_grad_norms["grad_norm/weighted_action_to_isp"] = action_weighted_norm.item()
                    if exposure_loss is not None:
                        exposure_grads = torch.autograd.grad(
                            exposure_weight * exposure_loss,
                            frontend_params,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        exposure_norm = torch.stack(
                            [gradient.detach().float().square().sum() for gradient in exposure_grads if gradient is not None]
                        ).sum().sqrt()
                        component_grad_norms["grad_norm/weighted_exposure_to_isp"] = exposure_norm.item()

                        def group_metrics(label, predicate):
                            selected = [
                                (action_gradient, exposure_gradient)
                                for (name, _), action_gradient, exposure_gradient in zip(
                                    frontend_named_params, action_grads, exposure_grads
                                )
                                if predicate(name)
                            ]
                            if not selected:
                                return
                            action_terms = [
                                action_gradient.detach().float().square().sum()
                                for action_gradient, _ in selected
                                if action_gradient is not None
                            ]
                            exposure_terms = [
                                exposure_gradient.detach().float().square().sum()
                                for _, exposure_gradient in selected
                                if exposure_gradient is not None
                            ]
                            action_group_norm = (
                                torch.stack(action_terms).sum().sqrt()
                                if action_terms
                                else action_loss.new_zeros(())
                            )
                            exposure_group_norm = (
                                torch.stack(exposure_terms).sum().sqrt()
                                if exposure_terms
                                else action_loss.new_zeros(())
                            )
                            component_grad_norms[f"grad_norm/weighted_action_to_{label}"] = action_group_norm.item()
                            component_grad_norms[f"grad_norm/weighted_exposure_to_{label}"] = exposure_group_norm.item()
                            pairs = [
                                (action_gradient, exposure_gradient)
                                for action_gradient, exposure_gradient in selected
                                if action_gradient is not None and exposure_gradient is not None
                            ]
                            if not pairs or action_group_norm.item() == 0.0 or exposure_group_norm.item() == 0.0:
                                return
                            dot = torch.stack(
                                [
                                    (action_gradient.detach().float() * exposure_gradient.detach().float()).sum()
                                    for action_gradient, exposure_gradient in pairs
                                ]
                            ).sum()
                            cosine = dot / (action_group_norm * exposure_group_norm).clamp_min(1.0e-20)
                            component_grad_norms[f"grad_cosine/action_exposure_{label}"] = cosine.item()

                        group_metrics(
                            "luma",
                            lambda name: any(
                                token in name
                                for token in (
                                    "luma_",
                                    "spatial_encoder",
                                    "spatial_pool",
                                    "exposure_head",
                                    "tone_head",
                                    "denoise_head",
                                )
                            ),
                        )
                        group_metrics("chroma", lambda name: "chroma_" in name)

                    if chroma_loss is not None:
                        chroma_grads = torch.autograd.grad(
                            chroma_weight * chroma_loss,
                            frontend_params,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        chroma_terms = [
                            gradient.detach().float().square().sum()
                            for gradient in chroma_grads
                            if gradient is not None
                        ]
                        chroma_norm = (
                            torch.stack(chroma_terms).sum().sqrt()
                            if chroma_terms
                            else chroma_loss.new_zeros(())
                        )
                        component_grad_norms["grad_norm/weighted_chroma_prior_to_isp"] = chroma_norm.item()

                        group_predicates = {
                            "luma": lambda name: any(
                                token in name
                                for token in (
                                    "luma_",
                                    "spatial_encoder",
                                    "spatial_pool",
                                    "exposure_head",
                                    "tone_head",
                                    "denoise_head",
                                )
                            ),
                            "chroma": lambda name: "chroma_" in name,
                        }
                        for label, predicate in group_predicates.items():
                            selected = [
                                (action_gradient, chroma_gradient)
                                for (name, _), action_gradient, chroma_gradient in zip(
                                    frontend_named_params, action_grads, chroma_grads
                                )
                                if predicate(name)
                            ]
                            action_terms = [
                                action_gradient.detach().float().square().sum()
                                for action_gradient, _ in selected
                                if action_gradient is not None
                            ]
                            prior_terms = [
                                chroma_gradient.detach().float().square().sum()
                                for _, chroma_gradient in selected
                                if chroma_gradient is not None
                            ]
                            action_group_norm = (
                                torch.stack(action_terms).sum().sqrt()
                                if action_terms
                                else action_loss.new_zeros(())
                            )
                            prior_group_norm = (
                                torch.stack(prior_terms).sum().sqrt()
                                if prior_terms
                                else chroma_loss.new_zeros(())
                            )
                            component_grad_norms[f"grad_norm/weighted_chroma_prior_to_{label}"] = (
                                prior_group_norm.item()
                            )
                            pairs = [
                                (action_gradient, chroma_gradient)
                                for action_gradient, chroma_gradient in selected
                                if action_gradient is not None and chroma_gradient is not None
                            ]
                            if pairs and action_group_norm.item() > 0.0 and prior_group_norm.item() > 0.0:
                                dot = torch.stack(
                                    [
                                        (action_gradient.detach().float() * chroma_gradient.detach().float()).sum()
                                        for action_gradient, chroma_gradient in pairs
                                    ]
                                ).sum()
                                cosine = dot / (action_group_norm * prior_group_norm).clamp_min(1.0e-20)
                                component_grad_norms[f"grad_cosine/action_chroma_prior_{label}"] = cosine.item()

            self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
            # AcceleratedOptimizer.zero_grad() is a no-op on intermediate
            # accumulation microsteps and clears gradients after the synced
            # optimizer step. Calling it at the start would erase accumulated
            # gradients immediately before the final microstep.
            self.optimizer.zero_grad()

        metrics = {
            "action_dit_loss": action_loss.item(),
            "weighted_action_loss": action_weight * action_loss.item(),
            "total_loss": total_loss.item(),
        }
        metrics.update(component_grad_norms)
        if exposure_loss is not None:
            metrics["exposure_prior_loss"] = exposure_loss.item()
            metrics["weighted_exposure_prior_loss"] = exposure_weight * exposure_loss.item()
            for key in (
                "isp_output_mean",
                "isp_exposure_ev_mean",
                "isp_update_gate_mean",
                "isp_saturation_ratio",
            ):
                value = output_dict.get(key, None)
                if value is not None:
                    metrics[key] = value.item()
        if chroma_loss is not None:
            metrics["chroma_prior_loss"] = chroma_loss.item()
            metrics["weighted_chroma_prior_loss"] = chroma_weight * chroma_loss.item()
            for key in ("isp_log_rg_mean", "isp_log_bg_mean"):
                value = output_dict.get(key, None)
                if value is not None:
                    metrics[key] = value.item()
        if chroma_supervision_loss is not None:
            metrics["chroma_supervision_loss"] = chroma_supervision_loss.item()
            metrics["weighted_chroma_supervision_loss"] = (
                chroma_supervision_weight * chroma_supervision_loss.item()
            )
            for key in (
                "chroma_supervision_sample_weight_mean",
                "chroma_supervision_highlight_loss",
                "chroma_supervision_highlight_weight_mean",
                "chroma_supervision_valid_ratio",
                "chroma_target_mean",
            ):
                value = output_dict.get(key, None)
                if value is not None:
                    metrics[key] = value.item()
        return metrics

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            save_trainable_only = bool(getattr(self.config.trainer, "save_trainable_only", False))
            if save_trainable_only:
                unwrapped = self.accelerator.unwrap_model(self.model)
                frontend = getattr(unwrapped, "raw_frontend", None)
                if frontend is None:
                    raise RuntimeError("save_trainable_only requires model.raw_frontend")
                state_dict = {key: value.detach().cpu() for key, value in frontend.state_dict().items()}
            else:
                state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process and getattr(self, "_wandb_enabled", False):
            try:
                wandb.finish()
            except Exception:
                pass

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = load_config(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_tighten.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
