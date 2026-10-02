# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Policy server wrapper.

Encapsulates a `baseframework` instance plus a :class:`PolicyNormProcessor`
that reuses the *training-time* :class:`ComposedModalityTransform` for action
un-normalization (no hand-rolled math). The websocket server returns
already-unnormalized actions.

Client-side responsibilities that REMAIN on the client:
  - environment-specific adapters (image_history, gripper sticky, action
    ensembling)
  - chunk-cache scheduling (`step % chunk_size == 0` triggers a new infer)

Exposed API:
  - ``metadata`` (dict, sent at handshake): ``action_chunk_size``,
    ``available_unnorm_keys``, ``action_keys``, ``state_keys``.
  - ``predict_action(examples, unnorm_key=None, **kwargs)`` returns
    ``{"actions": np.ndarray[B, T, action_dim]}``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework, merge_config_overrides
from starVLA.model.framework.share_tools import read_mode_config

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from deployment.model_server.tools.image_tools import audit_image_once, validate_image_array


def _training_obs_image_size(model_cfg: Dict[str, Any]) -> Optional[List[int]]:
    """Return training image size metadata when it is explicitly configured.

    This is only a consistency check for eval clients. It is not used to infer
    camera order or to rewrite eval observations.
    """
    vla_data_cfg = model_cfg.get("datasets", {}).get("vla_data", {})
    size = vla_data_cfg.get("obs_image_size") or vla_data_cfg.get("image_size")
    if size is None and "default_image_resolution" in vla_data_cfg:
        default_resolution = vla_data_cfg["default_image_resolution"]
        if isinstance(default_resolution, (list, tuple)) and len(default_resolution) >= 2:
            size = default_resolution[-2:]
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return None
    return [int(size[0]), int(size[1])]


class PolicyServerWrapper:
    """Wraps a `baseframework` for use as a websocket-server policy."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
        config_overrides: Sequence[str] | None = None,
    ) -> None:
        self._ckpt_path = str(ckpt_path)

        logging.info("PolicyServerWrapper: loading framework from %s", self._ckpt_path)
        framework = baseframework.from_pretrained(self._ckpt_path, config_overrides=config_overrides)
        qwen_sharded = os.getenv("RAWVLA_QWEN_SHARD", "").strip().lower() in {"1", "true", "yes", "on"}
        skip_framework_to = os.getenv("RAWVLA_SKIP_FRAMEWORK_TO", "").strip().lower() in {"1", "true", "yes", "on"}
        if qwen_sharded:
            target_dtype = torch.bfloat16 if use_bf16 else None
            action_device = os.getenv("RAWVLA_QWEN_ACTION_DEVICE", device)
            for child_name, child in framework.named_children():
                if child_name == "qwen_vl_interface":
                    continue
                if child_name == "raw_frontend":
                    child.to(device=action_device, dtype=torch.float32)
                    continue
                if target_dtype is not None:
                    child.to(device=action_device, dtype=target_dtype)
                else:
                    child.to(device=action_device)
            framework.eval()
            logging.info("[Qwen3 shard] kept qwen_vl_interface dispatch map; moved other modules to %s", action_device)
        elif skip_framework_to:
            target_dtype = torch.bfloat16 if use_bf16 else None
            action_device = os.getenv("RAWVLA_ACTION_DEVICE", device)
            for child_name, child in framework.named_children():
                if child_name == "backbone":
                    if target_dtype is not None:
                        child.to(dtype=target_dtype)
                    continue
                if child_name == "raw_frontend":
                    child.to(device=action_device, dtype=torch.float32)
                    continue
                if target_dtype is not None:
                    child.to(device=action_device, dtype=target_dtype)
                else:
                    child.to(device=action_device)
            framework.eval()
            logging.info(
                "RAWVLA_SKIP_FRAMEWORK_TO enabled; preserved backbone placement and moved other modules to %s",
                action_device,
            )
        else:
            if use_bf16:
                framework = framework.to(torch.bfloat16)
            framework = framework.to(device).eval()
            raw_frontend = getattr(framework, "raw_frontend", None)
            if raw_frontend is not None:
                # RAW frontends were trained in fp32; RAWVLA also contains FFT
                # operations that do not support bfloat16 on all CUDA stacks.
                raw_frontend.to(device=device, dtype=torch.float32)
        self._framework = framework

        # Co-located metadata.
        model_cfg, norm_stats = read_mode_config(self._ckpt_path)
        model_cfg = merge_config_overrides(model_cfg, config_overrides)
        self._model_cfg = model_cfg

        # action_chunk_size = future_action_window_size + 1 (matches old client).
        action_model_cfg = model_cfg["framework"]["action_model"]

        if "action_horizon" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["action_horizon"])
        elif "future_action_window_size" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
        else:
            raise ValueError(
                f"PolicyServerWrapper: no action_horizon or future_action_window_size found in model config for {self._ckpt_path}"
            )
        # Cache of PolicyNormProcessor instances per unnorm_key.
        # For single-dataset ckpts unnorm_key is auto-selected; for multi-dataset
        # ckpts clients must pass unnorm_key per request.
        self._default_unnorm_key = unnorm_key
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}

        # Peek at available keys without building a full processor.
        self._available_unnorm_keys: List[str] = list(norm_stats.keys())

        # Eagerly build when unambiguous; defer for multi-key / no explicit key.
        if unnorm_key is not None or len(self._available_unnorm_keys) == 1:
            default_proc = self._get_processor(unnorm_key)
            self._default_unnorm_key = default_proc.unnorm_key
            logging.info(
                "PolicyServerWrapper ready: action_chunk_size=%d, default_unnorm_key=%s, "
                "available_unnorm_keys=%s, action_keys=%s, state_keys=%s",
                self._action_chunk_size,
                default_proc.unnorm_key,
                default_proc.available_unnorm_keys,
                default_proc.action_keys,
                default_proc.state_keys,
            )
        else:
            logging.info(
                "PolicyServerWrapper ready (multi-key): action_chunk_size=%d, "
                "available_unnorm_keys=%s — clients must pass unnorm_key per request.",
                self._action_chunk_size,
                self._available_unnorm_keys,
            )

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            self._norm_processors[cache_key] = PolicyNormProcessor(
                self._ckpt_path, unnorm_key=unnorm_key
            )
        return self._norm_processors[cache_key]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model-invariant metadata; sent to client at websocket handshake."""
        raw_frontend_cfg = self._model_cfg.get("framework", {}).get("raw_frontend") or {}
        base = {
            "env": "starvla_policy_server",
            "ckpt_path": self._ckpt_path,
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
            "training_data_mix": self._model_cfg.get("datasets", {}).get("vla_data", {}).get("data_mix"),
            "training_obs_image_size": _training_obs_image_size(self._model_cfg),
            "eval_image_contract": (
                "Eval clients must explicitly choose image count and order. "
                "The server does not infer or reorder camera views from training config."
            ),
            "accepted_transport_image_formats": ["uint8_0_255", "float32_0_1"],
            "raw_frontend": str(raw_frontend_cfg.get("name", "none") or "none"),
            "raw_frontend_source_key": str(raw_frontend_cfg.get("source_key", "image")),
            "raw_frontend_burst_frames": int(raw_frontend_cfg.get("burst_frames", 1)),
            "raw_frontend_rnn_frames": int(raw_frontend_cfg.get("rnn_frames", 1)),
        }
        # Enrich with per-embodiment keys when a default processor already exists.
        if self._default_unnorm_key is not None:
            proc = self._get_processor(self._default_unnorm_key)
            base["action_keys"] = proc.action_keys
            base["state_keys"] = proc.state_keys
        return base

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Run the framework, then un-normalize via training-time transforms.

        Args:
            examples: list of dicts (each with ``image`` / ``lang`` / optional ``state``).
            unnorm_key: dataset key for un-normalization stats. ``None`` -->
                use the wrapper's default (auto-picked at startup).
            **kwargs: forwarded to the framework's ``predict_action``
                (``do_sample``, ``use_ddim``, ``num_ddim_steps``, ...).

        Returns:
            ``{"actions": np.ndarray[B, T, D]}`` -- un-normalized.
        """
        require_float_images = os.getenv("STARVLA_REQUIRE_FLOAT_IMAGE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        for example_idx, example in enumerate(examples):
            images = example.get("image")
            images = images if isinstance(images, (list, tuple)) else [images]
            for image_idx, image in enumerate(images):
                if image is None:
                    raise ValueError(
                        f"Missing image at example={example_idx}, image_index={image_idx}"
                    )
                try:
                    image_array = validate_image_array(image)
                    if require_float_images and image_array.dtype != np.float32:
                        raise TypeError(
                            "STARVLA_REQUIRE_FLOAT_IMAGE=1 requires float32 [0, 1] images, "
                            f"got dtype={image_array.dtype} at example={example_idx}, image_index={image_idx}"
                        )
                    audit_image_once("policy_server_received", image)
                except (TypeError, ValueError) as exc:
                    raise type(exc)(
                        f"Invalid image at example={example_idx}, image_index={image_idx}: {exc}"
                    ) from exc

        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    f"predict_action: unnorm_key not specified and no default set. "
                    f"Pass one of {self._available_unnorm_keys}."
                )
        proc = self._get_processor(effective_key)

        with torch.inference_mode():
            out = self._framework.predict_action(examples=examples, **kwargs)
        normalized = np.asarray(out["normalized_actions"])  # (B, T, D)

        unnorm = np.stack(
            [proc.unapply_actions(normalized[b]) for b in range(normalized.shape[0])],
            axis=0,
        )
        del out, normalized
        if torch.cuda.is_available() and os.getenv("RAWVLA_EMPTY_CACHE_EACH_STEP", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }:
            torch.cuda.empty_cache()
        return {"actions": unnorm}
