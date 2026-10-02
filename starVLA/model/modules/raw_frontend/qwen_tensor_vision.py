"""Differentiable Qwen vision packing for trainable RAW frontends."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F

from .image_contract import as_rgb_chw_float01, assert_rgb_float01


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_chw_float01(image: Any) -> Tensor:
    return as_rgb_chw_float01(image, name="RAW frontend image")


def _as_raw_frontend_tensor(image: Any) -> Tensor:
    """Convert an image/burst/recurrent-burst sequence to channel-first form."""

    if isinstance(image, Image.Image) or getattr(image, "ndim", None) == 3:
        return _as_chw_float01(image)
    if isinstance(image, np.ndarray):
        tensor = torch.from_numpy(np.asarray(image).copy())
    elif torch.is_tensor(image):
        tensor = image
    else:
        raise TypeError(f"Unsupported RAW frontend image type: {type(image)!r}")
    if tensor.ndim not in {4, 5}:
        raise ValueError(f"Expected image, burst, or recurrent bursts, got shape {tuple(tensor.shape)}")
    if tensor.shape[-3] in {1, 3, 4} and tensor.shape[-1] not in {1, 3, 4}:
        channel_first = tensor[..., :3, :, :]
    elif tensor.shape[-1] in {1, 3, 4}:
        if tensor.ndim == 4:
            channel_first = tensor.permute(0, 3, 1, 2)[..., :3, :, :]
        else:
            channel_first = tensor.permute(0, 1, 4, 2, 3)[..., :3, :, :]
    else:
        raise ValueError(f"Cannot identify channel axis for RAW burst shape {tuple(tensor.shape)}")
    if channel_first.shape[-3] == 1:
        repeats = [1] * channel_first.ndim
        repeats[-3] = 3
        channel_first = channel_first.repeat(*repeats)
    if channel_first.is_floating_point():
        channel_first = channel_first.to(dtype=torch.float32)
    else:
        if channel_first.dtype != torch.uint8:
            raise TypeError(f"Integer RAW input must be uint8, got {channel_first.dtype}")
        channel_first = channel_first.to(dtype=torch.float32).div(255.0)
    if not bool(torch.isfinite(channel_first.detach()).all()):
        raise ValueError("RAW frontend input contains NaN or Inf")
    minimum = float(channel_first.detach().amin().cpu())
    maximum = float(channel_first.detach().amax().cpu())
    if minimum < -1.0e-6 or maximum > 1.0 + 1.0e-6:
        raise ValueError(
            f"Floating RAW input must already be in [0, 1], got min={minimum:.8g}, max={maximum:.8g}"
        )
    return channel_first.contiguous()


def _to_processor_layout_image(image: Any) -> Any:
    """Create a processor-safe image with the same spatial layout.

    Qwen processors only need this path to derive image placeholders and
    image_grid_thw. The differentiable pixel values are replaced later from
    the original RAW/float image leaves, so uint8 conversion here does not
    affect the trainable frontend.
    """

    if isinstance(image, Image.Image):
        return image
    # The processor only establishes the current observation's spatial token
    # layout; differentiable pixels are replaced after RAWVLA processing.
    while getattr(image, "ndim", None) is not None and image.ndim > 3:
        image = image[-1]
    chw = _as_chw_float01(image)
    hwc = chw.permute(1, 2, 0).detach().cpu().numpy()
    return Image.fromarray(np.rint(hwc * 255.0).astype(np.uint8), mode="RGB")


def _images_for_processor_layout(batch_images: Sequence[Sequence[Any]]) -> list[Any]:
    layout_images: list[Any] = []
    for sample in batch_images:
        images = sample if isinstance(sample, (list, tuple)) else [sample]
        layout_images.append([_to_processor_layout_image(image) for image in images])
    return layout_images


def _flatten_images(batch_images: Sequence[Sequence[Any]]) -> list[Any]:
    flat: list[Any] = []
    for sample in batch_images:
        images = sample if isinstance(sample, (list, tuple)) else [sample]
        flat.extend(images)
    return flat


def _normalize_image_batch(images: Any) -> list[Any]:
    if isinstance(images, (list, tuple)):
        return list(images)
    return [images]


def select_raw_frontend_images(
    examples: Sequence[dict],
    raw_frontend_cfg: Any = None,
    *,
    default_key: str = "image_float32",
) -> list[list[Any]]:
    """Select the image stream consumed by the trainable RAW frontend.

    By default this prefers a strict float32 [0, 1] stream under
    ``example["image_float32"]``. Existing datasets that only expose
    ``example["image"]`` still work through the final fallback.
    """

    source_key = str(_cfg_get(raw_frontend_cfg, "source_key", default_key))
    fallback_keys = _cfg_get(
        raw_frontend_cfg,
        "fallback_source_keys",
        ("float32_image", "raw_float32", "raw_image_float32", "raw_image", "image"),
    )
    if isinstance(fallback_keys, str):
        fallback_keys = tuple(key.strip() for key in fallback_keys.split(",") if key.strip())
    keys = (source_key, *tuple(fallback_keys))

    selected: list[list[Any]] = []
    for example in examples:
        for key in keys:
            if key in example and example[key] is not None:
                selected.append(_normalize_image_batch(example[key]))
                break
        else:
            raise KeyError(f"No RAW frontend image source found. Tried keys={keys}")
    return selected


def apply_raw_frontend_to_images(
    batch_images: Sequence[Sequence[Any]],
    *,
    raw_frontend: nn.Module | None,
    raw_frontend_cfg: Any = None,
    raw_frontend_metadata: Sequence[dict[str, Any] | None] | None = None,
    chroma_targets: Sequence[Sequence[Any]] | None = None,
) -> list[list[Tensor]] | Sequence[Sequence[Any]]:
    """Run a RAW frontend while preserving the nested sample/view layout.

    Each leaf may be one RGB RAW frame or a causal ``[K, ...]`` burst. The
    returned leaves are differentiable CHW RGB tensors in ``[0, 1]`` and can
    be consumed directly by tensor-aware vision/VAE preprocessors.
    """

    if raw_frontend is None:
        return batch_images
    normalized = [_normalize_image_batch(sample) for sample in batch_images]
    counts = [len(sample) for sample in normalized]
    flat_images = _flatten_images(normalized)
    frontend_param = next(raw_frontend.parameters(), None)
    device = frontend_param.device if frontend_param is not None else torch.device("cpu")
    raw = torch.stack([_as_raw_frontend_tensor(image) for image in flat_images], dim=0).to(device=device)
    input_format = str(_cfg_get(raw_frontend_cfg, "input_format", "rgb_raw"))
    lighting_domains = None
    if raw_frontend_metadata is not None:
        if len(raw_frontend_metadata) != len(normalized):
            raise ValueError("raw_frontend_metadata must have one entry per batch sample")
        lighting_domains = []
        for sample, metadata in zip(normalized, raw_frontend_metadata):
            domain = "" if metadata is None else str(metadata.get("lighting_domain", ""))
            lighting_domains.extend([domain] * len(sample))
    flat_chroma_targets = None
    if chroma_targets is not None:
        normalized_targets = [_normalize_image_batch(sample) for sample in chroma_targets]
        target_counts = [len(sample) for sample in normalized_targets]
        if target_counts != counts:
            raise ValueError(
                f"Chroma target view counts {target_counts} do not match RAW view counts {counts}"
            )
        flat_chroma_targets = torch.stack(
            [_as_raw_frontend_tensor(image) for image in _flatten_images(normalized_targets)], dim=0
        ).to(device=device, dtype=torch.float32)
    rgb = raw_frontend(
        raw,
        input_format=input_format,
        lighting_domains=lighting_domains,
        chroma_targets=flat_chroma_targets,
    ).to(dtype=torch.float32)
    assert_rgb_float01(rgb, name="RAW frontend policy output")

    nested: list[list[Tensor]] = []
    offset = 0
    for count in counts:
        nested.append([rgb[index] for index in range(offset, offset + count)])
        offset += count
    return nested


def _processor_attr(image_processor: Any, name: str, default: Any) -> Any:
    return getattr(image_processor, name, default)


def _pack_one_qwen_image(
    image: Tensor,
    grid_thw: Tensor,
    *,
    image_processor: Any,
    dtype: torch.dtype,
) -> Tensor:
    patch_size = int(_processor_attr(image_processor, "patch_size", 14))
    temporal_patch_size = int(_processor_attr(image_processor, "temporal_patch_size", 2))
    merge_size = int(_processor_attr(image_processor, "merge_size", 2))
    image_mean = _processor_attr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
    image_std = _processor_attr(image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])

    grid_t, grid_h, grid_w = [int(x) for x in grid_thw.detach().cpu().tolist()]
    target_h = grid_h * patch_size
    target_w = grid_w * patch_size
    image = F.interpolate(
        image.unsqueeze(0),
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0).clamp(0.0, 1.0)

    mean = torch.tensor(image_mean, device=image.device, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(image_std, device=image.device, dtype=image.dtype).view(3, 1, 1)
    image = (image - mean) / std

    frames = image.unsqueeze(0).repeat(grid_t * temporal_patch_size, 1, 1, 1)
    channel = frames.shape[1]
    patches = frames.view(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return patches.reshape(
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    ).to(dtype=dtype)


def replace_qwen_pixel_values_with_raw_frontend(
    *,
    qwen_vl_interface: nn.Module,
    qwen_inputs: dict,
    batch_images: Sequence[Sequence[Any]],
    raw_frontend: nn.Module | None,
    raw_frontend_cfg: Any = None,
    raw_frontend_metadata: Sequence[dict[str, Any] | None] | None = None,
) -> dict:
    if raw_frontend is None:
        return qwen_inputs
    if "pixel_values" not in qwen_inputs or "image_grid_thw" not in qwen_inputs:
        raise ValueError("Qwen inputs do not contain pixel_values/image_grid_thw for RAW frontend replacement.")

    flat_images = _flatten_images(batch_images)
    image_grid_thw = qwen_inputs["image_grid_thw"]
    if len(flat_images) != int(image_grid_thw.shape[0]):
        raise ValueError(
            f"RAW frontend image count mismatch: {len(flat_images)} images vs "
            f"{int(image_grid_thw.shape[0])} image_grid_thw rows"
        )

    frontend_param = next(raw_frontend.parameters(), None)
    device = frontend_param.device if frontend_param is not None else qwen_inputs["pixel_values"].device
    raw = torch.stack([_as_raw_frontend_tensor(image) for image in flat_images], dim=0).to(device=device)
    input_format = str(_cfg_get(raw_frontend_cfg, "input_format", "rgb_raw"))
    lighting_domains = None
    if raw_frontend_metadata is not None:
        normalized_samples = [_normalize_image_batch(sample) for sample in batch_images]
        if len(raw_frontend_metadata) != len(normalized_samples):
            raise ValueError("raw_frontend_metadata must have one entry per batch sample")
        lighting_domains = []
        for sample, metadata in zip(normalized_samples, raw_frontend_metadata):
            domain = "" if metadata is None else str(metadata.get("lighting_domain", ""))
            lighting_domains.extend([domain] * len(sample))
    rgb = raw_frontend(
        raw,
        input_format=input_format,
        lighting_domains=lighting_domains,
    ).to(dtype=torch.float32)
    assert_rgb_float01(rgb, name="RAW frontend policy output")

    pixel_device = qwen_inputs["pixel_values"].device
    image_processor = getattr(qwen_vl_interface.processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("Qwen processor has no image_processor; cannot pack tensor RAW frontend outputs.")
    pixel_dtype = qwen_inputs["pixel_values"].dtype
    packed = [
        _pack_one_qwen_image(
            rgb[idx],
            image_grid_thw[idx],
            image_processor=image_processor,
            dtype=pixel_dtype,
        )
        for idx in range(rgb.shape[0])
    ]
    qwen_inputs["pixel_values"] = torch.cat(packed, dim=0).to(device=pixel_device)
    return qwen_inputs


def build_qwen_inputs_with_raw_frontend(
    *,
    qwen_vl_interface: nn.Module,
    images: Sequence[Sequence[Any]],
    instructions: Sequence[str],
    raw_images: Sequence[Sequence[Any]] | None = None,
    raw_frontend: nn.Module | None = None,
    raw_frontend_cfg: Any = None,
    raw_frontend_metadata: Sequence[dict[str, Any] | None] | None = None,
    **kwargs: Any,
) -> dict:
    processor_images = _images_for_processor_layout(images) if raw_frontend is not None else images
    qwen_inputs = qwen_vl_interface.build_qwenvl_inputs(
        images=processor_images,
        instructions=instructions,
        **kwargs,
    )
    frontend_images = raw_images if raw_images is not None else images
    return replace_qwen_pixel_values_with_raw_frontend(
        qwen_vl_interface=qwen_vl_interface,
        qwen_inputs=qwen_inputs,
        batch_images=frontend_images,
        raw_frontend=raw_frontend,
        raw_frontend_cfg=raw_frontend_cfg,
        raw_frontend_metadata=raw_frontend_metadata,
    )
