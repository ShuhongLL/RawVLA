"""Benchmark-independent image contracts for trainable RAW frontends.

The public boundary between a RAW/ISP frontend and every downstream policy is
RGB, channel-first, float32 in [0, 1].  Policy-specific normalization belongs
inside the policy adapter and must never leak back into a benchmark dataset.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def assert_rgb_float01(image: Tensor, *, name: str = "policy image") -> Tensor:
    """Validate the differentiable RAW-frontend output without detaching it."""

    if not torch.is_tensor(image):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(image)!r}")
    if image.dtype != torch.float32:
        raise TypeError(f"{name} must be float32, got {image.dtype}")
    if image.ndim < 3 or image.shape[-3] != 3:
        raise ValueError(f"{name} must be RGB channel-first [..., 3, H, W], got {tuple(image.shape)}")
    detached = image.detach()
    if not bool(torch.isfinite(detached).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    minimum = float(detached.amin().cpu())
    maximum = float(detached.amax().cpu())
    tolerance = 1.0e-6
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError(f"{name} must be in [0, 1], got min={minimum:.8g}, max={maximum:.8g}")
    return image


def as_rgb_chw_float01(image: Any, *, name: str = "input image") -> Tensor:
    """Canonicalize ordinary RGB input; floating input is already [0, 1].

    Integer/PIL images are converted from the conventional uint8 [0, 255]
    representation.  Floating tensors are never guessed or divided by 255:
    accepting a float [0, 255] array would hide a broken caller contract.
    """

    if isinstance(image, Image.Image):
        tensor = torch.from_numpy(np.asarray(image.convert("RGB")).copy())
    elif isinstance(image, np.ndarray):
        tensor = torch.from_numpy(np.asarray(image).copy())
    elif torch.is_tensor(image):
        tensor = image
    else:
        raise TypeError(f"Unsupported {name} type: {type(image)!r}")
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have 3 dimensions, got {tuple(tensor.shape)}")
    if tensor.shape[0] in {1, 3, 4} and tensor.shape[-1] not in {1, 3, 4}:
        chw = tensor[:3]
    elif tensor.shape[-1] in {1, 3, 4}:
        chw = tensor.permute(2, 0, 1)[:3]
    else:
        raise ValueError(f"Cannot identify RGB channel axis for {name} shape {tuple(tensor.shape)}")
    if chw.shape[0] == 1:
        chw = chw.repeat(3, 1, 1)
    if chw.is_floating_point():
        chw = chw.to(dtype=torch.float32)
    else:
        if chw.dtype != torch.uint8:
            raise TypeError(f"Integer {name} must be uint8, got {chw.dtype}")
        chw = chw.to(dtype=torch.float32).div(255.0)
    return assert_rgb_float01(chw.contiguous(), name=name)
