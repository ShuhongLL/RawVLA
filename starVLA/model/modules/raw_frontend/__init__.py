"""Trainable RAW frontend modules for StarVLA."""

from .registry import build_raw_frontend, get_raw_frontend_aux_losses, is_raw_frontend_enabled
from .image_contract import as_rgb_chw_float01, assert_rgb_float01
from .qwen_tensor_vision import (
    apply_raw_frontend_to_images,
    build_qwen_inputs_with_raw_frontend,
    select_raw_frontend_images,
)

__all__ = [
    "apply_raw_frontend_to_images",
    "as_rgb_chw_float01",
    "assert_rgb_float01",
    "build_qwen_inputs_with_raw_frontend",
    "build_raw_frontend",
    "get_raw_frontend_aux_losses",
    "is_raw_frontend_enabled",
    "select_raw_frontend_images",
]
