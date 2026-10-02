"""Fixed ColorChecker-calibrated ISP for the D-Robotics RAW cameras.

The stored PNG/video values are treated as three-channel linear RAW samples.
There are deliberately only two camera profiles: ``agentview`` and ``wrist``.
Both wrist cameras resolve to the shared ``wrist`` profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_CONFIG = Path(__file__).with_name("default_isp.json")

VIEW_ALIASES = {
    "agent": "agentview",
    "agentview": "agentview",
    "head": "agentview",
    "head_camera": "agentview",
    "wrist": "wrist",
    "left_wrist": "wrist",
    "left_wrist_camera": "wrist",
    "right_wrist": "wrist",
    "right_wrist_camera": "wrist",
}


def load_isp_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_view(view: str) -> str:
    key = view.lower().replace("observation.images.", "")
    try:
        return VIEW_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown camera view {view!r}; expected one of {sorted(VIEW_ALIASES)}") from exc


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    linear = np.maximum(np.asarray(linear, dtype=np.float32), 0.0)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def apply_default_isp(
    raw_rgb: np.ndarray,
    view: str,
    *,
    config: dict[str, Any] | None = None,
    input_max: float | None = None,
    output_dtype: np.dtype[Any] | type[np.uint8] = np.uint8,
) -> np.ndarray:
    """Apply the fixed default ISP to an ``H x W x 3`` RAW RGB image.

    Integer input is normalized by its dtype maximum unless ``input_max`` is
    supplied. Floating-point input is assumed to be in [0, 1]. The returned
    image is uint8 by default; pass ``output_dtype=np.float32`` for [0, 1]
    floating-point output.
    """

    image = np.asarray(raw_rgb)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB input, got shape={image.shape}")

    if input_max is None:
        input_max = float(np.iinfo(image.dtype).max) if np.issubdtype(image.dtype, np.integer) else 1.0
    if input_max <= 0:
        raise ValueError(f"input_max must be positive, got {input_max}")

    cfg = load_isp_config() if config is None else config
    profile_name = resolve_view(view)
    profile = cfg["profiles"][profile_name]

    linear = image.astype(np.float32) / np.float32(input_max)
    black = np.asarray(profile.get("black_level", [0.0, 0.0, 0.0]), dtype=np.float32)
    linear = np.clip((linear - black) / np.maximum(1.0 - black, 1.0e-6), 0.0, 1.0)

    white_balance = np.asarray(profile["white_balance"], dtype=np.float32)
    exposure_gain = np.float32(profile["exposure_gain"])
    color_matrix = np.asarray(profile["color_matrix"], dtype=np.float32)
    corrected = np.einsum("...c,oc->...o", linear * white_balance * exposure_gain, color_matrix)
    display = np.clip(linear_to_srgb(corrected), 0.0, 1.0)

    dtype = np.dtype(output_dtype)
    if np.issubdtype(dtype, np.floating):
        return display.astype(dtype)
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f"Unsupported output dtype {dtype}")
    return np.rint(display * np.iinfo(dtype).max).astype(dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--view", required=True, choices=sorted(VIEW_ALIASES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    raw = np.asarray(Image.open(args.input).convert("RGB"))
    result = apply_default_isp(raw, args.view, config=load_isp_config(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(args.output)


if __name__ == "__main__":
    main()
