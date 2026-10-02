"""Bayer packing helpers for Dark-ISP.

The Dark-ISP paper feeds demosaiced Bayer RAW as four planes. For an RGGB
sensor those planes are ordered as ``[R, Gr, B, Gb]``.
"""

from __future__ import annotations

from typing import Final

import numpy as np


PACKED_ORDER: Final[tuple[str, str, str, str]] = ("R", "Gr", "B", "Gb")
_PATTERN_OFFSETS: Final[dict[str, dict[str, tuple[int, int]]]] = {
    "RGGB": {"R": (0, 0), "Gr": (0, 1), "Gb": (1, 0), "B": (1, 1)},
    "BGGR": {"B": (0, 0), "Gb": (0, 1), "Gr": (1, 0), "R": (1, 1)},
    "GRBG": {"Gr": (0, 0), "R": (0, 1), "B": (1, 0), "Gb": (1, 1)},
    "GBRG": {"Gb": (0, 0), "B": (0, 1), "R": (1, 0), "Gr": (1, 1)},
}


def _pattern_offsets(pattern: str) -> dict[str, tuple[int, int]]:
    key = pattern.upper()
    if key not in _PATTERN_OFFSETS:
        raise ValueError(f"Unsupported Bayer pattern {pattern!r}; expected one of {sorted(_PATTERN_OFFSETS)}")
    return _PATTERN_OFFSETS[key]


def numpy_mosaic_to_packed4(mosaic: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Pack an HxW Bayer mosaic into H/2 x W/2 x 4 planes.

    Args:
        mosaic: A single-channel Bayer mosaic with shape ``[H, W]`` or
            ``[..., H, W]``. Height and width must be even.
        pattern: Bayer pattern of the first 2x2 tile.

    Returns:
        Array with shape ``[..., H/2, W/2, 4]`` ordered as ``[R, Gr, B, Gb]``.
    """

    arr = np.asarray(mosaic)
    if arr.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions, got shape={arr.shape}")
    height, width = arr.shape[-2:]
    if height % 2 or width % 2:
        raise ValueError(f"Bayer mosaic height and width must be even, got {(height, width)}")

    offsets = _pattern_offsets(pattern)
    planes = [arr[..., y::2, x::2] for channel in PACKED_ORDER for y, x in [offsets[channel]]]
    return np.stack(planes, axis=-1)


def numpy_packed4_to_mosaic(packed: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Unpack ``[..., H, W, 4]`` planes into a single-channel Bayer mosaic."""

    arr = np.asarray(packed)
    if arr.ndim < 3 or arr.shape[-1] != 4:
        raise ValueError(f"Expected packed shape [..., H, W, 4], got {arr.shape}")

    offsets = _pattern_offsets(pattern)
    out = np.empty((*arr.shape[:-3], arr.shape[-3] * 2, arr.shape[-2] * 2), dtype=arr.dtype)
    for index, channel in enumerate(PACKED_ORDER):
        y, x = offsets[channel]
        out[..., y::2, x::2] = arr[..., index]
    return out
