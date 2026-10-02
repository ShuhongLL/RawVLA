"""RAW observation formation for RAWVLA-Bench."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .lighting import SENSOR_SATURATION_FORMULA


def _ensure_starvla_on_path() -> None:
    root = Path(__file__).resolve().parents[3]
    starvla = root / "starVLA"
    if starvla.exists() and str(starvla) not in sys.path:
        sys.path.insert(0, str(starvla))


_ensure_starvla_on_path()

try:
    from examples.simBenchmarks.LIBERO.eval_files.raw_rgb10_unprocessing import (  # noqa: E402
        FIXED_RAW_RGB10_PARAMS,
        _raw10_to_rgb8,
        _rgb8_to_exposed_raw10,
        params_dict,
    )
except ModuleNotFoundError:
    @dataclass(frozen=True)
    class RawRGB10Params:
        red_gain: float
        green_gain: float
        blue_gain: float
        inverse_global_gain: float

    FIXED_RAW_RGB10_PARAMS = RawRGB10Params(
        red_gain=1.8,
        green_gain=1.0,
        blue_gain=1.7,
        inverse_global_gain=1.0,
    )

    def params_dict(params: RawRGB10Params) -> dict[str, float]:
        return asdict(params)

    def _inverse_smoothstep(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0)
        return 0.5 - np.sin(np.arcsin(1.0 - 2.0 * x) / 3.0)

    def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
        return np.where(
            x <= 0.04045,
            x / 12.92,
            np.power((x + 0.055) / 1.055, 2.4),
        )

    def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0)
        return np.where(
            x <= 0.0031308,
            12.92 * x,
            1.055 * np.power(x, 1.0 / 2.4) - 0.055,
        )

    def _smoothstep(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _rgb8_to_exposed_raw10(rgb8: np.ndarray, params: RawRGB10Params, exposure_ev: float) -> np.ndarray:
        rgb = np.asarray(rgb8)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"Expected HWC uint8 RGB input, got shape={rgb.shape}, dtype={rgb.dtype}")
        linear = _srgb_to_linear(_inverse_smoothstep(rgb.astype(np.float32) / 255.0))
        inverse_gains = params.inverse_global_gain / np.asarray(
            [params.red_gain, params.green_gain, params.blue_gain], dtype=np.float32
        )
        raw = np.clip(linear * inverse_gains * np.float32(2.0**exposure_ev), 0.0, 1.0)
        return np.rint(raw * 1023.0).astype(np.uint16)

    def _raw10_to_rgb8(raw10: np.ndarray, params: RawRGB10Params) -> np.ndarray:
        gains = np.asarray([params.red_gain, params.green_gain, params.blue_gain], dtype=np.float32)
        linear = np.clip(raw10.astype(np.float32) / 1023.0 * gains / params.inverse_global_gain, 0.0, 1.0)
        display = _smoothstep(_linear_to_srgb(linear))
        return np.rint(display * 255.0).astype(np.uint8)


@dataclass(frozen=True)
class SensorNoiseParams:
    shot_noise: float = 4.0e-4
    read_noise: float = 1.0e-5


DEFAULT_SENSOR_NOISE = SensorNoiseParams()


def stable_uint32(*parts: object) -> int:
    text = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def raw10_to_view8(raw10: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(raw10, dtype=np.float32) * (255.0 / 1023.0)).astype(np.uint8)


def apply_full_well_saturation(raw10: np.ndarray, sensor_saturation_ev: float) -> np.ndarray:
    """Apply continuous RAW-domain exposure pressure and full-well clipping."""
    if float(sensor_saturation_ev) <= 0.0:
        return np.asarray(raw10)
    raw = np.asarray(raw10, dtype=np.float32) / 1023.0
    saturated = np.clip(raw * np.float32(2.0 ** float(sensor_saturation_ev)), 0.0, 1.0)
    return np.rint(saturated * 1023.0).astype(np.uint16)


def add_sensor_noise(
    raw10: np.ndarray,
    *,
    noise_seed: int,
    frame_index: int,
    view_index: int,
    noise: SensorNoiseParams = DEFAULT_SENSOR_NOISE,
) -> np.ndarray:
    raw = np.asarray(raw10, dtype=np.float32) / 1023.0
    seed = stable_uint32("frame_noise", noise_seed, frame_index, view_index)
    rng = np.random.default_rng(seed)
    variance = np.maximum(np.float32(noise.shot_noise) * raw + np.float32(noise.read_noise), 0.0)
    noisy = np.clip(raw + rng.normal(0.0, np.sqrt(variance), size=raw.shape).astype(np.float32), 0.0, 1.0)
    return np.rint(noisy * 1023.0).astype(np.uint16)


def make_rawvla_observation(
    rgb8: np.ndarray,
    *,
    noise_seed: int,
    frame_index: int,
    view_index: int,
    representation: str,
    sensor_saturation_ev: float = 0.0,
    exposure_ev: float = 0.0,
    noise: SensorNoiseParams = DEFAULT_SENSOR_NOISE,
) -> np.ndarray:
    """Convert rendered RGB8 to RAWVLA-Bench model input.

    ``representation="raw"`` returns the noisy three-channel RAW10 linear view
    as uint8. ``representation="default_isp"`` reprocesses that same noisy RAW10
    through the fixed default ISP from the current LIBERO unprocessing module.
    """
    raw10_clean = _rgb8_to_exposed_raw10(rgb8, FIXED_RAW_RGB10_PARAMS, exposure_ev=float(exposure_ev))
    raw10_clean = apply_full_well_saturation(raw10_clean, sensor_saturation_ev)
    raw10_noisy = add_sensor_noise(
        raw10_clean,
        noise_seed=noise_seed,
        frame_index=frame_index,
        view_index=view_index,
        noise=noise,
    )
    if representation == "raw":
        return raw10_to_view8(raw10_noisy)
    if representation == "default_isp":
        return _raw10_to_rgb8(raw10_noisy, FIXED_RAW_RGB10_PARAMS)
    raise ValueError(f"Unknown RAWVLA-Bench representation: {representation!r}")


def make_rawvla_preview_pair(
    rgb8: np.ndarray,
    *,
    noise_seed: int,
    frame_index: int = 0,
    view_index: int = 0,
    sensor_saturation_ev: float = 0.0,
    noise: SensorNoiseParams = DEFAULT_SENSOR_NOISE,
) -> tuple[np.ndarray, np.ndarray]:
    raw_view = make_rawvla_observation(
        rgb8,
        noise_seed=noise_seed,
        frame_index=frame_index,
        view_index=view_index,
        representation="raw",
        sensor_saturation_ev=sensor_saturation_ev,
        noise=noise,
    )
    default_isp = make_rawvla_observation(
        rgb8,
        noise_seed=noise_seed,
        frame_index=frame_index,
        view_index=view_index,
        representation="default_isp",
        sensor_saturation_ev=sensor_saturation_ev,
        noise=noise,
    )
    return raw_view, default_isp


def unprocess_metadata() -> dict[str, object]:
    return {
        "raw_params": params_dict(FIXED_RAW_RGB10_PARAMS),
        "sensor_noise": asdict(DEFAULT_SENSOR_NOISE),
        "sensor_saturation": {
            "domain": "pseudo_raw_before_noise",
            "function": SENSOR_SATURATION_FORMULA,
        },
        "raw_bit_depth": 10,
        "raw_representation": "three_channel_linear_rgb10",
    }
