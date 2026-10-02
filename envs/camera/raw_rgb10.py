"""Fixed three-channel RGB10 pseudo-RAW utilities for RoboTwin.

This mirrors the LIBERO raw_rgb10_unprocessing defaults so RoboTwin camera
observations can expose raw-style inputs and the matching default ISP output.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os

import numpy as np


@dataclass(frozen=True)
class RawRGB10Params:
    red_gain: float
    green_gain: float
    blue_gain: float
    inverse_global_gain: float


@dataclass(frozen=True)
class SensorNoiseParams:
    shot_noise: float = 4.0e-4
    read_noise: float = 1.0e-5


FIXED_RAW_RGB10_PARAMS = RawRGB10Params(
    red_gain=1.8,
    green_gain=1.0,
    blue_gain=1.7,
    inverse_global_gain=1.0,
)
DEFAULT_SENSOR_NOISE = SensorNoiseParams()
_NOISE_CALL_INDEX = 0

CHROMATIC_THETAS = {0, 45, 90, 135, 180, 225, 270, 315}
UV_RADII = {
    "uv_r0p075": 0.075,
}
D65_UV = (0.1978, 0.4683)
SRGB_LUMA = np.asarray([0.2126729, 0.7151522, 0.0721750], dtype=np.float32)
TONAL_LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
XYZ_TO_SRGB = np.asarray(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float32,
)
OKLAB_M1 = np.asarray(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float32,
)
OKLAB_M2 = np.asarray(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float32,
)
OKLAB_M1_INV = np.asarray(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ],
    dtype=np.float32,
)
OKLAB_M2_INV = np.asarray(
    [
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ],
    dtype=np.float32,
)


def inverse_smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return 0.5 - np.sin(np.arcsin(1.0 - 2.0 * x) / 3.0)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(
        x <= 0.04045,
        x / 12.92,
        np.power((x + 0.055) / 1.055, 2.4),
    )


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * np.power(x, 1.0 / 2.4) - 0.055,
    )


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def rgb_to_raw10(
    rgb_image: np.ndarray,
    params: RawRGB10Params = FIXED_RAW_RGB10_PARAMS,
    exposure_ev: float = 0.0,
) -> np.ndarray:
    rgb = np.asarray(rgb_image)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB input, got shape={rgb.shape}, dtype={rgb.dtype}")
    if rgb.dtype == np.uint8:
        display = rgb.astype(np.float32) / 255.0
    else:
        display = np.clip(rgb.astype(np.float32), 0.0, 1.0)

    linear = srgb_to_linear(inverse_smoothstep(display))
    inverse_gains = params.inverse_global_gain / np.asarray(
        [params.red_gain, params.green_gain, params.blue_gain],
        dtype=np.float32,
    )
    raw = np.clip(linear * inverse_gains * np.float32(2.0**exposure_ev), 0.0, 1.0)
    return np.rint(raw * 1023.0).astype(np.uint16)


def rgb8_to_raw10(
    rgb8: np.ndarray,
    params: RawRGB10Params = FIXED_RAW_RGB10_PARAMS,
    exposure_ev: float = 0.0,
) -> np.ndarray:
    return rgb_to_raw10(rgb8, params=params, exposure_ev=exposure_ev)


def raw10_to_rgb_float(
    raw10: np.ndarray,
    params: RawRGB10Params = FIXED_RAW_RGB10_PARAMS,
) -> np.ndarray:
    """Run the default ISP and return HWC float32 RGB in [0, 1]."""
    raw = np.asarray(raw10)
    if raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError(f"Expected HWC three-channel RAW10 input, got shape={raw.shape}")

    gains = np.asarray([params.red_gain, params.green_gain, params.blue_gain], dtype=np.float32)
    linear = np.clip(raw.astype(np.float32) / 1023.0 * gains / params.inverse_global_gain, 0.0, 1.0)
    display = smoothstep(linear_to_srgb(linear))
    return np.clip(display, 0.0, 1.0).astype(np.float32, copy=False)


def raw10_to_rgb8(
    raw10: np.ndarray,
    params: RawRGB10Params = FIXED_RAW_RGB10_PARAMS,
) -> np.ndarray:
    """Display-only conversion of the default ISP output to RGB8."""
    return float_to_u8(raw10_to_rgb_float(raw10, params=params))


def raw10_to_rgb_float_with_extra_gains(
    raw10: np.ndarray,
    extra_gains: np.ndarray,
    params: RawRGB10Params = FIXED_RAW_RGB10_PARAMS,
) -> np.ndarray:
    """Run the gain-adjusted ISP and return HWC float32 RGB in [0, 1]."""
    gains = np.asarray([params.red_gain, params.green_gain, params.blue_gain], dtype=np.float32)
    gains = gains * np.asarray(extra_gains, dtype=np.float32)
    linear = np.clip(raw10.astype(np.float32) / 1023.0 * gains / params.inverse_global_gain, 0.0, 1.0)
    display = smoothstep(linear_to_srgb(linear))
    return np.clip(display, 0.0, 1.0).astype(np.float32, copy=False)


def raw10_to_rgb8_with_extra_gains(
    raw10: np.ndarray,
    extra_gains: np.ndarray,
    params: RawRGB10Params = FIXED_RAW_RGB10_PARAMS,
) -> np.ndarray:
    """Display-only conversion of the gain-adjusted ISP output to RGB8."""
    return float_to_u8(raw10_to_rgb_float_with_extra_gains(raw10, extra_gains, params=params))


def raw10_to_view8(raw10: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw10)
    if raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError(f"Expected HWC three-channel RAW10 input, got shape={raw.shape}")
    return np.rint(np.clip(raw.astype(np.float32), 0.0, 1023.0) * (255.0 / 1023.0)).astype(np.uint8)


def apply_full_well_saturation(raw10: np.ndarray, sensor_saturation_ev: float) -> np.ndarray:
    """Apply LIBERO-compatible RAW-domain full-well pressure and clipping."""
    if float(sensor_saturation_ev) <= 0.0:
        return np.asarray(raw10)
    raw = np.asarray(raw10, dtype=np.float32) / np.float32(1023.0)
    saturated = np.clip(raw * np.float32(2.0 ** float(sensor_saturation_ev)), 0.0, 1.0)
    return np.rint(saturated * np.float32(1023.0)).astype(np.uint16)


def hdr_linear_to_raw_float(
    hdr_rgb: np.ndarray,
    white_level: float,
    params: RawRGB10Params = FIXED_RAW_RGB10_PARAMS,
) -> np.ndarray:
    """Map pre-tonemap HDR radiance to an unquantized 3-channel sensor RAW.

    ``HdrColor`` is already linear, so unlike the legacy Color-buffer
    unprocessing path this performs no inverse gamma or inverse tone mapping.
    The renderer's display white-balance gains are removed to approximate
    three separate sensor channels; consequently a direct preview has the
    expected green cast. ``white_level`` is the sensor full-well value in
    renderer-radiance units.
    """
    hdr = np.asarray(hdr_rgb, dtype=np.float32)
    if hdr.ndim != 3 or hdr.shape[-1] != 3:
        raise ValueError(f"Expected HWC linear HDR RGB input, got shape={hdr.shape}")
    if not np.isfinite(white_level) or float(white_level) <= 0.0:
        raise ValueError(f"white_level must be finite and positive, got {white_level}")
    inverse_wb = np.float32(params.inverse_global_gain) / np.asarray(
        [params.red_gain, params.green_gain, params.blue_gain], dtype=np.float32
    )
    return np.clip(hdr * inverse_wb / np.float32(white_level), 0.0, 1.0).astype(
        np.float32, copy=False
    )


def apply_full_well_saturation_float(raw: np.ndarray, sensor_saturation_ev: float) -> np.ndarray:
    """Apply full-well pressure to an unquantized float RAW image."""
    image = np.asarray(raw, dtype=np.float32)
    if float(sensor_saturation_ev) <= 0.0:
        return image
    return np.clip(image * np.float32(2.0 ** float(sensor_saturation_ev)), 0.0, 1.0).astype(
        np.float32, copy=False
    )


def float_to_u8(image: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(image.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)


def quantize_float_image(image: np.ndarray, bits: int) -> np.ndarray:
    """Apply bit-depth quantization while preserving float32 [0, 1] transport."""
    x = image.astype(np.float32) / np.float32(255.0) if image.dtype == np.uint8 else image.astype(np.float32)
    x = np.clip(x, 0.0, 1.0)
    if bits <= 0:
        return x.astype(np.float32, copy=False)
    if bits > 8:
        raise ValueError(f"Expected bit depth in [1, 8], got {bits}")
    levels = np.float32((1 << bits) - 1)
    return (np.rint(x * levels) / levels).astype(np.float32, copy=False)


def ev_perturb(raw10_clean: np.ndarray, exposure_ev: float, representation: str) -> np.ndarray:
    raw_exposed = np.clip(
        raw10_clean.astype(np.float32) / 1023.0 * np.float32(2.0**exposure_ev),
        0.0,
        1.0,
    )
    raw10_exposed = np.rint(raw_exposed * 1023.0).astype(np.uint16)
    recovery_gain = np.float32(2.0 ** (-exposure_ev))

    if representation == "raw_direct":
        return (raw10_exposed.astype(np.float32) / np.float32(1023.0)).astype(np.float32, copy=False)
    if representation == "raw_recovered":
        recovered = np.clip(raw10_exposed.astype(np.float32) / 1023.0 * recovery_gain, 0.0, 1.0)
        return recovered.astype(np.float32, copy=False)

    rgb_direct = raw10_to_rgb_float(raw10_exposed)
    if representation == "rgb_direct":
        return rgb_direct
    if representation == "rgb_recovered":
        linear = srgb_to_linear(inverse_smoothstep(rgb_direct))
        recovered = smoothstep(linear_to_srgb(np.clip(linear * recovery_gain, 0.0, 1.0)))
        return np.clip(recovered, 0.0, 1.0).astype(np.float32, copy=False)
    raise ValueError(f"Unknown EV representation: {representation}")


def _stable_uint32(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _next_noise_frame_index() -> int:
    global _NOISE_CALL_INDEX
    value = _NOISE_CALL_INDEX
    _NOISE_CALL_INDEX += 1
    return value


def add_sensor_noise(
    raw_capture: np.ndarray,
    seed: int,
    noise: SensorNoiseParams = DEFAULT_SENSOR_NOISE,
) -> np.ndarray:
    raw = np.clip(raw_capture.astype(np.float32), 0.0, 1.0)
    variance = np.maximum(np.float32(noise.shot_noise) * raw + np.float32(noise.read_noise), 0.0)
    rng = np.random.default_rng(int(seed))
    noisy = raw + rng.normal(0.0, np.sqrt(variance), size=raw.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def noise_level_perturb(
    raw10_clean: np.ndarray,
    capture_ev: float,
    noise_seed: int,
    frame_index: int | None = None,
    view_index: int = 0,
    noise: SensorNoiseParams = DEFAULT_SENSOR_NOISE,
) -> np.ndarray:
    frame = _next_noise_frame_index() if frame_index is None else int(frame_index)
    raw_capture = np.clip(
        raw10_clean.astype(np.float32) / 1023.0 * np.float32(2.0**capture_ev),
        0.0,
        1.0,
    )
    seed = _stable_uint32(noise_seed, frame, view_index, f"{capture_ev:g}")
    raw_noisy = add_sensor_noise(raw_capture, seed=seed, noise=noise)
    raw_recovered = np.clip(raw_noisy * np.float32(2.0 ** (-capture_ev)), 0.0, 1.0)
    return raw10_to_rgb_float(np.rint(raw_recovered * 1023.0).astype(np.uint16))


def uv_to_xyz(up: float, vp: float, y_value: float = 1.0) -> np.ndarray:
    denom = max(float(vp), 1e-6)
    x_value = (9.0 * float(up) * y_value) / (4.0 * denom)
    z_value = ((12.0 - 3.0 * float(up) - 20.0 * float(vp)) * y_value) / (4.0 * denom)
    return np.asarray([x_value, y_value, z_value], dtype=np.float32)


def whitepoint_gains_from_uv(up: float, vp: float) -> np.ndarray:
    target_rgb = XYZ_TO_SRGB @ uv_to_xyz(up, vp)
    gains = np.clip(target_rgb, 0.02, 8.0)
    return gains / max(float(np.dot(gains, SRGB_LUMA)), 1e-6)


def uv_whitepoint_for_setting(setting: str, theta_deg: int) -> tuple[float, float]:
    if setting not in UV_RADII:
        raise ValueError(f"Unknown uv_whitepoint setting: {setting}")
    if int(theta_deg) not in CHROMATIC_THETAS:
        raise ValueError(f"theta_deg must be one of {sorted(CHROMATIC_THETAS)}, got {theta_deg}")
    theta = np.deg2rad(float(theta_deg))
    radius = UV_RADII[setting]
    return (
        float(D65_UV[0] + radius * np.cos(theta)),
        float(D65_UV[1] + radius * np.sin(theta)),
    )


def linear_rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    lms = rgb @ OKLAB_M1.T
    return np.cbrt(np.maximum(lms, 0.0)) @ OKLAB_M2.T


def oklab_to_linear_rgb(lab: np.ndarray) -> np.ndarray:
    lms_cbrt = lab @ OKLAB_M2_INV.T
    return (lms_cbrt**3) @ OKLAB_M1_INV.T


def rotate_chroma(lab: np.ndarray, theta_deg: int, scale: float) -> np.ndarray:
    theta = np.deg2rad(float(theta_deg))
    rotation = np.asarray(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float32,
    )
    result = lab.copy()
    result[..., 1:3] = result[..., 1:3] @ (np.float32(scale) * rotation).T
    return result


def chromatic_perturb(raw10_clean: np.ndarray, family: str, setting: str, theta_deg: int) -> np.ndarray:
    if family == "uv_whitepoint":
        up, vp = uv_whitepoint_for_setting(setting, theta_deg)
        return raw10_to_rgb_float_with_extra_gains(raw10_clean, whitepoint_gains_from_uv(up, vp))
    if family == "color_relation":
        if setting != "rel_s5p0":
            raise ValueError(f"Unknown color_relation setting: {setting}")
        if int(theta_deg) not in CHROMATIC_THETAS:
            raise ValueError(f"theta_deg must be one of {sorted(CHROMATIC_THETAS)}, got {theta_deg}")
        rgb_base = raw10_to_rgb_float(raw10_clean)
        linear = srgb_to_linear(rgb_base)
        lab = linear_rgb_to_oklab(linear)
        shifted = np.clip(oklab_to_linear_rgb(rotate_chroma(lab, int(theta_deg), 5.0)), 0.0, 1.0)
        return np.clip(linear_to_srgb(shifted), 0.0, 1.0).astype(np.float32, copy=False)
    raise ValueError(f"Unknown chromatic family: {family}")


def tone_map_luma(y: np.ndarray, c: float, pivot: float = 0.18) -> np.ndarray:
    eps = np.float32(1e-6)
    y = np.clip(y.astype(np.float32), eps, 1.0 - eps)
    pivot = np.float32(np.clip(pivot, eps, 1.0 - eps))
    logit_y = np.log(y / (1.0 - y))
    logit_pivot = np.log(pivot / (1.0 - pivot))
    mapped = 1.0 / (1.0 + np.exp(-(logit_pivot + np.float32(c) * (logit_y - logit_pivot))))
    return np.clip(mapped, 0.0, 1.0)


def tonal_perturb(raw10_clean: np.ndarray, c: float, pivot: float = 0.18) -> np.ndarray:
    gains = np.asarray(
        [FIXED_RAW_RGB10_PARAMS.red_gain, FIXED_RAW_RGB10_PARAMS.green_gain, FIXED_RAW_RGB10_PARAMS.blue_gain],
        dtype=np.float32,
    )
    linear = np.clip(
        raw10_clean.astype(np.float32) / 1023.0 * gains / FIXED_RAW_RGB10_PARAMS.inverse_global_gain,
        0.0,
        1.0,
    )
    y = np.sum(linear * TONAL_LUMA, axis=-1, keepdims=True)
    y_prime = tone_map_luma(y, c, pivot)
    linear_prime = np.where(y > 1e-6, linear * (y_prime / np.maximum(y, 1e-6)), 0.0)
    return np.clip(
        smoothstep(linear_to_srgb(np.clip(linear_prime, 0.0, 1.0))), 0.0, 1.0
    ).astype(np.float32, copy=False)


def apply_isp_perturbation_from_env(raw10_clean: np.ndarray, default_isp: np.ndarray) -> np.ndarray:
    axis = os.environ.get("ROBOTWIN_ISP_AXIS", "none").strip().lower()
    if axis in {"", "none", "default", "default_isp"}:
        return default_isp
    if axis == "bitdepth":
        return quantize_float_image(default_isp, int(os.environ["ROBOTWIN_ISP_BITS"]))
    if axis == "ev":
        return ev_perturb(
            raw10_clean,
            float(os.environ["ROBOTWIN_ISP_EV"]),
            os.environ["ROBOTWIN_ISP_EV_REPRESENTATION"],
        )
    if axis == "noise_level":
        frame_index_env = os.environ.get("ROBOTWIN_NOISE_FRAME_INDEX")
        return noise_level_perturb(
            raw10_clean,
            float(os.environ["ROBOTWIN_NOISE_CAPTURE_EV"]),
            int(os.environ.get("ROBOTWIN_RAW_SENSOR_NOISE_SEED", "1695213855")),
            None if frame_index_env is None else int(frame_index_env),
            int(os.environ.get("ROBOTWIN_NOISE_VIEW_INDEX", "0")),
            SensorNoiseParams(
                shot_noise=float(os.environ.get("ROBOTWIN_NOISE_SHOT", DEFAULT_SENSOR_NOISE.shot_noise)),
                read_noise=float(os.environ.get("ROBOTWIN_NOISE_READ", DEFAULT_SENSOR_NOISE.read_noise)),
            ),
        )
    if axis == "chromatic":
        return chromatic_perturb(
            raw10_clean,
            os.environ["ROBOTWIN_CHROMATIC_FAMILY"],
            os.environ["ROBOTWIN_CHROMATIC_SETTING"],
            int(os.environ["ROBOTWIN_CHROMATIC_THETA"]),
        )
    if axis == "tonal":
        return tonal_perturb(raw10_clean, float(os.environ["ROBOTWIN_TONAL_C"]))
    raise ValueError(f"Unknown ROBOTWIN_ISP_AXIS={axis!r}")


def unprocess_rgb8_to_raw10_and_default_isp(
    rgb8: np.ndarray,
    params: RawRGB10Params = FIXED_RAW_RGB10_PARAMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return RAW10, a display-only uint8 raw view, and float32 default ISP."""

    raw10 = rgb8_to_raw10(rgb8, params=params, exposure_ev=0.0)
    raw_view8 = raw10_to_view8(raw10)
    default_isp = raw10_to_rgb_float(raw10, params=params)
    return raw10, raw_view8, default_isp


def unprocess_rgb_to_raw10_and_default_isp(
    rgb_image: np.ndarray,
    params: RawRGB10Params = FIXED_RAW_RGB10_PARAMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return RAW10, a display-only uint8 raw view, and float32 default ISP.

    ``rgb_image`` can be either uint8 RGB or a float RGB image in [0, 1].
    Passing SAPIEN's float color texture preserves information before the final
    RAW10 quantization instead of first collapsing it to RGB8.
    """

    raw10 = rgb_to_raw10(rgb_image, params=params, exposure_ev=0.0)
    raw_view8 = raw10_to_view8(raw10)
    default_isp = raw10_to_rgb_float(raw10, params=params)
    return raw10, raw_view8, default_isp
