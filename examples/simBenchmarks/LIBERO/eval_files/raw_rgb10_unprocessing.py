"""Fixed three-channel RGB10 pseudo-RAW unprocessing for LIBERO.

The clean RAW10 path intentionally omits Bayer mosaicing and a camera-specific
CCM. All tasks, trajectories, checkpoints, and camera views use the same fixed
RAW parameters. The noise-level experiment below adds a fixed shot/read noise
model after lowering RAW exposure, then recovers by the inverse exposure gain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

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


def sample_raw_rgb10_params(seed: int) -> RawRGB10Params:
    """Return the experiment-wide fixed parameters.

    ``seed`` remains in the signature so existing evaluation callers and
    resume behavior stay compatible; it intentionally has no effect.
    """
    del seed
    return FIXED_RAW_RGB10_PARAMS


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


def _stable_uint32(*values: int) -> int:
    payload = ",".join(str(int(value)) for value in values).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _add_sensor_noise(
    raw10: np.ndarray,
    *,
    noise_seed: int,
    frame_index: int,
    view_index: int,
    noise: SensorNoiseParams = DEFAULT_SENSOR_NOISE,
) -> np.ndarray:
    raw = np.asarray(raw10, dtype=np.float32) / 1023.0
    seed = _stable_uint32(noise_seed, frame_index, view_index)
    rng = np.random.default_rng(seed)
    variance = np.float32(noise.shot_noise) * raw + np.float32(noise.read_noise)
    noisy = raw + rng.normal(0.0, np.sqrt(np.maximum(variance, 0.0)), raw.shape).astype(np.float32)
    return np.rint(np.clip(noisy, 0.0, 1.0) * 1023.0).astype(np.uint16)


def make_noise_level_model_input(
    rgb8: np.ndarray,
    params: RawRGB10Params,
    capture_ev: float,
    *,
    noise_seed: int,
    frame_index: int,
    view_index: int,
    noise: SensorNoiseParams = DEFAULT_SENSOR_NOISE,
) -> np.ndarray:
    """Create one float32 [0,1] default-ISP RGB input with RAW-domain noise.

    The shot/read noise parameters are fixed across levels. Higher noise levels
    come from capturing a lower RAW signal, adding the same sensor model there,
    then scaling the noisy RAW back up by ``2 ** -capture_ev`` before ISP.
    """
    raw10_capture = _rgb8_to_exposed_raw10(rgb8, params, capture_ev)
    noisy_raw10 = _add_sensor_noise(
        raw10_capture,
        noise_seed=noise_seed,
        frame_index=frame_index,
        view_index=view_index,
        noise=noise,
    )
    recovery_gain = np.float32(2.0 ** (-float(capture_ev)))
    recovered_raw10 = np.rint(
        np.clip(noisy_raw10.astype(np.float32) / 1023.0 * recovery_gain, 0.0, 1.0) * 1023.0
    ).astype(np.uint16)
    rgb8_out = _raw10_to_rgb8(recovered_raw10, params)
    return np.ascontiguousarray(rgb8_out.astype(np.float32) / 255.0, dtype=np.float32)


def _raw10_to_rgb8_with_extra_gains(
    raw10: np.ndarray,
    params: RawRGB10Params,
    extra_gains: np.ndarray,
) -> np.ndarray:
    gains = np.asarray([params.red_gain, params.green_gain, params.blue_gain], dtype=np.float32)
    gains = gains * np.asarray(extra_gains, dtype=np.float32)
    linear = np.clip(raw10.astype(np.float32) / 1023.0 * gains / params.inverse_global_gain, 0.0, 1.0)
    display = _smoothstep(_linear_to_srgb(linear))
    return np.rint(display * 255.0).astype(np.uint8)


_XYZ_TO_SRGB = np.asarray(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float32,
)
_SRGB_LUMA = np.asarray([0.2126729, 0.7151522, 0.0721750], dtype=np.float32)
_D65_UV = (0.1978, 0.4683)
_CHROMATIC_THETAS = {0, 45, 90, 135, 180, 225, 270, 315}
_UV_RADII = {
    "uv_r0p060": 0.060,
    "uv_r0p075": 0.075,
}
_RELATION_SCALES = {
    "rel_s4p5": 4.5,
    "rel_s5p5": 5.5,
}
_TONAL_SETTINGS = {
    "tonal_c0p05": 0.05,
    "tonal_c0p335": 0.3351,
    "tonal_c0p646": 0.6458,
    "tonal_c2p50": 2.50,
    "tonal_c5p00": 5.00,
    "tonal_c12p00": 12.00,
}
_TONAL_LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)

_OKLAB_M1 = np.asarray(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float32,
)
_OKLAB_M2 = np.asarray(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float32,
)
_OKLAB_M1_INV = np.asarray(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ],
    dtype=np.float32,
)
_OKLAB_M2_INV = np.asarray(
    [
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ],
    dtype=np.float32,
)


def _uv_to_xyz(up: float, vp: float, y_value: float = 1.0) -> np.ndarray:
    denom = max(float(vp), 1e-6)
    x_value = (9.0 * float(up) * y_value) / (4.0 * denom)
    z_value = ((12.0 - 3.0 * float(up) - 20.0 * float(vp)) * y_value) / (4.0 * denom)
    return np.asarray([x_value, y_value, z_value], dtype=np.float32)


def _whitepoint_gains_from_uv(up: float, vp: float) -> np.ndarray:
    target_rgb = _XYZ_TO_SRGB @ _uv_to_xyz(up, vp)
    gains = np.clip(target_rgb, 0.02, 8.0)
    return gains / max(float(np.dot(gains, _SRGB_LUMA)), 1e-6)


def uv_whitepoint_for_setting(setting: str, theta_deg: int) -> tuple[float, float]:
    if setting not in _UV_RADII:
        raise ValueError(f"Unknown uv_whitepoint setting: {setting}")
    if int(theta_deg) not in _CHROMATIC_THETAS:
        raise ValueError(f"theta_deg must be one of {sorted(_CHROMATIC_THETAS)}, got {theta_deg}")
    theta = np.deg2rad(float(theta_deg))
    radius = _UV_RADII[setting]
    return (
        float(_D65_UV[0] + radius * np.cos(theta)),
        float(_D65_UV[1] + radius * np.sin(theta)),
    )


def _linear_rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    lms = rgb @ _OKLAB_M1.T
    return np.cbrt(np.maximum(lms, 0.0)) @ _OKLAB_M2.T


def _oklab_to_linear_rgb(lab: np.ndarray) -> np.ndarray:
    lms_cbrt = lab @ _OKLAB_M2_INV.T
    return (lms_cbrt**3) @ _OKLAB_M1_INV.T


def _rotate_chroma(lab: np.ndarray, theta_deg: int, scale: float) -> np.ndarray:
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


def make_chromatic_model_input(
    rgb8: np.ndarray,
    params: RawRGB10Params,
    family: str,
    setting: str,
    theta_deg: int,
) -> np.ndarray:
    """Create one float32 [0,1] input for the chromatic experiment.

    ``uv_whitepoint`` perturbs the raw-to-RGB white-point gains, so neutral
    gray is allowed to become tinted. ``color_relation`` first reprocesses
    through the default raw-to-RGB path, then applies a gray-preserving OKLab
    chroma transform.
    """
    raw10 = _rgb8_to_exposed_raw10(rgb8, params, exposure_ev=0.0)
    theta = int(theta_deg)
    if family == "uv_whitepoint":
        up, vp = uv_whitepoint_for_setting(setting, theta)
        rgb8_out = _raw10_to_rgb8_with_extra_gains(raw10, params, _whitepoint_gains_from_uv(up, vp))
        result = rgb8_out.astype(np.float32) / 255.0
    elif family == "color_relation":
        if setting not in _RELATION_SCALES:
            raise ValueError(f"Unknown color_relation setting: {setting}")
        if theta not in _CHROMATIC_THETAS:
            raise ValueError(f"theta_deg must be one of {sorted(_CHROMATIC_THETAS)}, got {theta_deg}")
        rgb8_base = _raw10_to_rgb8(raw10, params)
        linear = _srgb_to_linear(rgb8_base.astype(np.float32) / 255.0)
        lab = _linear_rgb_to_oklab(linear)
        shifted = np.clip(_oklab_to_linear_rgb(_rotate_chroma(lab, theta, _RELATION_SCALES[setting])), 0.0, 1.0)
        result = _linear_to_srgb(shifted)
    else:
        raise ValueError(f"Unknown chromatic family: {family}")
    return np.ascontiguousarray(result, dtype=np.float32)


def _tone_map_luma(y: np.ndarray, c: float, pivot: float) -> np.ndarray:
    eps = np.float32(1e-6)
    y = np.clip(y.astype(np.float32), eps, 1.0 - eps)
    pivot = np.float32(np.clip(pivot, eps, 1.0 - eps))
    logit_y = np.log(y / (1.0 - y))
    logit_pivot = np.log(pivot / (1.0 - pivot))
    mapped = 1.0 / (1.0 + np.exp(-(logit_pivot + np.float32(c) * (logit_y - logit_pivot))))
    return np.clip(mapped, 0.0, 1.0)


def tonal_c_for_setting(setting: str) -> float:
    if setting not in _TONAL_SETTINGS:
        raise ValueError(f"Unknown tonal setting: {setting}")
    return _TONAL_SETTINGS[setting]


def make_tonal_model_input(
    rgb8: np.ndarray,
    params: RawRGB10Params,
    mode: str,
    setting: str,
    c: float,
    pivot: float,
) -> np.ndarray:
    """Create one float32 [0,1] input for the tonal-response experiment."""
    if mode != "raw_reprocess_isp_tone":
        raise ValueError(f"Unknown tonal mode: {mode}")
    expected_c = tonal_c_for_setting(setting)
    if not np.isclose(float(c), expected_c, rtol=1e-4, atol=1e-6):
        raise ValueError(f"tonal setting {setting} expects c={expected_c}, got {c}")

    raw10 = _rgb8_to_exposed_raw10(rgb8, params, exposure_ev=0.0)
    gains = np.asarray([params.red_gain, params.green_gain, params.blue_gain], dtype=np.float32)
    linear = np.clip(raw10.astype(np.float32) / 1023.0 * gains / params.inverse_global_gain, 0.0, 1.0)
    y = np.sum(linear * _TONAL_LUMA, axis=-1, keepdims=True)
    y_prime = _tone_map_luma(y, c, pivot)
    linear_prime = np.where(y > 1e-6, linear * (y_prime / np.maximum(y, 1e-6)), 0.0)
    result = _smoothstep(_linear_to_srgb(np.clip(linear_prime, 0.0, 1.0)))
    return np.ascontiguousarray(result, dtype=np.float32)


def make_ev_model_input(
    rgb8: np.ndarray,
    params: RawRGB10Params,
    exposure_ev: float,
    representation: str,
) -> np.ndarray:
    """Create one strict float32 [0,1] model input for the EV experiment."""
    raw10 = _rgb8_to_exposed_raw10(rgb8, params, exposure_ev)
    recovery_gain = np.float32(2.0 ** (-exposure_ev))
    if representation == "raw_direct":
        result = raw10.astype(np.float32) / 1023.0
    elif representation == "raw_recovered":
        result = np.clip(raw10.astype(np.float32) / 1023.0 * recovery_gain, 0.0, 1.0)
    else:
        rgb_direct8 = _raw10_to_rgb8(raw10, params)
        if representation == "rgb_direct":
            result = rgb_direct8.astype(np.float32) / 255.0
        elif representation == "rgb_recovered":
            display = rgb_direct8.astype(np.float32) / 255.0
            linear = _srgb_to_linear(_inverse_smoothstep(display))
            recovered = _smoothstep(_linear_to_srgb(np.clip(linear * recovery_gain, 0.0, 1.0)))
            result = np.rint(recovered * 255.0).astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unknown EV representation: {representation}")
    return np.ascontiguousarray(result, dtype=np.float32)


def quantize_image_to_float_bits(image: np.ndarray, bits: int) -> np.ndarray:
    """Quantize an image value grid while keeping the model input float32 [0, 1]."""
    if bits <= 0:
        arr = np.asarray(image)
        if arr.dtype == np.uint8:
            return np.ascontiguousarray(arr.astype(np.float32) / 255.0, dtype=np.float32)
        return np.ascontiguousarray(arr.astype(np.float32), dtype=np.float32)
    if bits > 8:
        raise ValueError(f"Expected image quantization bits in [1, 8], got {bits}")

    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        x = arr.astype(np.float32) / 255.0
    else:
        x = arr.astype(np.float32)
    levels = np.float32((1 << bits) - 1)
    quantized = np.rint(np.clip(x, 0.0, 1.0) * levels) / levels
    return np.ascontiguousarray(quantized, dtype=np.float32)


def unprocess_rgb8_to_raw_rgb10_view8(
    rgb8: np.ndarray,
    params: RawRGB10Params,
) -> np.ndarray:
    """Return an HWC uint8 view of quantized three-channel linear RGB10 RAW.

    The uint8 mapping is required by the existing frozen VLA image processors:
    RGB10 values are mapped linearly via round(raw10 * 255 / 1023), without
    white balance, display gamma, tone mapping, or auto contrast.
    """
    rgb = np.asarray(rgb8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB input, got {rgb.shape}")

    x = _srgb_to_linear(_inverse_smoothstep(rgb.astype(np.float32) / 255.0))
    inverse_gains = params.inverse_global_gain / np.asarray(
        [params.red_gain, params.green_gain, params.blue_gain],
        dtype=np.float32,
    )
    raw = np.clip(x * inverse_gains, 0.0, 1.0)

    raw10 = np.rint(raw * 1023.0).astype(np.uint16)
    return np.rint(raw10.astype(np.float32) * (255.0 / 1023.0)).astype(np.uint8)


def params_dict(params: RawRGB10Params) -> dict[str, float]:
    return asdict(params)


def episode_view_seed(base_seed: int, task_id: int, episode_idx: int, view_idx: int) -> int:
    # Explicit arithmetic avoids Python's randomized hash and remains stable
    # across processes, hosts, and resumed runs.
    return int(base_seed) * 1_000_003 + int(task_id) * 10_007 + int(episode_idx) * 101 + int(view_idx)
