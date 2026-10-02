from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def _ensure_rawvla_bench_on_path() -> None:
    root = Path(__file__).resolve().parents[3]
    rawvla_bench = root / "benchmark" / "rawvla-bench"
    if rawvla_bench.exists() and str(rawvla_bench) not in sys.path:
        sys.path.insert(0, str(rawvla_bench))


_ensure_rawvla_bench_on_path()

from rawvla_bench.lighting import DEFAULT_EV_RANGES, LIGHTING_DOMAINS, lighting_parameters_for_sample  # noqa: E402
from rawvla_bench.raw import make_rawvla_observation, stable_uint32  # noqa: E402


@dataclass(frozen=True)
class RawVLALightSample:
    domain: str
    lighting_ev: float
    lighting_scale: float
    lighting_rig: str
    agentview_table_flood: float
    sensor_saturation_ev: float
    noise_seed: int


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def is_enabled(cfg: Any) -> bool:
    value = cfg_get(cfg_get(cfg, "rawvla_light", None), "enabled", False)
    return value not in (False, "False", "false", "0", 0, None)


def _cfg_list(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        if value.lower() in {"all", "*"}:
            return default
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part) for part in value)


def sample_trajectory_lighting(
    *,
    dataset_name: str,
    trajectory_id: int,
    epoch: int,
    cfg: Any,
) -> RawVLALightSample:
    raw_cfg = cfg_get(cfg, "rawvla_light", None)
    seed = int(cfg_get(raw_cfg, "seed", 20260813))
    domains = _cfg_list(cfg_get(raw_cfg, "domains", None), LIGHTING_DOMAINS)
    if not domains:
        raise ValueError("datasets.vla_data.rawvla_light.domains cannot be empty")
    for domain in domains:
        if domain not in DEFAULT_EV_RANGES:
            raise ValueError(f"Unknown RAWVLA-Light domain {domain!r}; valid domains are {LIGHTING_DOMAINS}")

    rng = np.random.default_rng(stable_uint32("train_lighting", seed, dataset_name, int(trajectory_id), int(epoch)))
    domain = str(domains[int(rng.integers(0, len(domains)))])
    ev_low, ev_high = DEFAULT_EV_RANGES[domain]
    sample_unit = float(rng.random())
    lighting_ev = float(ev_low + sample_unit * (ev_high - ev_low))
    params = lighting_parameters_for_sample(domain, lighting_ev, sample_unit=sample_unit)
    noise_seed = stable_uint32("train_raw_noise", seed, dataset_name, int(trajectory_id), int(epoch), domain)
    return RawVLALightSample(
        domain=domain,
        lighting_ev=float(params["lighting_ev"]),
        lighting_scale=float(params["lighting_scale"]),
        lighting_rig=str(params["lighting_rig"]),
        agentview_table_flood=float(params["agentview_table_flood"]),
        sensor_saturation_ev=float(params["sensor_saturation_ev"]),
        noise_seed=int(noise_seed),
    )


def to_rgb8_array(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"))
    elif isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        arr = tensor.numpy()
    else:
        arr = np.asarray(image)

    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {arr.shape}")

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    arr = arr.astype(np.float32, copy=False)
    if float(np.nanmax(arr)) <= 1.0:
        arr = arr * 255.0
    return np.ascontiguousarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))


def to_float32_image(image: Any) -> np.ndarray:
    rgb8 = to_rgb8_array(image)
    return np.ascontiguousarray(rgb8.astype(np.float32) / 255.0)


def resize_float32_image(image: np.ndarray, height: int, width: int, resample: int = Image.BILINEAR) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC float32 image, got shape {arr.shape}")
    if arr.shape[:2] == (height, width):
        return np.ascontiguousarray(arr)
    channels = [
        np.asarray(Image.fromarray(arr[..., index], mode="F").resize((width, height), resample=resample))
        for index in range(arr.shape[-1])
    ]
    return np.ascontiguousarray(np.stack(channels, axis=-1).astype(np.float32, copy=False))


def make_raw_float32_image(
    image: Any,
    *,
    light: RawVLALightSample,
    frame_index: int,
    view_index: int,
    representation: str = "raw",
) -> np.ndarray:
    rgb8 = to_rgb8_array(image)
    raw_like = make_rawvla_observation(
        rgb8,
        noise_seed=light.noise_seed,
        frame_index=int(frame_index),
        view_index=int(view_index),
        representation=representation,
        sensor_saturation_ev=light.sensor_saturation_ev,
        exposure_ev=0.0,
    )
    return np.ascontiguousarray(raw_like.astype(np.float32) / 255.0)


def metadata_dict(light: RawVLALightSample) -> dict[str, Any]:
    return {
        "benchmark": "rawvla-bench-light-libero-v1-train",
        "lighting_domain": light.domain,
        "lighting_ev": light.lighting_ev,
        "lighting_scale": light.lighting_scale,
        "lighting_rig": light.lighting_rig,
        "agentview_table_flood": light.agentview_table_flood,
        "sensor_saturation_ev": light.sensor_saturation_ev,
        "noise_seed": light.noise_seed,
    }
