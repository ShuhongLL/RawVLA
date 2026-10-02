"""RAWVLA-Bench lighting controls for RoboTwin 2.0 / SAPIEN.

The five domains mirror the calibrated LIBERO setup.  The three strategies
make the cumulative parts of that setup independently testable:

``environment_only``
    Scale RoboTwin's original ambient, directional, and point lights.
``environment_plus_extra_lights``
    Also enable a fixed camera-side key/flood rig.
``full``
    Also apply RAW full-well pressure in the camera pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from typing import Iterable

import numpy as np


LIGHTING_DOMAINS = ("ExtremeLow", "Low", "Normal", "Over", "ExtremeOver")
LIGHTING_STRATEGIES = ("environment_only", "environment_plus_extra_lights", "full")


@dataclass(frozen=True)
class LightingDomain:
    ev_range: tuple[float, float]
    extra_light_range: tuple[float, float] = (0.0, 0.0)
    sensor_saturation_ev_range: tuple[float, float] = (0.0, 0.0)

    def at(self, sample_unit: float = 0.5) -> tuple[float, float, float]:
        unit = float(np.clip(sample_unit, 0.0, 1.0))
        interpolate = lambda bounds: float(bounds[0] + unit * (bounds[1] - bounds[0]))
        return interpolate(self.ev_range), interpolate(self.extra_light_range), interpolate(
            self.sensor_saturation_ev_range
        )


LIGHTING_DOMAIN_CONFIGS = {
    # SAPIEN retains substantially more visible indirect/background signal
    # than MuJoCo at the LIBERO negative-EV values.  These two ranges are
    # RoboTwin-calibrated so their midpoints produce clearly separated dark
    # (about 0.10 mean luma) and extreme-dark (about 0.06) observations.
    "ExtremeLow": LightingDomain((-9.0, -8.5)),
    "Low": LightingDomain((-7.0, -6.0)),
    "Normal": LightingDomain((-0.5, 0.5)),
    # Positive EV is now calibrated against the renderer's pre-tonemap HDR
    # buffer.  Color clips for the displayed RGB, while the float RAW sensor
    # retains several stops of headroom.
    "Over": LightingDomain((1.0, 1.5), (0.05, 0.15), (0.15, 0.3)),
    "ExtremeOver": LightingDomain((2.0, 2.5), (0.25, 0.45), (0.4, 0.7)),
}

# RoboTwin's head camera is at approximately (-0.032, -0.45, 1.35).  The
# extra rig is fixed in world coordinates and illuminates a 3 x 3 tabletop
# grid, matching the topology of LIBERO's common_agentview_key rig.
EXTRA_LIGHT_ORIGIN = np.asarray([-0.032, -0.45, 1.35], dtype=np.float32)
TABLE_FLOOD_TARGETS = tuple(
    (x, y, 0.75) for x in (-0.45, 0.0, 0.45) for y in (-0.25, 0.0, 0.25)
)
KEY_LIGHT_BASE_INTENSITY = 0.02


def _colors(lights: Iterable[object]) -> list[np.ndarray]:
    return [np.asarray(light.get_color(), dtype=np.float32).copy() for light in lights]


def install_extra_light_rig(task: object) -> list[object]:
    """Install the fixed SAPIEN key/flood slots once, initially switched off."""
    if hasattr(task, "rawvla_extra_light_lst"):
        return task.rawvla_extra_light_lst

    lights = []
    key_direction = np.asarray([0.0, 0.0, 0.75], dtype=np.float32) - EXTRA_LIGHT_ORIGIN
    key_direction /= max(float(np.linalg.norm(key_direction)), 1e-6)
    lights.append(
        task.scene.add_spot_light(
            EXTRA_LIGHT_ORIGIN.tolist(), key_direction.tolist(), np.deg2rad(45), np.deg2rad(90), [0, 0, 0], shadow=False
        )
    )
    for target in TABLE_FLOOD_TARGETS:
        direction = np.asarray(target, dtype=np.float32) - EXTRA_LIGHT_ORIGIN
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        lights.append(
            task.scene.add_spot_light(
                EXTRA_LIGHT_ORIGIN.tolist(), direction.tolist(), np.deg2rad(45), np.deg2rad(90), [0, 0, 0], shadow=False
            )
        )
    task.rawvla_extra_light_lst = lights
    return lights


def capture_lighting_baseline(task: object) -> None:
    if hasattr(task, "_rawvla_lighting_baseline"):
        return
    task._rawvla_lighting_baseline = {
        "ambient": np.asarray(task.scene.ambient_light, dtype=np.float32).copy(),
        "directional": _colors(task.direction_light_lst),
        "point": _colors(task.point_light_lst),
    }


def apply_lighting(
    task: object,
    domain: str,
    strategy: str = "full",
    *,
    sample_unit: float = 0.5,
) -> dict[str, float | str]:
    """Apply one five-domain setting without rebuilding or moving the scene."""
    if domain not in LIGHTING_DOMAIN_CONFIGS:
        raise ValueError(f"Unknown lighting domain {domain!r}; expected one of {LIGHTING_DOMAINS}")
    if strategy not in LIGHTING_STRATEGIES:
        raise ValueError(f"Unknown lighting strategy {strategy!r}; expected one of {LIGHTING_STRATEGIES}")

    capture_lighting_baseline(task)
    extra_lights = install_extra_light_rig(task)
    lighting_ev, extra_intensity, saturation_ev = LIGHTING_DOMAIN_CONFIGS[domain].at(sample_unit)
    scale = float(2.0**lighting_ev)
    baseline = task._rawvla_lighting_baseline
    task.scene.set_ambient_light((baseline["ambient"] * scale).tolist())
    for light, color in zip(task.direction_light_lst, baseline["directional"]):
        light.set_color((color * scale).tolist())
    for light, color in zip(task.point_light_lst, baseline["point"]):
        light.set_color((color * scale).tolist())

    use_extra = strategy in {"environment_plus_extra_lights", "full"}
    key_intensity = KEY_LIGHT_BASE_INTENSITY * scale if use_extra else 0.0
    flood_intensity = extra_intensity if use_extra else 0.0
    extra_lights[0].set_color([key_intensity] * 3)
    for light in extra_lights[1:]:
        light.set_color([flood_intensity] * 3)

    applied_saturation_ev = saturation_ev if strategy == "full" else 0.0
    if hasattr(task, "cameras"):
        task.cameras.sensor_saturation_ev = applied_saturation_ev
        # Reset on each controlled lighting capture so all five domains use
        # the same deterministic Gaussian sample; EV changes only its SNR.
        task.cameras.raw_sensor_noise_frame_index = 0
    task.rawvla_lighting_metadata = {
        "domain": domain,
        "strategy": strategy,
        "sample_unit": float(sample_unit),
        "lighting_ev": lighting_ev,
        "lighting_scale": scale,
        "extra_light_intensity": flood_intensity,
        "sensor_saturation_ev": applied_saturation_ev,
        "raw_white_level": float(getattr(getattr(task, "cameras", None), "hdr_raw_white_level", 0.0)),
        "raw_sensor_shot_noise": float(
            getattr(getattr(getattr(task, "cameras", None), "raw_sensor_noise", None), "shot_noise", 0.0)
        ),
        "raw_sensor_read_noise": float(
            getattr(getattr(getattr(task, "cameras", None), "raw_sensor_noise", None), "read_noise", 0.0)
        ),
    }
    return dict(task.rawvla_lighting_metadata)


def _paired_sample_unit(task: object, seed: int, domain: str) -> float:
    benchmark_seed = int(os.environ.get("ROBOTWIN_LIGHTING_BENCHMARK_SEED", "20260824"))
    task_name = os.environ.get("ROBOTWIN_ACTIVE_TASK", task.__class__.__name__)
    text = "\x1f".join(
        str(part) for part in ("lighting_ev", benchmark_seed, "robotwin2", task_name, int(seed), domain)
    )
    stable_seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")
    return float(np.random.default_rng(stable_seed).random())


def _paired_noise_seed(task: object, seed: int, domain: str) -> int:
    benchmark_seed = int(os.environ.get("ROBOTWIN_LIGHTING_BENCHMARK_SEED", "20260824"))
    task_name = os.environ.get("ROBOTWIN_ACTIVE_TASK", task.__class__.__name__)
    text = "\x1f".join(
        str(part) for part in ("raw_noise", benchmark_seed, "robotwin2", task_name, int(seed), domain)
    )
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")


def apply_lighting_from_env(task: object, *, seed: int = 0) -> dict[str, float | str] | None:
    """Apply a setting when ``ROBOTWIN_LIGHTING_DOMAIN`` is configured."""
    domain = os.environ.get("ROBOTWIN_LIGHTING_DOMAIN", "").strip()
    if not domain:
        return None
    strategy = os.environ.get("ROBOTWIN_LIGHTING_STRATEGY", "full").strip()
    sample_text = os.environ.get("ROBOTWIN_LIGHTING_SAMPLE_UNIT", "0.5").strip().lower()
    paired = sample_text == "paired"
    sample_unit = _paired_sample_unit(task, seed, domain) if paired else float(sample_text)
    metadata = apply_lighting(task, domain, strategy, sample_unit=sample_unit)
    if paired and hasattr(task, "cameras"):
        task.cameras.raw_sensor_noise_seed = _paired_noise_seed(task, seed, domain)
        task.cameras.raw_sensor_noise_frame_index = 0
        metadata["noise_seed"] = int(task.cameras.raw_sensor_noise_seed)
        task.rawvla_lighting_metadata = dict(metadata)
    return metadata
